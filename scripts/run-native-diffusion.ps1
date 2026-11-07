param(
  [Parameter(Mandatory=$true)][string]$ModelRoot,
  [Parameter(Mandatory=$true)][string]$ClasspathFile,
  [Parameter(Mandatory=$true)][string]$SchedulerConfig,
  [Parameter(Mandatory=$true)][string]$OutputDirectory,
  [ValidateRange(8,4096)][int]$Side=512,
  [ValidateRange(1,1000)][int]$Steps=20,
  [string]$InputImage='',
  [ValidateRange(0.0,1.0)][double]$Strength=0.5,
  [switch]$VaeTiling,
  [switch]$LatentHires,
  [ValidateRange(1.01,4.0)][double]$HiresScale=1.5,
  [string]$Prompt='a photograph of a red apple on a wooden table'
)
# Direct Java acceptance harness. Explicit scheduler required; no Python invocation.
# Each run owns a new directory and child process; no timeout/restart policy.
$ErrorActionPreference='Stop'
$model=(Resolve-Path -LiteralPath $ModelRoot).Path
$config=(Resolve-Path -LiteralPath $SchedulerConfig).Path
$classpathPath=(Resolve-Path -LiteralPath $ClasspathFile).Path
$classpath=[IO.File]::ReadAllText($classpathPath).Trim()
if($classpath.Contains("`n") -or $classpath.Contains("`r")) {throw 'Classpath must be a single line'}
$output=[IO.Path]::GetFullPath($OutputDirectory)
if(Test-Path -LiteralPath $output) {throw 'Output directory must be new to preserve earlier evidence'}
$source=if($InputImage) {(Resolve-Path -LiteralPath $InputImage).Path} else {''}
# A JVM lazily loads classes/resources. Never point a long-running acceptance job
# at jars that sbt may replace while independent development/tests continue.
$entries=$classpath.Split([IO.Path]::PathSeparator)
$resolvedEntries=@(foreach($entry in $entries) {
  $sourcePath=if([IO.Path]::IsPathRooted($entry)) {$entry} else {Join-Path (Split-Path -Parent $classpathPath) $entry}
  if(-not(Test-Path -LiteralPath $sourcePath -PathType Leaf)) {throw "Acceptance classpath entry is missing: $sourcePath"}
  (Resolve-Path -LiteralPath $sourcePath).Path
})
$runtime=Join-Path $output '.runtime'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
$snapshots=@()
$manifest=@()
for($index=0;$index -lt $resolvedEntries.Length;$index++) {
  $sourcePath=$resolvedEntries[$index]
  $target=Join-Path $runtime ("{0:D3}-{1}" -f $index,[IO.Path]::GetFileName($sourcePath))
  Copy-Item -LiteralPath $sourcePath -Destination $target
  $snapshots+=$target
  $manifest+=@{source=$sourcePath;snapshot=$target;sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()}
}
$classpath=$snapshots -join [IO.Path]::PathSeparator
[IO.File]::WriteAllText("$output/runtime-manifest.json",($manifest | ConvertTo-Json -Depth 4))
$parameters=@{prompt=$Prompt;negative_prompt='';width=$Side;height=$Side;steps=$Steps;seed=42;cfg_scale=7.5;sampler_name='DPM++ 2M';scheduler='Karras'}
$kind='txt2img'
if($source) {
  $kind='img2img'
  $parameters.img2img=@{init_image=$source;denoising_strength=$Strength;resize_mode=0}
}
if($LatentHires) {
  $parameters.hires=@{enabled=$true;scale=$HiresScale;upscaler='Latent';denoising_strength=0.55;second_pass_steps=$Steps}
}
$request=@{id='generate';type=$kind;params=@{output_dir=$output;params=$parameters}} | ConvertTo-Json -Compress -Depth 8
[IO.File]::WriteAllText("$output/request.json",$request)
$info=New-Object Diagnostics.ProcessStartInfo
$info.FileName=(Get-Command java.exe).Source
$info.Arguments='-Xmx12g --enable-native-access=ALL-UNNAMED -cp "'+$classpath+'" scalatorch.models.worker.NativeWorkerMain "'+$model+'" "'+$config+'"'
$info.UseShellExecute=$false
$info.CreateNoWindow=$true
$info.RedirectStandardInput=$true
$info.RedirectStandardOutput=$true
$info.RedirectStandardError=$true
$info.EnvironmentVariables.Remove('JAVA_TOOL_OPTIONS')
foreach($component in @('UNET','VAE')) {
  $info.EnvironmentVariables["SDFX_SCALATORCH_${component}_DEVICE"]='cuda'
  $info.EnvironmentVariables["SDFX_SCALATORCH_${component}_OFFLOAD_WEIGHTS"]='true'
  $info.EnvironmentVariables["SDFX_SCALATORCH_${component}_MAX_BYTES"]='1073741824'
}
$info.EnvironmentVariables['SDFX_SCALATORCH_MAX_ATTENTION_SCORE_ELEMENTS']='134217728'
$process=[Diagnostics.Process]::Start($info)
$errors=$process.StandardError.ReadToEndAsync()
$log=New-Object IO.StreamWriter("$output/protocol.jsonl",$false)
function Read-Event {
  $line=$process.StandardOutput.ReadLine()
  if($null -eq $line) {throw 'Worker closed output before completion'}
  $log.WriteLine($line);$log.Flush()
  $event=$line | ConvertFrom-Json
  if($event.event -eq 'preview') {Write-Host "Preview step $($event.step)"} else {Write-Host $line}
  if($event.type -eq 'error') {throw "Worker error: $line"}
  return $event
}
try {
  Write-Host "Worker PID=$($process.Id)"
  $hello=Read-Event
  if($hello.type -ne 'hello') {throw 'Missing worker handshake'}
  $loadParams=@{path=$model;dtype='fp32';vram_mode='cpu';vae_tiling=[bool]$VaeTiling}
  $process.StandardInput.WriteLine((@{id='load';type='load_model';params=$loadParams} | ConvertTo-Json -Compress))
  $loaded=Read-Event
  if($loaded.type -ne 'result') {throw 'Model load did not return a result'}
  $process.StandardInput.WriteLine($request)
  do {$event=Read-Event} while($event.type -ne 'result')
  $process.StandardInput.WriteLine('{"id":"shutdown","type":"shutdown"}')
  $null=Read-Event
  $process.WaitForExit()
  if($process.ExitCode -ne 0) {throw "Worker exit $($process.ExitCode)"}
} finally {
  if(-not $process.HasExited) {$process.Kill();$process.WaitForExit()}
  [IO.File]::WriteAllText("$output/stderr.txt",$errors.GetAwaiter().GetResult())
  $log.Dispose();$process.Dispose()
}
