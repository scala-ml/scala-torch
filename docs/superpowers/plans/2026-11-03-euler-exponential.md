# NUI Euler Exponential

1. Extend the independent NUI/Diffusers Euler schedule fixture to include Exponential at 1, 3, 4 and 9 steps; add a failing scala-torch comparison and worker integration coverage.
2. Implement the exponential sigma schedule and log-sigma timestep inversion in scala-torch. Route Euler Exponential through both decoded and latent Hires passes, including img2img Hires overrides. Reject unsupported sampler combinations explicitly.
3. Verify the fixtures, worker integration and existing Euler/Hires regression suites. Record the scope and limitations.
