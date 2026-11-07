package scalatorch.models.worker

import scalatorch.Tensor
import scalatorch.models.*
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import java.util.Base64
import javax.imageio.ImageIO

object NativeIntermediateRecorder:
  import NativeJson.*

  private def hookFields(hooksOpt: Option[NativeJson], name: String): Option[Map[String, NativeJson]] =
    hooksOpt match
      case Some(Obj(hooks)) =>
        hooks.get(name).flatMap {
          case Null | Bool(false) => None
          case Obj(fields) if fields.get("enabled").contains(Bool(false)) => None
          case Bool(true) => Some(Map.empty[String, NativeJson])
          case Obj(fields) => Some(fields)
          case _ => throw IllegalArgumentException(s"invalid $name hook")
        }
      case _ => None

  private def str(fields: Map[String, NativeJson], key: String, default: String): String =
    fields.get(key) match
      case None | Some(Null) => default
      case Some(Str(v)) => v
      case _ => throw IllegalArgumentException(s"$key must be a string")

  private def num(fields: Map[String, NativeJson], key: String, default: Double): Double =
    fields.get(key) match
      case None | Some(Null) => default
      case Some(Number(v)) => v.toDouble
      case Some(Str(s)) =>
        try s.trim.toDouble
        catch case _: NumberFormatException => throw IllegalArgumentException(s"$key must be numeric")
      case _ => throw IllegalArgumentException(s"$key must be numeric")

  private def intVal(fields: Map[String, NativeJson], key: String, default: Int): Int =
    fields.get(key) match
      case None | Some(Null) => default
      case Some(Number(v)) if v.isValidInt => v.toInt
      case Some(Str(s)) =>
        try s.trim.toInt
        catch case _: NumberFormatException => throw IllegalArgumentException(s"$key must be an integer")
      case _ => throw IllegalArgumentException(s"$key must be an integer")

  private def boolVal(fields: Map[String, NativeJson], key: String, default: Boolean): Boolean =
    fields.get(key) match
      case None | Some(Null) => default
      case Some(Bool(v)) => v
      case Some(Number(v)) => v != 0
      case Some(Str(s)) => Set("1", "true", "yes", "on").contains(s.trim.toLowerCase(java.util.Locale.ROOT))
      case _ => default

  def parseSaveIntermediate(hooksOpt: Option[NativeJson]): Option[SaveIntermediateConfig] =
    hookFields(hooksOpt, "save_intermediate").map { fields =>
      val allowed = Set(
        "enabled", "every_n", "start_at", "stop_at", "image_type", "mode", "final_save",
        "format", "codec", "fps", "add_first_frames", "add_last_frames", "smooth", "seconds",
        "lores", "hires", "ffmpeg_bat", "bat_only", "output_dir"
      )
      require(fields.keySet.subsetOf(allowed), "unknown save_intermediate option")
      val raw = SaveIntermediateConfig(
        everyN = intVal(fields, "every_n", 5),
        startAt = intVal(fields, "start_at", 0),
        stopAt = intVal(fields, "stop_at", 0),
        imageType = str(fields, "image_type", "Denoised"),
        mode = str(fields, "mode", "Standard"),
        finalSave = boolVal(fields, "final_save", false),
        format = str(fields, "format", "mp4"),
        codec = str(fields, "codec", "h264"),
        fps = num(fields, "fps", 2.0),
        addFirstFrames = intVal(fields, "add_first_frames", 0),
        addLastFrames = intVal(fields, "add_last_frames", 0),
        smooth = boolVal(fields, "smooth", false),
        seconds = num(fields, "seconds", 0.0),
        lores = boolVal(fields, "lores", true),
        hires = boolVal(fields, "hires", true),
        ffmpegBat = boolVal(fields, "ffmpeg_bat", false),
        batOnly = boolVal(fields, "bat_only", false),
        outputDirOverride = fields.get("output_dir").collect { case Str(s) if s.trim.nonEmpty => s.trim }
      )
      SaveIntermediateConfig.validate(raw)
    }

  def parseStepsAnimation(
      hooksOpt: Option[NativeJson],
      batchSize: Int,
      nIter: Int,
      hasSaveIntermediate: Boolean
  ): Option[StepsAnimationConfig] =
    hookFields(hooksOpt, "steps_animation").map { fields =>
      require(batchSize == 1 && nIter == 1, "steps animation requires batch size and count equal to one")
      require(!hasSaveIntermediate, "steps animation cannot run together with Save intermediate images")
      val allowed = Set(
        "enabled", "duration", "fps", "codec", "interpolation", "skip_initial",
        "debug", "run_incomplete", "delete_intermediate", "create_output", "frames_dir", "output_dir"
      )
      require(fields.keySet.subsetOf(allowed), "unknown steps_animation option")
      val raw = StepsAnimationConfig(
        duration = num(fields, "duration", 5.0),
        fps = intVal(fields, "fps", 30),
        codec = str(fields, "codec", "h264"),
        interpolation = str(fields, "interpolation", "blend"),
        skipInitial = intVal(fields, "skip_initial", 0),
        debug = boolVal(fields, "debug", false),
        runIncomplete = boolVal(fields, "run_incomplete", false),
        deleteIntermediate = boolVal(fields, "delete_intermediate", false),
        createOutput = boolVal(fields, "create_output", false),
        framesDir = str(fields, "frames_dir", "steps-animation/frames"),
        outputDir = str(fields, "output_dir", "steps-animation/videos")
      )
      StepsAnimationConfig.validate(raw)
    }

  def parseImg2ImgAlt(hooksOpt: Option[NativeJson]): Option[Img2ImgAltConfig] =
    hookFields(hooksOpt, "img2imgalt").map { fields =>
      val allowed = Set("enabled", "override_prompt", "override_negative", "decode_cfg_scale", "randomness", "inversion_steps")
      require(fields.keySet.subsetOf(allowed), "unknown img2imgalt option")
      val cfg = Img2ImgAltConfig(
        overridePrompt = str(fields, "override_prompt", ""),
        overrideNegative = str(fields, "override_negative", ""),
        decodeCfgScale = num(fields, "decode_cfg_scale", 1.0),
        randomness = num(fields, "randomness", 0.0),
        inversionSteps = intVal(fields, "inversion_steps", 0)
      )
      require(cfg.decodeCfgScale.isFinite, "img2imgalt decode_cfg_scale must be finite")
      require(cfg.randomness.isFinite && cfg.randomness >= 0.0 && cfg.randomness <= 1.0, "img2imgalt randomness must be in [0, 1]")
      require(cfg.inversionSteps >= 0 && cfg.inversionSteps <= 1000, "img2imgalt inversion_steps must be in 0..1000")
      cfg
    }

  def parseCdTuner(hooksOpt: Option[NativeJson]): Option[CdTunerConfig] =
    hookFields(hooksOpt, "cd_tuner").map { fields =>
      val allowed = Set(
        "enabled", "d1", "d2", "con1", "con2", "bri", "col1", "col2", "col3",
        "hd1", "hd2", "hrs", "st1", "st2", "dis", "sat", "sat2",
        "split_ratio", "split_by", "colors", "map_stop_step", "map_strength", "apply_once"
      )
      require(fields.keySet.subsetOf(allowed), "unknown cd_tuner option")
      val hrsVal = fields.get("hrs") match
        case Some(Bool(b)) => if b then 1.0 else 0.0
        case _ => num(fields, "hrs", 0.0)
      val cfg = CdTunerConfig(
        d1 = num(fields, "d1", 0.0),
        d2 = num(fields, "d2", 0.0),
        con1 = num(fields, "con1", 0.0),
        con2 = num(fields, "con2", 0.0),
        bri = num(fields, "bri", 0.0),
        col1 = num(fields, "col1", 0.0),
        col2 = num(fields, "col2", 0.0),
        col3 = num(fields, "col3", 0.0),
        hd1 = num(fields, "hd1", 0.0),
        hd2 = num(fields, "hd2", 0.0),
        hrs = hrsVal,
        st1 = intVal(fields, "st1", -1),
        st2 = intVal(fields, "st2", -1),
        dis = intVal(fields, "dis", 0),
        sat = num(fields, "sat", 0.0),
        sat2 = num(fields, "sat2", 0.0),
        splitRatio = str(fields, "split_ratio", "1,1"),
        splitBy = str(fields, "split_by", "Horizontal"),
        colors = str(fields, "colors", ""),
        mapStopStep = intVal(fields, "map_stop_step", 2),
        mapStrength = num(fields, "map_strength", 1.0),
        applyOnce = boolVal(fields, "apply_once", false)
      )
      CdTuner.validate(cfg)
    }

  def parseLlul(hooksOpt: Option[NativeJson]): Option[LlulConfig] =
    hookFields(hooksOpt, "llul").map { fields =>
      val allowed = Set(
        "enabled", "weight", "multiply", "x", "y", "layers", "apply_to",
        "start_steps", "max_steps", "up", "up_aa", "down", "down_aa", "intp", "force_float", "mask"
      )
      require(fields.keySet.subsetOf(allowed), s"LLuL: unknown settings ${fields.keySet -- allowed}")
      val applyTo = fields.get("apply_to") match
        case None | Some(Null) => Vector("out")
        case Some(Str(s)) => s.split(",").map(_.trim.toLowerCase(java.util.Locale.ROOT)).filter(_.nonEmpty).toVector
        case Some(Arr(items)) => items.map {
          case Str(s) => s.trim.toLowerCase(java.util.Locale.ROOT)
          case _ => throw IllegalArgumentException("LLuL: apply_to must be a list of layer kinds")
        }.toVector
        case _ => throw IllegalArgumentException("LLuL: apply_to must be a list of layer kinds")
      val layers = fields.get("layers") match
        case None | Some(Null) => Some(Vector("OUT"))
        case Some(Str(s)) if s.trim.isEmpty => None
        case Some(Str(s)) => Some(s.split(",").map(_.trim).toVector)
        case Some(Arr(items)) => Some(items.map {
          case Str(s) => s.trim
          case other => other.toString.trim
        }.toVector)
        case _ => throw IllegalArgumentException("LLuL: layers must be text")
      val xOpt = fields.get("x") match
        case None | Some(Null) | Some(Str("")) => None
        case _ => Some(num(fields, "x", 0.0))
      val yOpt = fields.get("y") match
        case None | Some(Null) | Some(Str("")) => None
        case _ => Some(num(fields, "y", 0.0))
      val maskPath = str(fields, "mask", "").trim
      val maskTensor = if maskPath.isEmpty then None else
        val p = Path.of(maskPath)
        require(Files.isRegularFile(p), s"LLuL: cannot read the mask image '$maskPath'")
        val img = DiffusionImageInput.readMask(p)
        val w = img.getWidth
        val h = img.getHeight
        val arr = new Array[Float](w * h)
        for py <- 0 until h; px <- 0 until w do
          arr(py * w + px) = img.getRaster.getSample(px, py, 0) / 255.0f
        Some(scalatorch.Tensor.fromFloats(arr, scalatorch.Shape.from(Seq(1L, 1L, h.toLong, w.toLong))))
      val rawIntp = str(fields, "intp", "lerp").trim.toLowerCase(java.util.Locale.ROOT)
      LlulConfig(
        weight = num(fields, "weight", 0.15),
        multiply = intVal(fields, "multiply", 1),
        x = xOpt,
        y = yOpt,
        layers = layers,
        applyTo = applyTo,
        startSteps = intVal(fields, "start_steps", 5),
        maxSteps = intVal(fields, "max_steps", 0),
        up = str(fields, "up", "bilinear").trim.toLowerCase(java.util.Locale.ROOT),
        upAa = boolVal(fields, "up_aa", false),
        down = str(fields, "down", "bilinear").trim.toLowerCase(java.util.Locale.ROOT),
        downAa = boolVal(fields, "down_aa", false),
        intp = if maskTensor.isDefined then "lerp" else rawIntp,
        forceFloat = boolVal(fields, "force_float", false),
        mask = maskPath,
        maskTensor = maskTensor
      )
    }

  def parseAgentAttention(hooksOpt: Option[NativeJson]): Option[AgentAttentionConfig] =
    hookFields(hooksOpt, "agent_attention").map { fields =>
      val allowed = Set(
        "enabled", "use_sp", "sp_step", "sx", "sy", "ratio", "agent_ratio",
        "sp_sx", "sp_sy", "sp_ratio", "sp_agent_ratio", "use_fp32", "max_downsample", "hires_fix_only"
      )
      require(fields.keySet.subsetOf(allowed), s"Agent Attention: unknown settings ${fields.keySet -- allowed}")
      AgentAttentionConfig(
        useSp = boolVal(fields, "use_sp", false),
        spStep = intVal(fields, "sp_step", 20),
        sx = intVal(fields, "sx", 4),
        sy = intVal(fields, "sy", 4),
        ratio = num(fields, "ratio", 0.4),
        agentRatio = num(fields, "agent_ratio", 0.95),
        spSx = intVal(fields, "sp_sx", 2),
        spSy = intVal(fields, "sp_sy", 2),
        spRatio = num(fields, "sp_ratio", 0.4),
        spAgentRatio = num(fields, "sp_agent_ratio", 0.5),
        useFp32 = boolVal(fields, "use_fp32", false),
        maxDownsample = intVal(fields, "max_downsample", 1),
        hiresFixOnly = boolVal(fields, "hires_fix_only", false)
      )
    }

  def parseIncantations(hooksOpt: Option[NativeJson]): Option[IncantationsConfig] =
    hookFields(hooksOpt, "incantations").map { fields =>
      val allowed = Set(
        "enabled", "seg_active", "seg_blur_sigma", "seg_start_step", "seg_end_step",
        "scfg_active", "scfg_scale", "scfg_rate_min", "scfg_rate_max", "scfg_rate_clamp",
        "scfg_start_step", "scfg_end_step", "scfg_r",
        "pag_active", "pag_sanf", "pag_scale", "pag_start_step", "pag_end_step",
        "cfg_interval_enable", "cfg_interval_schedule", "cfg_interval_low", "cfg_interval_high",
        "t2i0_active", "t2i0_step_start", "t2i0_step_end", "t2i0_tokens", "t2i0_window_size",
        "t2i0_correction_threshold", "t2i0_correction_strength", "t2i0_ctnms_alpha", "t2i0_ema_factor"
      )
      require(fields.keySet.subsetOf(allowed), s"Incantations: unknown settings ${fields.keySet -- allowed}")
      IncantationsConfig(
        segActive = boolVal(fields, "seg_active", false),
        segBlurSigma = num(fields, "seg_blur_sigma", 11.0),
        segStartStep = intVal(fields, "seg_start_step", 0),
        segEndStep = intVal(fields, "seg_end_step", 150),
        scfgActive = boolVal(fields, "scfg_active", false),
        scfgScale = num(fields, "scfg_scale", 1.0),
        scfgRateMin = num(fields, "scfg_rate_min", 0.8),
        scfgRateMax = num(fields, "scfg_rate_max", 3.0),
        scfgRateClamp = num(fields, "scfg_rate_clamp", 0.0),
        scfgStartStep = intVal(fields, "scfg_start_step", 0),
        scfgEndStep = intVal(fields, "scfg_end_step", 150),
        scfgR = intVal(fields, "scfg_r", 4),
        pagActive = boolVal(fields, "pag_active", false),
        pagSanf = boolVal(fields, "pag_sanf", false),
        pagScale = num(fields, "pag_scale", 0.0),
        pagStartStep = intVal(fields, "pag_start_step", 0),
        pagEndStep = intVal(fields, "pag_end_step", 150),
        cfgIntervalEnable = boolVal(fields, "cfg_interval_enable", false),
        cfgIntervalSchedule = str(fields, "cfg_interval_schedule", "Constant"),
        cfgIntervalLow = num(fields, "cfg_interval_low", 0.0),
        cfgIntervalHigh = num(fields, "cfg_interval_high", 100.0),
        t2i0Active = boolVal(fields, "t2i0_active", false),
        t2i0StepStart = intVal(fields, "t2i0_step_start", 1),
        t2i0StepEnd = intVal(fields, "t2i0_step_end", 25),
        t2i0Tokens = str(fields, "t2i0_tokens", ""),
        t2i0WindowSize = intVal(fields, "t2i0_window_size", 2),
        t2i0CorrectionThreshold = num(fields, "t2i0_correction_threshold", 0.5),
        t2i0CorrectionStrength = num(fields, "t2i0_correction_strength", 0.0),
        t2i0CtnmsAlpha = num(fields, "t2i0_ctnms_alpha", 0.1),
        t2i0EmaFactor = num(fields, "t2i0_ema_factor", 0.0)
      )
    }

  def parseCharacteristicGuidance(hooksOpt: Option[NativeJson]): Option[CharacteristicGuidanceConfig] =
    hookFields(hooksOpt, "characteristic_guidance").map { fields =>
      val allowed = Set(
        "enabled", "reg_ini", "reg_range", "ite", "noise_base", "chara_decay",
        "res", "lr", "reg_size", "reg_w", "aa_dim", "controlnet_mode", "start_step", "stop_step"
      )
      require(fields.keySet.subsetOf(allowed), s"Characteristic Guidance: unknown settings ${fields.keySet -- allowed}")
      CharacteristicGuidanceConfig(
        regIni = num(fields, "reg_ini", 1.0),
        regRange = num(fields, "reg_range", 1.0),
        ite = intVal(fields, "ite", 50),
        noiseBase = intVal(fields, "noise_base", 0),
        charaDecay = num(fields, "chara_decay", 1.0),
        res = num(fields, "res", -4.0),
        lr = num(fields, "lr", 1.0),
        regSize = num(fields, "reg_size", 0.4),
        regW = num(fields, "reg_w", 0.5),
        aaDim = intVal(fields, "aa_dim", 2),
        controlnetMode = str(fields, "controlnet_mode", "More ControlNet"),
        startStep = num(fields, "start_step", 0.0),
        stopStep = num(fields, "stop_step", 1.0)
      )
    }

  def parseClipIo(hooksOpt: Option[NativeJson]): Option[ClipIoConfig] =
    hookFields(hooksOpt, "clip_io").map { fields =>
      ClipIoConfig(
        inputEnabled = boolVal(fields, "input_enabled", false),
        inputName = str(fields, "input_name", "").trim,
        inputMode = str(fields, "input_mode", "off").trim.toLowerCase(java.util.Locale.ROOT),
        negativeInputName = str(fields, "negative_input_name", "").trim,
        negativeInputMode = str(fields, "negative_input_mode", "off").trim.toLowerCase(java.util.Locale.ROOT),
        inputFormat = str(fields, "input_format", "csv").trim.toLowerCase(java.util.Locale.ROOT),
        outputEnabled = boolVal(fields, "output_enabled", false),
        outputName = str(fields, "output_name", "").trim,
        negativeOutputName = str(fields, "negative_output_name", "").trim,
        outputFormat = str(fields, "output_format", "csv").trim.toLowerCase(java.util.Locale.ROOT),
        overwrite = boolVal(fields, "overwrite", false),
        storageDir = str(fields, "storage_dir", "").trim
      )
    }

  def parsePromptFusion(hooksOpt: Option[NativeJson]): Boolean =
    hookFields(hooksOpt, "prompt_fusion").isDefined

  def parseNeutralPrompt(hooksOpt: Option[NativeJson]): Option[NeutralPromptConfig] =
    hookFields(hooksOpt, "neutral_prompt").map { fields =>
      NeutralPromptConfig(cfgRescale = num(fields, "cfg_rescale", 0.0))
    }

  def parseComposableLora(hooksOpt: Option[NativeJson]): Option[ComposableLoraConfig] =
    hookFields(hooksOpt, "composable_lora").map { fields =>
      ComposableLoraConfig(
        enabled = boolVal(fields, "enabled", true),
        optUcTextModel = boolVal(fields, "opt_uc_text_model", false),
        optUcDiffusionModel = boolVal(fields, "opt_uc_diffusion_model", false),
        optPlotLoraWeight = boolVal(fields, "opt_plot_lora_weight", false),
        optSingleNoUc = boolVal(fields, "opt_single_no_uc", false),
        hiresGlobalStep = boolVal(fields, "hires_global_step", false),
        customTimestamp = str(fields, "custom_Timestamp", "")
      )
    }

  def parseAestheticGradients(hooksOpt: Option[NativeJson]): Option[AestheticGradientsConfig] =
    hookFields(hooksOpt, "aesthetic_gradients").map { fields =>
      AestheticGradientsConfig(
        embedding = str(fields, "embedding", "").trim,
        weight = num(fields, "weight", 0.9),
        steps = intVal(fields, "steps", 20),
        lr = num(fields, "lr", 0.0001),
        slerp = boolVal(fields, "slerp", false),
        slerpAngle = num(fields, "slerp_angle", 0.1),
        text = str(fields, "text", ""),
        negative = boolVal(fields, "negative", false)
      )
    }

  def parseSag(hooksOpt: Option[NativeJson]): Option[SagConfig] =
    hookFields(hooksOpt, "sag").map { fields =>
      val allowed = Set("enabled", "scale", "blur_sigma")
      require(fields.keySet.subsetOf(allowed), s"unknown sag option: ${fields.keySet -- allowed}")
      SagConfig(
        enabled = boolVal(fields, "enabled", true),
        scale = num(fields, "scale", 0.5),
        blurSigma = num(fields, "blur_sigma", 2.0)
      )
    }

  def parseReference(hooksOpt: Option[NativeJson]): Option[ReferenceConfig] =
    hookFields(hooksOpt, "reference").map { fields =>
      val allowed = Set("enabled", "image", "style_fidelity", "attention_auto_machine_weight")
      require(fields.keySet.subsetOf(allowed), s"unknown reference option: ${fields.keySet -- allowed}")
      ReferenceConfig(
        enabled = boolVal(fields, "enabled", true),
        image = str(fields, "image", "").trim,
        styleFidelity = num(fields, "style_fidelity", 0.5),
        attentionAutoMachineWeight = num(fields, "attention_auto_machine_weight", 1.0)
      )
    }

  def parseRevision(hooksOpt: Option[NativeJson]): Option[RevisionConfig] =
    hookFields(hooksOpt, "revision").map { fields =>
      val allowed = Set("enabled", "image", "strength", "ignore_prompt")
      require(fields.keySet.subsetOf(allowed), s"unknown revision option: ${fields.keySet -- allowed}")
      RevisionConfig(
        enabled = boolVal(fields, "enabled", true),
        image = str(fields, "image", "").trim,
        strength = num(fields, "strength", 1.0),
        ignorePrompt = boolVal(fields, "ignore_prompt", false)
      )
    }

  def parseIpAdapter(hooksOpt: Option[NativeJson]): Option[IpAdapterConfig] =
    hookFields(hooksOpt, "ip_adapter").map { fields =>
      val allowed = Set("enabled", "model", "image", "scale", "image_encoder")
      require(fields.keySet.subsetOf(allowed), s"unknown ip_adapter option: ${fields.keySet -- allowed}")
      IpAdapterConfig(
        enabled = boolVal(fields, "enabled", true),
        model = str(fields, "model", "").trim,
        image = str(fields, "image", "").trim,
        scale = num(fields, "scale", 0.7),
        imageEncoder = str(fields, "image_encoder", "").trim
      )
    }

  def parseControlLLLite(hooksOpt: Option[NativeJson]): Option[ControlLLLiteConfig] =
    hookFields(hooksOpt, "controllllite").map { fields =>
      val allowed = Set("enabled", "model", "image", "strength", "start_percent", "end_percent")
      require(fields.keySet.subsetOf(allowed), s"unknown controllllite option: ${fields.keySet -- allowed}")
      ControlLLLiteConfig(
        enabled = boolVal(fields, "enabled", true),
        model = str(fields, "model", "").trim,
        image = str(fields, "image", "").trim,
        strength = num(fields, "strength", 1.0),
        startPercent = num(fields, "start_percent", 0.0),
        endPercent = num(fields, "end_percent", 1.0)
      )
    }

  def parseFooocusInpaint(hooksOpt: Option[NativeJson]): Option[FooocusInpaintConfig] =
    hookFields(hooksOpt, "fooocus_inpaint").map { fields =>
      val allowed = Set("enabled", "model", "head", "start_percent", "end_percent")
      require(fields.keySet.subsetOf(allowed), s"unknown fooocus_inpaint option: ${fields.keySet -- allowed}")
      FooocusInpaintConfig(
        enabled = boolVal(fields, "enabled", true),
        model = str(fields, "model", "").trim,
        head = str(fields, "head", "").trim,
        startPercent = num(fields, "start_percent", 0.0),
        endPercent = num(fields, "end_percent", 1.0)
      )
    }

  def parseCompile(hooksOpt: Option[NativeJson]): Option[CompileConfig] =
    hookFields(hooksOpt, "compile").map { fields =>
      val allowed = Set("enabled", "backend", "mode", "min_block_size", "optimization_level", "cache_engines")
      require(fields.keySet.subsetOf(allowed), s"unknown compile option: ${fields.keySet -- allowed}")
      CompileConfig(
        enabled = boolVal(fields, "enabled", true),
        backend = str(fields, "backend", "inductor").trim,
        mode = str(fields, "mode", "default").trim,
        minBlockSize = intVal(fields, "min_block_size", 5),
        optimizationLevel = intVal(fields, "optimization_level", 3),
        cacheEngines = boolVal(fields, "cache_engines", true)
      )
    }

  private def scopedDirectory(root: Path, relative: String): Path =
    val normRoot = root.toAbsolutePath.normalize()
    val target = normRoot.resolve(relative).normalize()
    require(target.startsWith(normRoot), "steps animation: output directory escapes generation folder")
    Files.createDirectories(target)
    target

  final class Session(
      outputDir: Path,
      saveConfigOpt: Option[SaveIntermediateConfig],
      stepsAnimOpt: Option[StepsAnimationConfig],
      metadata: Map[String, NativeJson],
      cancelled: () => Boolean
  ):
    require(saveConfigOpt.isDefined || stepsAnimOpt.isDefined, "Session requires save_intermediate or steps_animation config")

    private val effectiveSave: SaveIntermediateConfig = saveConfigOpt.getOrElse {
      val sa = stepsAnimOpt.get
      SaveIntermediateConfig.validate(
        SaveIntermediateConfig(everyN = 1, startAt = sa.skipInitial + 1, imageType = "Denoised", mode = "Standard")
      )
    }

    val directory: Path = stepsAnimOpt match
      case Some(sa) =>
        val framesBase = scopedDirectory(outputDir, sa.framesDir)
        scopedDirectory(outputDir, sa.outputDir)
        Files.createTempDirectory(framesBase, "generation-")
      case None =>
        val base = effectiveSave.outputDirOverride.map(Path.of(_).toAbsolutePath.normalize()).getOrElse(outputDir)
        val interRoot = base.resolve("intermediates")
        Files.createDirectories(interRoot)
        Files.createTempDirectory(interRoot, "generation-")

    private var frameCounter = 0
    private val frames = scala.collection.mutable.ArrayBuffer.empty[IntermediateFrameRecord]
    private val artifacts = scala.collection.mutable.ArrayBuffer.empty[IntermediateArtifact]
    private var finished = false

    private def nextOutputPath(suffix: String): Path =
      val idx = frameCounter
      frameCounter += 1
      directory.resolve(f"$idx%05d-$suffix.png")

    private def frameToJson(f: IntermediateFrameRecord): Map[String, NativeJson] =
      Map(
        "path" -> Str(f.path),
        "seed" -> Number(BigDecimal(f.seed)),
        "pass_name" -> Str(f.passName),
        "step" -> f.step.fold[NativeJson](Null)(s => Number(s)),
        "pass_step" -> f.passStep.fold[NativeJson](Null)(s => Number(s)),
        "iteration" -> Number(f.iteration),
        "batch_index" -> Number(f.batchIndex),
        "width" -> Number(f.width),
        "height" -> Number(f.height),
        "image_type" -> Str(f.imageType)
      )

    private def artifactToJson(a: IntermediateArtifact): NativeJson =
      val base = scala.collection.mutable.LinkedHashMap[String, NativeJson](
        "type" -> Str(a.kind),
        "path" -> Str(a.path)
      )
      a.seed.foreach(v => base("seed") = Number(BigDecimal(v)))
      a.passName.foreach(v => base("pass_name") = Str(v))
      if a.kind == "image" then
        base("step") = a.step.fold[NativeJson](Null)(s => Number(s))
        base("pass_step") = a.passStep.fold[NativeJson](Null)(s => Number(s))
      a.iteration.foreach(v => base("iteration") = Number(v))
      a.batchIndex.foreach(v => base("batch_index") = Number(v))
      a.width.foreach(v => base("width") = Number(v))
      a.height.foreach(v => base("height") = Number(v))
      a.imageType.foreach(v => base("image_type") = Str(v))
      a.frames.foreach(v => base("frames") = Number(v))
      a.format.foreach(v => base("format") = Str(v))
      Obj(base.toMap)

    private def savePng(
        image: BufferedImage,
        seed: Long,
        passName: String,
        passStep: Option[Int],
        absStep: Option[Int],
        iteration: Int,
        batchIndex: Int
    ): Unit =
      if cancelled() || Thread.currentThread().isInterrupted then
        throw new java.util.concurrent.CancellationException("intermediate capture cancelled")
      val stepLabel = absStep.map(_.toString).getOrElse("final")
      val suffix =
        if effectiveSave.mode == "Suffix" then s"$seed-$passName-$stepLabel"
        else s"$passName-$stepLabel-$seed"
      val path = nextOutputPath(suffix)
      val imgType = if passName == "final" then "Final" else effectiveSave.imageType
      val record = IntermediateFrameRecord(
        path = path.toString,
        seed = seed,
        passName = passName,
        step = absStep,
        passStep = passStep,
        iteration = iteration,
        batchIndex = batchIndex,
        width = image.getWidth,
        height = image.getHeight,
        imageType = imgType
      )
      val sdfxJson = NativeJson.render(Obj(metadata ++ frameToJson(record)))
      val promptStr = metadata.get("prompt").collect { case Str(s) => s }.getOrElse("")
      val negStr = metadata.get("negative_prompt").collect { case Str(s) => s }.getOrElse("")
      val stepsVal = metadata.get("steps").collect { case Number(n) => n.toInt }.getOrElse(0)
      val samplerStr = metadata.get("sampler").collect { case Str(s) => s }.getOrElse("")
      val cfgVal = metadata.get("cfg_scale").collect { case Number(n) => n.toString }.getOrElse("7")
      val paramLines =
        promptStr + (if negStr.nonEmpty then s"\nNegative prompt: $negStr" else "") + "\n" +
          Vector(
            Some(s"Steps: $stepsVal"),
            Some(s"Sampler: $samplerStr"),
            Some(s"CFG scale: $cfgVal"),
            Some(s"Seed: $seed"),
            Some(s"Size: ${image.getWidth}x${image.getHeight}"),
            Some(s"Intermediate pass: $passName"),
            absStep.map(s => s"Intermediate step: $s"),
            Some(s"Intermediate subject: $imgType")
          ).flatten.mkString(", ")
      IntermediatePngWriter.writePngWithMetadata(
        image,
        path,
        Vector("sdfx_intermediate" -> sdfxJson, "parameters" -> paramLines)
      )
      frames += record
      artifacts += IntermediateArtifact(
        kind = "image",
        path = record.path,
        seed = Some(record.seed),
        passName = Some(record.passName),
        step = record.step,
        passStep = record.passStep,
        iteration = Some(record.iteration),
        batchIndex = Some(record.batchIndex),
        width = Some(record.width),
        height = Some(record.height),
        imageType = Some(record.imageType)
      )

    def observeStep(
        passName: String,
        step1Based: Int,
        stepOffset: Int,
        seed: Long,
        iteration: Int,
        batchIndex: Int,
        noisyLatent: Tensor[Float],
        denoisedLatent: Tensor[Float],
        nextLatent: Tensor[Float],
        decodeLatent: Tensor[Float] => BufferedImage,
        previewLatent: Tensor[Float] => String
    ): Unit =
      val absStep = stepOffset + step1Based
      if effectiveSave.isSelected(absStep) then
        val buffered =
          if effectiveSave.imageType == "Preview" then
            val b64 = previewLatent(nextLatent)
            val bytes = Base64.getDecoder.decode(b64)
            ImageIO.read(new ByteArrayInputStream(bytes))
          else
            val targetLat = if effectiveSave.imageType == "Noisy" then noisyLatent else denoisedLatent
            decodeLatent(targetLat)
        savePng(buffered, seed, passName, Some(step1Based), Some(absStep), iteration, batchIndex)

    def recordFinal(image: BufferedImage, seed: Long, iteration: Int, batchIndex: Int): Unit =
      if effectiveSave.finalSave then
        savePng(image, seed, "final", None, None, iteration, batchIndex)

    private def timeline(group: Vector[IntermediateFrameRecord]): Vector[IntermediateFrameRecord] =
      val selected = group.filter { f =>
        f.passName == "final" ||
        (f.passName == "hires" && effectiveSave.hires) ||
        (f.passName == "base" && effectiveSave.lores)
      }
      if selected.isEmpty then Vector.empty
      else
        Vector.fill(effectiveSave.addFirstFrames)(selected.last) ++ selected ++
          Vector.fill(effectiveSave.addLastFrames)(selected.last)

    private def videoTiming(count: Int): (Double, Double) =
      val duration =
        if effectiveSave.smooth then math.max(count.toDouble, effectiveSave.seconds)
        else if effectiveSave.seconds > 0.0 then effectiveSave.seconds
        else count.toDouble / effectiveSave.fps
      (count.toDouble / duration, duration)

    private def writeRecipe(part: Vector[IntermediateFrameRecord], name: String): Unit =
      val listing = directory.resolve(s"$name.ffconcat")
      val (rate, duration) = videoTiming(part.size)
      val lines = scala.collection.mutable.ArrayBuffer[String]("ffconcat version 1.0")
      part.foreach { f =>
        val fname = Path.of(f.path).getFileName.toString
        lines += s"file '$fname'"
        lines += f"duration ${1.0 / rate}%.9f"
      }
      lines += s"file '${Path.of(part.last.path).getFileName}'"
      Files.writeString(listing, lines.mkString("\n") + "\n", StandardCharsets.UTF_8)
      val bat = directory.resolve(s"$name.bat")
      val target = s"$name.${effectiveSave.format}"
      val batContent =
        s"@echo off\ncd /d \"%~dp0\"\nffmpeg -n -f concat -safe 1 -i ${listing.getFileName} -t $duration $target\n"
      Files.writeString(bat, batContent, StandardCharsets.UTF_8)
      artifacts += IntermediateArtifact(kind = "ffmpeg_recipe", path = bat.toString)
      artifacts += IntermediateArtifact(kind = "ffmpeg_playlist", path = listing.toString)

    def finish(): Vector[NativeJson] =
      if finished then return artifacts.toVector.map(artifactToJson)
      if frames.isEmpty && saveConfigOpt.isDefined then
        Files.deleteIfExists(directory)
        finished = true
        return Vector.empty
      if effectiveSave.mode == "Video" && (effectiveSave.ffmpegBat || effectiveSave.batOnly) then
        val groups = frames.toVector.map(f => (f.iteration, f.batchIndex, f.seed)).distinct.sorted
        for (iter, bIdx, sd) <- groups do
          val grp = frames.toVector.filter(f => (f.iteration, f.batchIndex, f.seed) == (iter, bIdx, sd))
          val name = s"iteration-$iter-image-$bIdx-seed-$sd"
          for passName <- Vector("base", "hires") do
            if grp.exists(_.passName == passName) then
              val part = timeline(grp.filter(f => f.passName == passName || f.passName == "final"))
              if part.nonEmpty then writeRecipe(part, s"$name-$passName")
      val manifest = directory.resolve("manifest.json")
      val manifestObj = Obj(
        Map(
          "status" -> Str("complete"),
          "metadata" -> Obj(metadata),
          "frames" -> Arr(frames.toVector.map(f => Obj(frameToJson(f)))),
          "artifacts" -> Arr(artifacts.toVector.map(artifactToJson)),
          "export_errors" -> Arr(Vector.empty)
        )
      )
      Files.writeString(manifest, NativeJson.render(manifestObj) + "\n", StandardCharsets.UTF_8)
      artifacts += IntermediateArtifact(kind = "manifest", path = manifest.toString)
      finished = true
      artifacts.toVector.map(artifactToJson)

    def closeIncomplete(): Unit =
      if !finished then
        if frames.isEmpty && saveConfigOpt.isDefined then
          try Files.deleteIfExists(directory) catch case _: Throwable => ()
        else
          try
            val manifest = directory.resolve("manifest.json")
            val manifestObj = Obj(
              Map(
                "status" -> Str("incomplete"),
                "metadata" -> Obj(metadata),
                "frames" -> Arr(frames.toVector.map(f => Obj(frameToJson(f)))),
                "artifacts" -> Arr(artifacts.toVector.map(artifactToJson))
              )
            )
            Files.writeString(manifest, NativeJson.render(manifestObj) + "\n", StandardCharsets.UTF_8)
          catch case _: Throwable => ()
        finished = true
