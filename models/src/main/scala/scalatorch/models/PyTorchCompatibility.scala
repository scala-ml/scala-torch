package scalatorch.models

import scalatorch.io.PyTorchArchiveLimits

/** Explicit worker compatibility setting for archives saved without CRC32 records. */
private[models] object PyTorchCompatibility:
  def limits(base: PyTorchArchiveLimits,env: Map[String,String]=sys.env): PyTorchArchiveLimits =
    base.copy(allowMissingCrc=env.get("SDFX_SCALATORCH_ALLOW_MISSING_CRC").contains("1"))
