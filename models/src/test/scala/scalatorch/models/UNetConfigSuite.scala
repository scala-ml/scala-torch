package scalatorch.models

class UNetConfigSuite extends munit.FunSuite:
  private val base = """{"_class_name":"UNet2DConditionModel","block_out_channels":[32,64],"down_block_types":["DownBlock2D","CrossAttnDownBlock2D"],"up_block_types":["CrossAttnUpBlock2D","UpBlock2D"],"cross_attention_dim":32,"attention_head_dim":8}"""
  private def extra(s: String)=base.dropRight(1)+","+s+"}"
  test("parse supported two-stage public config and legacy head count") {
    val c=UNetConfig.fromJson(base)
    assertEquals(c.channels,Vector(32,64))
    assertEquals(c.heads,8)
    assertEquals(c.timeWidth,128)
    assertEquals(c.downAttention,Vector(false,true))
    assertEquals(c.upAttention,Vector(true,false))
    assertEquals(c.layersPerBlock,2)
    assert(UNetConfig.fromJson(extra("\"use_linear_projection\":true")).linearProjection)
  }
  test("reject unsupported semantics, duplicates and malformed values") {
    for field <- Seq("\"dropout\":0.1","\"only_cross_attention\":true","\"unknown_option\":false",
      "\"num_attention_heads\":8","\"time_embedding_type\":\"fourier\"",
      "\"norm_eps\":0","\"layers_per_block\":0","\"layers_per_block\":1.5",
      "\"attention_head_dim\":3","\"in_channels\":2147483647","\"norm_num_groups\":3",
      "\"freq_shift\":16","\"block_out_channels\":[32]","\"cross_attention_dim\":32") do
      intercept[IllegalArgumentException](UNetConfig.fromJson(extra(field)))
    for json <- Seq(base+"{}","[]",base.dropRight(2),"{"+" "*1048576+"}") do
      intercept[IllegalArgumentException](UNetConfig.fromJson(json))
  }

  test("reject incompatible dimensions independently of duplicate detection") {
    for json <- Vector(base.replace("[32,64]","[32]"),base.replace("[32,64]","[33,64]"),
      base.replace("[32,64]","[]"),base.replace("\"attention_head_dim\":8","\"attention_head_dim\":3"),
      extra("\"transformer_layers_per_block\":17"),extra("\"time_embedding_dim\":0"),
      extra("\"mid_block_scale_factor\":1e-100"),extra("\"sample_size\":[1,2,3]"),
      extra("\"addition_embed_type\":\"text_time\"")) do
      intercept[IllegalArgumentException](UNetConfig.fromJson(json))
  }
