// Offline reference fixture generator only; not a Scala build or runtime dependency.
use tokenizers::{Tokenizer,AddedToken};
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::{Sequence,NFC,Lowercase,Replace};
use tokenizers::pre_tokenizers::{sequence::Sequence as PreSequence,split::{Split,SplitPattern},byte_level::ByteLevel};
use tokenizers::processors::roberta::RobertaProcessing;
use tokenizers::SplitDelimiterBehavior;
use tokenizers::normalizers::replace::ReplacePattern;
fn main() -> Result<(),Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String>=std::env::args().collect();
    let model=BPE::from_file(&format!("{}/vocab.json",args[1]),&format!("{}/merges.txt",args[1]))
        .end_of_word_suffix("</w>".into()).unk_token("<|endoftext|>".into()).build()?;
    let mut tokenizer=Tokenizer::new(model);
    tokenizer.with_normalizer(Some(Sequence::new(vec![NFC.into(),Replace::new(ReplacePattern::Regex(r"\s+".into())," ")?.into(),Lowercase.into()])));
    tokenizer.with_pre_tokenizer(Some(PreSequence::new(vec![
        Split::new(SplitPattern::Regex(r"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+".into()),SplitDelimiterBehavior::Removed,true)?.into(),
        ByteLevel::default().add_prefix_space(false).into()])));
    tokenizer.with_post_processor(Some(RobertaProcessing::new(("<|endoftext|>".into(),1),("<|startoftext|>".into(),0)).add_prefix_space(false).trim_offsets(false)));
    tokenizer.add_special_tokens(&[AddedToken::from("<|startoftext|>",true).normalized(true),AddedToken::from("<|endoftext|>",true).normalized(true)]);
    let prompts=["", "A red cat on a blue chair.", "  HELLO\tworld\n", "can't I'LL we're 12345", "Café cafe\u{301} Straße ΟΣ οσ", "猫と犬 🐈🙂", "&amp; &lt;b&gt;", "<|startoftext|>cat<|endoftext|>", "<|ENDOFTEXT|>", "a\u{00a0}b\u{2003}c", "İ I ı ﬃ", "foo\u{0000}bar", "éàñ🎉", "...!!!??", "a—b – c"];
    let mut cases=Vec::new();
    for prompt in prompts {
        let encoding=tokenizer.encode(prompt,true)?;
        cases.push(serde_json::json!({"text":prompt,"ids":encoding.get_ids()}));
    }
    std::fs::write(&args[2],serde_json::to_string_pretty(&cases)?)?;
    Ok(())
}
