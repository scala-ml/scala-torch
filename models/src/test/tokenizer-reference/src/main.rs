use tokenizers::Tokenizer;
fn hex(s: &str) -> String { s.as_bytes().iter().map(|b| format!("{b:02x}")).collect() }
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let t = Tokenizer::from_file(path).unwrap();
    let cases = ["", "Hello, world!", " leading  spaces   ", "\t\n\r\nHello\u{a0}world\u{2003}!", "I'm we're can't I'M WE'RE", "12345 １２３ ١٢٣ Ⅻ²½", "日本語とScala 🦀👩‍💻 café e\u{301}", "a\0b\u{1f}c", "a\u{4}b\u{6}c\u{40000}", "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n", "x<|im_end|><|im_end|>y<|unknown|>", "aaaaaaa abcabcabc ......\n\n", "\u{85}\u{b}\u{c}\u{2028}\u{2029}"];
    for text in cases {
        let ids = t.encode(text, false).unwrap();
        let csv = ids.get_ids().iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",");
        println!("{}\t{}\t{}\t{}", hex(text), csv, hex(&t.decode(ids.get_ids(), false).unwrap()), hex(&t.decode(ids.get_ids(), true).unwrap()));
    }
    let cases: &[&[u8]] = &[&[0xed,0xa0,0x80], &[0xf0,0x90,0x80], &[0xe2,0x82,0x41], &[0xe0,0x80,0x80], &[0xf4,0x90,0x80,0x80], &[0xc3,0xa9], &[0x80,0x81]];
    let mut output = String::new();
    for bytes in cases {
        let ids: Vec<u32> = bytes.iter().map(|b| {
            let direct = |n: u8| (33..=126).contains(&n) || (161..=172).contains(&n) || n >= 174;
            let cp = if direct(*b) { *b as u32 } else { 256 + (0..*b).filter(|n| !direct(*n)).count() as u32 };
            t.token_to_id(&char::from_u32(cp).unwrap().to_string()).unwrap()
        }).collect();
        output += &format!("{}\t{}\n", ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(","), hex(&t.decode(&ids, false).unwrap()));
    }
    std::fs::write(std::env::args().nth(2).unwrap(), output).unwrap();
}
