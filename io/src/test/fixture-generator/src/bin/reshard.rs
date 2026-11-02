use safetensors::{SafeTensors, tensor::serialize_to_file};
use std::{collections::BTreeMap, path::Path};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    std::fs::create_dir_all(output).unwrap();
    let data = std::fs::read(input.join("model.safetensors")).unwrap();
    let archive = SafeTensors::deserialize(&data).unwrap();
    let mut tensors = archive.tensors();
    tensors.sort_by(|a,b| a.0.cmp(&b.0));
    let mut groups = [BTreeMap::new(),BTreeMap::new()];
    let mut weights = BTreeMap::new();
    let mut total = 0usize;
    let filenames = ["model-00001-of-00002.safetensors","model-00002-of-00002.safetensors"];
    for (i,(name,view)) in tensors.into_iter().enumerate() {
        total += view.data().len();
        weights.insert(name.clone(),filenames[i%2]);
        groups[i%2].insert(name,view);
    }
    for (i,group) in groups.into_iter().enumerate() {
        serialize_to_file(group,None,&output.join(filenames[i])).unwrap();
    }
    let index = serde_json::json!({"metadata":{"total_size":total},"weight_map":weights});
    std::fs::write(output.join("model.safetensors.index.json"),serde_json::to_string_pretty(&index).unwrap()).unwrap();
    std::fs::copy(input.join("config.json"),output.join("config.json")).unwrap();
    println!("resharded {total} payload bytes into two files");
}
