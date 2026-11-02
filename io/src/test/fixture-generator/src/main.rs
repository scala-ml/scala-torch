use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::{collections::{BTreeMap, HashMap}, path::Path};

fn main() {
    let output = std::env::args().nth(1).expect("output path");
    let weights: Vec<u8> = [1f32, 2., 3., 4., 5., 6.].iter().flat_map(|x| x.to_le_bytes()).collect();
    let bias: Vec<u8> = [10f32, 20.].iter().flat_map(|x| x.to_le_bytes()).collect();
    let half: Vec<u8> = [0x3c00u16, 0xc000, 0x0001, 0x7bff].iter().flat_map(|x| x.to_le_bytes()).collect();
    let bfloat: Vec<u8> = [0x3f80u16, 0xc000, 0x0001, 0x7fc1].iter().flat_map(|x| x.to_le_bytes()).collect();
    let longs: Vec<u8> = [9007199254740993i64, -7].iter().flat_map(|x| x.to_le_bytes()).collect();
    let doubles: Vec<u8> = [1.25f64, -2.5].iter().flat_map(|x| x.to_le_bytes()).collect();
    let booleans = [0u8, 1u8];
    let empty = [];
    let mut tensors = BTreeMap::new();
    tensors.insert("weight", TensorView::new(Dtype::F32, vec![2, 3], &weights).unwrap());
    tensors.insert("bias", TensorView::new(Dtype::F32, vec![2], &bias).unwrap());
    tensors.insert("half", TensorView::new(Dtype::F16, vec![4], &half).unwrap());
    tensors.insert("bfloat", TensorView::new(Dtype::BF16, vec![4], &bfloat).unwrap());
    tensors.insert("long", TensorView::new(Dtype::I64, vec![2], &longs).unwrap());
    tensors.insert("double", TensorView::new(Dtype::F64, vec![2], &doubles).unwrap());
    tensors.insert("bool", TensorView::new(Dtype::BOOL, vec![2], &booleans).unwrap());
    tensors.insert("empty", TensorView::new(Dtype::F32, vec![0, 3], &empty).unwrap());
    let metadata = Some(HashMap::from([("format".into(), "pt".into())]));
    serialize_to_file(tensors, metadata, Path::new(&output)).unwrap();
}
