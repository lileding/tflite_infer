use std::num::ParseIntError;
use std::path::Path;
use std::fs::File;
use std::io::{self, Read, BufRead};
use std::collections::BTreeMap;

use clap::Parser;

use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, InterpreterBuilder};

use image::imageops::FilterType;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about=None)]
struct Args {
    #[arg(help="TFLite model file")]
    model: String,
    #[arg(short, long, help="Label file")]
    label: Option<String>,
    #[arg(default_value="-", help="Image file")]
    image: String,
}

#[derive(Debug)]
enum Error {
    InvalidModel,
    InvalidLabel(ParseIntError),
    TFLiteError(tflite::Error),
    ImageError(image::error::ImageError),
    IOError(io::Error),
}

impl From<tflite::Error> for Error {
    fn from(e: tflite::Error) -> Self {
        Error::TFLiteError(e)
    }
}

impl From<image::error::ImageError> for Error {
    fn from(e: image::error::ImageError) -> Self {
        Error::ImageError(e)
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::IOError(e)
    }
}

impl From<ParseIntError> for Error {
    fn from(e: ParseIntError) -> Self {
        Error::InvalidLabel(e)
    }
}

type Result<T> = std::result::Result<T, Error>;

fn load_labels<P: AsRef<Path>>(path: P) -> Result<BTreeMap<u32, String>> {
    let mut labels = BTreeMap::new();
    let file = File::open(path)?;
    for line in io::BufReader::new(file).lines() {
        let line = line?;
        if let Some((key, value)) = line.split_once(char::is_whitespace) {
            let key = key.trim().parse::<u32>()?;
            let value = value.trim();
            if !value.is_empty() {
                labels.insert(key, value.to_string());
            }
        }
    }
    Ok(labels)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let labels = args.label.and_then(|path| load_labels(path).ok());

    let model = FlatBufferModel::build_from_file(&args.model)?;
    let resolver = BuiltinOpResolver::default();
    let builder = InterpreterBuilder::new(&model, &resolver)?;
    let mut interpreter = builder.build()?;
    interpreter.allocate_tensors()?;

    let inputs = interpreter.inputs().to_vec();
    let outputs = interpreter.outputs().to_vec();
    let input_tensor = interpreter.tensor_info(inputs[0]).ok_or(Error::InvalidModel)?;
    let input_height = input_tensor.dims[1] as u32;
    let input_width = input_tensor.dims[2] as u32;

    let img = match args.image.as_str() {
        "-" => {
            let mut buf = Vec::new();
            io::stdin().read_to_end(&mut buf)?;
            image::load_from_memory(&buf)
        },
        _ => image::open(&args.image),
    }?.resize_exact(input_width, input_height, FilterType::Lanczos3).to_rgb8();

    interpreter.tensor_data_mut(inputs[0])?.copy_from_slice(&img);
    interpreter.invoke()?;

    //let boxes: &[f32] = interpreter.tensor_data(outputs[0])?;
    let classes: &[f32] = interpreter.tensor_data(outputs[1])?;
    let scores: &[f32] = interpreter.tensor_data(outputs[2])?;
    let count = interpreter.tensor_data::<f32>(outputs[3])?[0] as usize;

    const THRESHOLD: f32 = 0.5;
    for i in 0..count {
        if scores[i] >= THRESHOLD {
            let klass = classes[i] as u32;
            let klass_id = klass.to_string();
            let klass = labels.as_ref().and_then(|labels| labels.get(&klass)).unwrap_or(&klass_id);
            let score = scores[i];
            println!("class: {}, score: {}", klass, score);
        }
    }

    Ok(())
}

