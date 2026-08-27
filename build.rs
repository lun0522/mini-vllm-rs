fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/model_runner.proto");
    prost_build::compile_protos(&["proto/model_runner.proto"], &["proto"])?;
    Ok(())
}
