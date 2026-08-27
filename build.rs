fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/model_runner.proto");
    tonic_build::configure()
        .client_mod_attribute(
            "model_runner",
            "#[allow(clippy::mixed_attributes_style, clippy::result_large_err)]",
        )
        .server_mod_attribute(
            "model_runner",
            "#[allow(clippy::mixed_attributes_style, clippy::result_large_err)]",
        )
        .compile_protos(&["proto/model_runner.proto"], &["proto"])?;
    Ok(())
}
