fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/model_runner.proto");
    println!("cargo:rerun-if-changed=proto/request_handler.proto");
    println!("cargo:rerun-if-changed=proto/main_process.proto");
    println!("cargo:rerun-if-changed=proto/model_config.proto");
    let descriptor_path =
        std::path::PathBuf::from(std::env::var("OUT_DIR")?).join("file_descriptor_set.bin");
    tonic_build::configure()
        .file_descriptor_set_path(descriptor_path)
        .client_mod_attribute(
            "model_runner",
            "#[allow(clippy::mixed_attributes_style, clippy::result_large_err)]",
        )
        .server_mod_attribute(
            "model_runner",
            "#[allow(clippy::mixed_attributes_style, clippy::result_large_err)]",
        )
        .client_mod_attribute(
            "request_handler",
            "#[allow(clippy::mixed_attributes_style, clippy::result_large_err)]",
        )
        .server_mod_attribute(
            "request_handler",
            "#[allow(clippy::mixed_attributes_style, clippy::result_large_err)]",
        )
        .client_mod_attribute(
            "main_process",
            "#[allow(clippy::mixed_attributes_style, clippy::result_large_err)]",
        )
        .server_mod_attribute(
            "main_process",
            "#[allow(clippy::mixed_attributes_style, clippy::result_large_err)]",
        )
        .compile_protos(
            &[
                "proto/model_config.proto",
                "proto/model_runner.proto",
                "proto/request_handler.proto",
                "proto/main_process.proto",
            ],
            &["proto"],
        )?;
    Ok(())
}
