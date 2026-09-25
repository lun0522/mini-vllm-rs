pub(crate) mod child_process;
pub(crate) mod domain_socket;
pub(crate) mod rpc_shutdown;
pub(crate) mod textproto;

pub(crate) fn environment_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be either true or false, got {value}")),
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("failed to read {name}: {error}"),
    }
}

pub(crate) fn environment_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be a non-negative integer, got {value}")),
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("failed to read {name}: {error}"),
    }
}
