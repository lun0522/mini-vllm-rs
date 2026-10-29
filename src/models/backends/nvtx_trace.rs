use crate::utils::environment_bool;
use std::sync::LazyLock;

static ENABLE_NVTX: LazyLock<bool> =
    LazyLock::new(|| environment_bool("MINI_VLLM_ENABLE_NVTX", false));

// RAII guard that pushes an NVTX range on entry and pops it when the guard is dropped.
pub(super) struct NvtxRangeGuard;

// Keep label formatting lazy so disabled NVTX tracing does not allocate per range.
pub(super) fn enter(nvtx_name: impl FnOnce() -> String) -> Option<NvtxRangeGuard> {
    if *ENABLE_NVTX {
        let name = nvtx_name();
        NvtxRangeGuard::enter(&name)
    } else {
        None
    }
}

impl NvtxRangeGuard {
    fn enter(name: &str) -> Option<Self> {
        (cudarc::nvtx::result::range_push(name) >= 0).then_some(Self)
    }
}

impl Drop for NvtxRangeGuard {
    fn drop(&mut self) {
        cudarc::nvtx::result::range_pop();
    }
}
