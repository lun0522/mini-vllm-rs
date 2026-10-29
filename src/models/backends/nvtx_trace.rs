use crate::utils::environment_bool;
use cudarc::nvtx::result;
use cudarc::nvtx::sys;
use std::ffi::CString;
use std::sync::LazyLock;

static ENABLE_NVTX: LazyLock<bool> =
    LazyLock::new(|| environment_bool("MINI_VLLM_ENABLE_NVTX", false));

// RAII guard that pushes an NVTX range on entry and pops it when the guard is dropped.
pub(super) struct NvtxRangeGuard;

// Keep label formatting lazy so disabled NVTX tracing does not allocate per range.
pub(super) fn enter(nvtx_name: impl FnOnce() -> String, color: u32) -> Option<NvtxRangeGuard> {
    if *ENABLE_NVTX {
        let name = nvtx_name();
        Some(NvtxRangeGuard::enter(&name, color))
    } else {
        None
    }
}

impl NvtxRangeGuard {
    fn enter(name: &str, color: u32) -> Self {
        let message = CString::new(name).expect("NVTX range name contained a NUL byte");
        let attributes = sys::nvtxEventAttributes_t {
            version: 3,
            size: std::mem::size_of::<sys::nvtxEventAttributes_t>() as u16,
            category: 0,
            colorType: sys::nvtxColorType_t::NVTX_COLOR_ARGB as i32,
            color,
            payloadType: sys::nvtxPayloadType_t::NVTX_PAYLOAD_UNKNOWN as i32,
            reserved0: 0,
            payload: sys::nvtxEventAttributes_v2_payload_t { iValue: 0 },
            messageType: sys::nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII as i32,
            message: sys::nvtxMessageValue_t {
                ascii: message.as_ptr(),
            },
        };
        // NVTX return values vary by implementation. We found that Nsight Systems returned -1
        // while recording the range, similar to this report:
        // https://forums.developer.nvidia.com/t/nvtxrangepusha-and-nvtxrangepop-returns-2/255993
        // So, don't branch on the return value.
        result::range_push_ex(&attributes);
        Self
    }
}

impl Drop for NvtxRangeGuard {
    fn drop(&mut self) {
        result::range_pop();
    }
}
