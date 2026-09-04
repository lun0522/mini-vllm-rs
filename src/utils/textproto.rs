use prost::Message;
use prost_reflect::DescriptorPool;
use prost_reflect::DynamicMessage;

/// Parses textproto into a generated Prost message identified by its full protobuf name.
pub(crate) fn parse_textproto<M>(value: &str, message_name: &str) -> Result<M, String>
where
    M: Message + Default,
{
    let descriptor_pool = descriptor_pool()?;
    let descriptor = descriptor_pool
        .get_message_by_name(message_name)
        .ok_or_else(|| format!("protobuf descriptor '{message_name}' is missing"))?;
    let dynamic_message = DynamicMessage::parse_text_format(descriptor, value)
        .map_err(|error| format!("invalid textproto: {error}"))?;
    M::decode(dynamic_message.encode_to_vec().as_slice())
        .map_err(|error| format!("failed to decode textproto message: {error}"))
}

fn descriptor_pool() -> Result<DescriptorPool, String> {
    DescriptorPool::decode(
        include_bytes!(concat!(env!("OUT_DIR"), "/file_descriptor_set.bin")).as_slice(),
    )
    .map_err(|error| format!("failed to read protobuf descriptors: {error}"))
}
