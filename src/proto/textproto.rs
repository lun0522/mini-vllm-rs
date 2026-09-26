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

/// Formats a generated Prost message using its protobuf text representation.
pub(crate) fn format_textproto<M>(value: &M, message_name: &str) -> Result<String, String>
where
    M: Message,
{
    let descriptor_pool = descriptor_pool()?;
    let descriptor = descriptor_pool
        .get_message_by_name(message_name)
        .ok_or_else(|| format!("protobuf descriptor '{message_name}' is missing"))?;
    let encoded = value.encode_to_vec();
    let dynamic_message = DynamicMessage::decode(descriptor, encoded.as_slice())
        .map_err(|error| format!("failed to convert protobuf message to textproto: {error}"))?;
    Ok(dynamic_message.to_text_format())
}

fn descriptor_pool() -> Result<DescriptorPool, String> {
    DescriptorPool::decode(
        include_bytes!(concat!(env!("OUT_DIR"), "/file_descriptor_set.bin")).as_slice(),
    )
    .map_err(|error| format!("failed to read protobuf descriptors: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::inference_config::draft_token_count_policy::Policy;
    use crate::proto::inference_config::DraftTokenCountPolicy;
    use crate::proto::inference_config::FixedDraftTokenCountPolicy;

    #[test]
    fn round_trips_generated_messages() {
        let policy = DraftTokenCountPolicy {
            policy: Some(Policy::Fixed(FixedDraftTokenCountPolicy {
                draft_token_count: 4,
            })),
        };

        let formatted =
            format_textproto(&policy, "inference_config.DraftTokenCountPolicy").unwrap();
        let parsed = parse_textproto::<DraftTokenCountPolicy>(
            &formatted,
            "inference_config.DraftTokenCountPolicy",
        )
        .unwrap();

        assert_eq!(parsed, policy);
    }
}
