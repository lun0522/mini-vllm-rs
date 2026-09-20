# Troubleshooting

## A gated tokenizer returns HTTP 401

The Llama example downloads its GGUF weights from the public
`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` repository, but downloads
`tokenizer.json` from the gated `meta-llama/Meta-Llama-3.1-8B-Instruct`
repository. Without approved access and local authentication, startup fails
with an error similar to:

```text
tokenizer model 'meta-llama/Meta-Llama-3.1-8B-Instruct' does not provide
tokenizer.json, or the file could not be downloaded: status code 401
```

To fix it:

1. Sign in to Hugging Face and open
   [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).
2. Accept the model terms and wait for access approval if it is not immediate.
3. Create a Hugging Face user token with read access.
4. Save the token where this project's `hf-hub` client can read it:

   ```shell
   hf auth login
   ```

5. Confirm that the CLI uses the approved account, then rerun the original
   command:

   ```shell
   hf auth whoami
   ```

If the `hf` command is unavailable, install the
[official Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

## Metal produces repetitive or nonsensical output after loading a draft model

Loading a draft model increases unified-memory usage even before speculative
decoding uses it: both models' weights and their preallocated KV caches remain
resident, alongside activations and Metal working buffers. Under severe memory
pressure, Metal may report an allocation or command-buffer error, but an error
is not guaranteed. Buffer allocation can succeed before the working set is
fully exercised, and inference may instead produce numerically corrupted logits
that appear as repetitive punctuation, a repeated token, or otherwise
nonsensical text.

If target-model output is correct without `--draft-model` but becomes corrupted
when the draft model is loaded, treat memory pressure as the first suspect. Try
omitting the draft model or using smaller target and draft models. You can also
open macOS Activity Monitor, select the Memory tab, and watch the Memory
Pressure graph and Swap Used while loading the models and generating text. A
yellow or red graph, or rapidly increasing swap usage, supports the
memory-pressure diagnosis. Lowering `--target-kv-cache-size-bytes` can confirm
the diagnosis. Changing the number of tokens per page does not materially
reduce the requested KV-cache budget.
