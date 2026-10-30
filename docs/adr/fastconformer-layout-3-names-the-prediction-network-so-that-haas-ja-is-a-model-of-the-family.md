# FastConformer's layout 3 names the prediction network, so that Haas-ja is a model of the family

## Context

Third Intelligence's Haas-ja-v1.0 (Apache 2.0, 615.8M parameters) is parakeet's architecture with one part replaced: a
FastConformer encoder of 24 layers, a joint and greedy TDT decoding as parakeet-tdt_ctc-0.6b-ja's, whose settings the
file's keys already give, and a prediction network of two pre-norm residual blocks of Kimi Delta Attention (KDA) from
flash-linear-attention 0.5.2 and a final LayerNorm (haas_asr/prednet.py at acc614f) where parakeet has LSTM layers. Its
SentencePiece tokenizer has byte fallback: the pieces `<0x00>` to `<0xFF>` stand for bytes, which a character outside the
vocabulary is written in.

Layout 2 has no key that says which prediction network a file holds, and its converter refused a tokenizer with byte
pieces.

## Decision

- **Haas-ja is a model of the fastconformer family**, read by the same reader, engine and decodings, with the same
  request options.
- **Layout 3** adds `fastconformer.decoder.prediction`, `lstm` or `kda`, which calls for its own keys and tensors: KDA's
  heads, head width, short convolution's width and the eps of its LayerNorms and of its heads' RMSNorm, and each block's
  tensors; and `fastconformer.tokenizer.byte_fallback`, whether the tokens hold byte pieces. Its `speech.requires` is
  0.8.12.
- **The upgrade sets the two keys of a layout 1 or 2 file to `lstm` and false**, which every such file holds, so no user
  converts or downloads a parakeet or ReazonSpeech file again.
- **A step of KDA follows fla's fused_recurrent kernel**, which the official model runs for one label, on
  `ggml_gated_delta_net()` with a decay for each key channel, whose state holds each head's values by key as fla's
  `state_v_first` does: the reader keeps the state as it is, with no transposition.
- **The reference dumps run on CUDA**, in float32 with TF32 off, since fla's kernels run on Triton alone, and the
  joint's output is turned into the log-softmax the joint gives on the CPU, which the checks compare with.

The alternatives were turned down:

- A family of its own for Haas-ja. The frontend, the encoder, the joint, the TDT decoding, the times and the segments
  would be copied, and their checks kept twice, for a part of 3.9M parameters.
- Telling the prediction network apart by the tensors a file holds. The tensors follow from the keys alone, and a file
  missing one tensor would be read as the other network.
- Running KDA as a chunked kernel, as fla trains it. A decoding feeds one label at a time, which fla itself runs with its
  recurrent kernel.

## Consequences

A KDA step computes about three times the nodes of an LSTM step, 105 against 35 for parakeet-tdt_ctc-0.6b-ja, of
vectors of 320 to 640, which sets the threads a decoding on the CPU takes ([the decoding
record](a-fastconformer-model-on-a-gpu-decodes-on-the-cpu-where-its-decoding-weights-are-quantized-and-a-request-can-choose-either-device.md)).
Metal and Vulkan are not checked with Haas-ja.
