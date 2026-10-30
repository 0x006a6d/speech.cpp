"""Converts a pinned NeMo FastConformer checkpoint with an RNN-T or TDT decoder to the one GGUF file the C++ port reads.

usage: uv run python convert.py <model> <out dir>

Writes parakeet-tdt_ctc-0.6B-ja-F32.gguf, parakeet-tdt-0.6B-v3-F32.gguf, reazonspeech-nemo-619M-v2-F32.gguf or, run from
reference/haas-ja/convert.py, Haas-616M-ja-v1.0-F32.gguf, named under GGUF's naming convention, in layout 3: the
frontend's window and mel filterbank, the subsampling, the conformer layers, the prediction network, LSTM layers or
Haas-ja's Kimi Delta Attention layers, and the joint, with the SentencePiece pieces the token ids name, the settings of the
decoding transcribe() runs (greedy TDT's durations and limit, or the beam and length of RNN-T's alignment-length
synchronous beam search) and, for RNN-T, the limit of the greedy decoding NeMo also runs on it, every other constant the
C++ reads and the model's identity in the GGUF specification's general keys. A hybrid checkpoint's CTC head is left out,
since NeMo decodes with its transducer.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis: a Linear weight [out, in] is stored as is
(ne = [in, out]). Every tensor is written in float32; `speech quantize` makes the other weight types of the file, each
tensor in the type src/families/fastconformer/layout.cpp gives it. The batch norm of each convolution
module is folded into its depthwise convolution, which it follows in evaluation, and each LSTM layer's two biases are
summed. A tokenizer with byte fallback keeps its byte pieces, <0x00> to <0xFF>, and fastconformer.tokenizer.byte_fallback
says so. A checkpoint whose conformer layers have no biases (ConformerEncoder's use_bias false) is written without them,
and fastconformer.encoder.use_bias says so.
"""

import argparse
import os

import numpy as np
from gguf import GGUFValueType, GGUFWriter, LlamaFileType, naming_convention, size_label
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel, EncDecRNNTBPEModel
from nemo.collections.asr.parts.preprocessing import features
from nemo.collections.asr.parts.submodules import multi_head_attention
from sentencepiece import sentencepiece_model_pb2

from pins import MODELS, restore

ARCH = "fastconformer"
# Each layout this converter has written, with the first release of speech.cpp whose reader takes it; it writes the
# last.
RELEASES = {1: "0.7.0", 2: "0.8.0", 3: "0.8.12"}
LAYOUT = max(RELEASES)
# The ISO 639 two-letter codes of the languages each checkpoint transcribes, from its model card, which requests give as
# BCP 47 tags. None takes a language: parakeet-tdt-0.6b-v3 finds the language of the audio itself.
LANGUAGES = {
    "parakeet-tdt_ctc-0.6b-ja": ["ja"],
    "reazonspeech-nemo-v2": ["ja"],
    "parakeet-tdt-0.6b-v3": ["en", "es", "fr", "de", "bg", "hr", "cs", "da", "nl", "et", "fi", "el", "hu", "it", "lv",
                             "lt", "mt", "pl", "pt", "ro", "sk", "sl", "sv", "ru", "uk"],
    "Haas-ja-v1.0": ["ja"],
}
# The SPDX identifier of each checkpoint's license, from its model card.
LICENSES = {"parakeet-tdt_ctc-0.6b-ja": "CC-BY-4.0", "parakeet-tdt-0.6b-v3": "CC-BY-4.0", "reazonspeech-nemo-v2": "Apache-2.0",
            "Haas-ja-v1.0": "Apache-2.0"}
# The parts of each model's name under GGUF's naming convention (ggml's docs/gguf.md): its line, its size where the name
# gives one, what it was trained toward (ja, the language of parakeet-tdt_ctc-0.6b-ja) and its version. Where the name
# gives no size, the size label is counted from the parameters of the file's tensors.
NAMES = {
    "parakeet-tdt_ctc-0.6b-ja": {"basename": "parakeet-tdt_ctc", "size_label": "0.6B", "finetune": "ja", "version": None},
    "parakeet-tdt-0.6b-v3": {"basename": "parakeet-tdt", "size_label": "0.6B", "finetune": None, "version": "v3"},
    "reazonspeech-nemo-v2": {"basename": "reazonspeech-nemo", "size_label": None, "finetune": None, "version": "v2"},
    "Haas-ja-v1.0": {"basename": "Haas", "size_label": None, "finetune": "ja", "version": "v1.0"},
}
# speech.cpp's addition to NeMo's segments: the marks that end a segment wherever they stand, for a model whose
# languages are written without spaces. NeMo ends a segment at one of its separators only where a word ends, which
# text without spaces between its words never reaches.
UNSPACED_LANGUAGES = {"ja", "zh"}
UNSPACED_BREAKS = ["。", "？", "！", "?", "!"]


class Writer(GGUFWriter):
    """A GGUFWriter that writes an empty array of a given element type, which GGUFWriter refuses."""

    def _pack_val(self, val, vtype, add_vtype, sub_type=None):
        if vtype == GGUFValueType.ARRAY and sub_type is not None and len(val) == 0:
            return (self._pack("I", vtype) if add_vtype else b"") + self._pack("I", sub_type) + self._pack("Q", 0)
        return super()._pack_val(val, vtype, add_vtype, sub_type)


parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

pin = MODELS[args.model]
model = restore(pin)
cfg = model.cfg
enc = cfg.encoder
featurizer = model.preprocessor.featurizer
encoder = model.encoder
assert enc.subsampling == "dw_striding" and not enc.get("causal_downsampling", False)
assert enc.conv_norm_type == "batch_norm" and encoder.conv_context_size == [(enc.conv_kernel_size - 1) // 2] * 2
# Attention over the whole utterance, or Longformer's local attention with as many frames on either side and one or
# more global tokens from the first frame on, whose queries, keys and values are the layer's own.
attention = encoder.self_attention_model
assert encoder.att_context_style == "regular" and len(encoder.att_context_size_all) == 1
context = list(encoder.att_context_size)
if attention == "rel_pos":
    assert context == [-1, -1]
else:
    assert attention == "rel_pos_local_attn" and context[0] == context[1] > 0, (attention, context)
    assert encoder.global_tokens == 1 or (encoder.global_tokens > 1 and encoder.global_tokens_spacing == 1)
    assert not encoder.global_attn_separate
assert featurizer.exact_pad is False and featurizer.frame_splicing == 1 and featurizer.mag_power == 2.0
assert featurizer.normalize == "per_feature" and featurizer.log and featurizer.log_zero_guard_type == "add"
assert featurizer.pad_value == 0
# The decoding transcribe() runs by default, which the C++ ports for each kind of transducer: TDT's greedy label
# looping (greedy_batch, GreedyBatchedTDTInfer), or RNN-T's alignment-length synchronous beam search (alsd,
# BeamRNNTInfer). Both feed the blank first as the prediction network's padding (its embedding zero, so the blank is
# the start of the sequence) and use a ReLU joint. The joint's log-softmax, which NeMo applies on the CPU alone, is
# left out; the beam search takes its own log-softmax over the tokens and the blank, which the C++ computes.
# A model with a prompt (EncDecRNNTBPEModelWithPrompt and its hybrid) is told its language through an input the C++
# does not have; a hybrid's CTC head is left aside while cur_decoder is "rnnt".
dec, joint, decoding = model.decoder, model.joint, model.decoding
assert type(model) in (EncDecRNNTBPEModel, EncDecHybridRNNTCTCBPEModel)
assert type(model) is EncDecRNNTBPEModel or model.cur_decoder == "rnnt"
decoder = {"GreedyBatchedTDTInfer": "tdt", "BeamRNNTInfer": "rnnt"}[type(decoding.decoding).__name__]
if decoder == "tdt":
    assert decoding.cfg.strategy == "greedy_batch" and not decoding.cfg.get("big_blank_durations")
    assert type(decoding.decoding.decoding_computer).__name__ == "GreedyBatchedTDTLabelLoopingComputer"
    assert not decoding.decoding.decoding_computer.has_fusion_models()
    durations = [int(d) for d in decoding.cfg.durations]
else:
    beam = decoding.decoding
    # A beam of 1 runs greedy_search() instead. The blank is the last class (index_incr 0), and a float target
    # length is a multiple of the frames.
    assert decoding.cfg.strategy == "alsd" and beam.search_algorithm == beam.align_length_sync_decoding
    assert beam.beam_size > 1 and beam.return_best_hypothesis and beam.softmax_temperature == 1.0
    assert beam.language_model is None and beam.ngram_lm is None and not beam.hat_subtract_ilm
    assert beam.blank == beam.vocab_size and isinstance(beam.alsd_max_target_length, float)
    # The greedy decoding a caller may choose instead, GreedyBatchedRNNTInfer or GreedyRNNTInfer, takes the most tokens
    # on one frame as RNNTDecoding reads it from the configuration's greedy settings; without a limit it would emit
    # tokens on one frame for as long as the joint gives them.
    greedy_max_symbols = decoding.cfg.greedy.get("max_symbols", None) or decoding.cfg.greedy.get("max_symbols_per_step", None)
    assert isinstance(greedy_max_symbols, int) and greedy_max_symbols > 0, greedy_max_symbols
    durations = []
assert dec.blank_as_pad and dec.blank_idx == decoding.blank_id and not dec.random_state_sampling
# The prediction network: LSTMDropout, the plain LSTM rnn() makes without a normalization, whose dropout is off in
# evaluation, or Haas-ja's KDAPredictionNetwork, pre-norm residual blocks of fla's KimiDeltaAttention and a final
# LayerNorm (haas_asr/prednet.py). A step of one label runs KDA's fused_recurrent kernel, whose rule the C++ follows:
# q and k divided by sqrt(sum of squares + 1e-6) in each head, q scaled by 1/sqrt(head_dim), the decay
# -exp(A_log) softplus(f + dt_bias) of each key channel and beta = sigmoid(b), on states with the value first.
dec_rnn = dec.prediction.dec_rnn
prediction = {"LSTMDropout": "lstm", "KDAPredictionNetwork": "kda"}[type(dec_rnn).__name__]
if prediction == "lstm":
    assert dec_rnn.lstm.proj_size == 0 and dec_rnn.lstm.bias and not dec_rnn.lstm.bidirectional
else:
    for kda in dec_rnn.layers:
        assert kda.use_short_conv and not kda.conv_bias and kda.num_v_heads == kda.num_heads and kda.head_v_dim == kda.head_k_dim
        assert not kda.allow_neg_eigval and not kda.safe_gate and kda.lower_bound is None
        assert kda.q_conv1d.activation == kda.k_conv1d.activation == kda.v_conv1d.activation == "silu"
        assert kda.o_norm.activation == "sigmoid" and kda.o_norm.bias is None
        assert kda.g_proj[1].bias is not None and kda.f_proj[0].bias is None and kda.f_proj[1].bias is None
    assert all(type(n).__name__ == "LayerNorm" and n.eps == dec_rnn.norms[0].eps for n in dec_rnn.norms)
    assert dec_rnn.final_norm is not None and dec_rnn.final_norm.eps == dec_rnn.norms[0].eps
assert not dec.is_adapter_available() and not joint.is_adapter_available()
assert float(dec.prediction.embed.weight[dec.blank_idx].abs().max()) == 0
assert joint.activation == "relu" and joint.temperature == 1.0
assert joint.num_extra_outputs == len(durations) == len(decoding.durations or [])
assert joint.joint_net[-1].out_features == decoding.blank_id + 1 + len(durations)

w = Writer(None, ARCH)


def add_array(key, values, element):
    """An array with its element type given, not inferred from its first value."""
    w.add_key_value(key, list(values), GGUFValueType.ARRAY, sub_type=element)


w.add_uint32("speech.layout", LAYOUT)
w.add_string("speech.requires", RELEASES[LAYOUT])
w.add_string("speech.task", "recognition")
w.add_uint32("speech.sample_rate", int(featurizer.sample_rate))
w.add_string("speech.language_use", "checked")

# The frontend: FilterbankFeatures' parameters, and the normalization guard, CONSTANT in features.py.
w.add_uint32("fastconformer.frontend.n_fft", int(featurizer.n_fft))
w.add_uint32("fastconformer.frontend.hop_length", int(featurizer.hop_length))
w.add_uint32("fastconformer.frontend.n_mels", int(featurizer.nfilt))
w.add_float32("fastconformer.frontend.preemphasis", float(featurizer.preemph))
w.add_float32("fastconformer.frontend.log_guard", float(featurizer.log_zero_guard_value))
w.add_float32("fastconformer.frontend.std_guard", float(features.CONSTANT))

# The encoder. pos_base is INF_VAL of multi_head_attention.py, the base of RelPositionalEncoding's
# wavelengths; ff_factor is ConformerLayer's fc_factor, the weight of each half-step feed-forward.
w.add_uint32("fastconformer.encoder.d_model", int(enc.d_model))
w.add_uint32("fastconformer.encoder.num_layers", int(enc.n_layers))
w.add_uint32("fastconformer.encoder.num_heads", int(enc.n_heads))
w.add_uint32("fastconformer.encoder.conv_kernel", int(enc.conv_kernel_size))
w.add_uint32("fastconformer.encoder.subsampling_factor", int(enc.subsampling_factor))
w.add_float32("fastconformer.encoder.norm_eps", float(model.encoder.layers[0].norm_out.eps))
w.add_float32("fastconformer.encoder.pos_base", float(multi_head_attention.INF_VAL))
w.add_float32("fastconformer.encoder.xscale", float(model.encoder.xscale or 1.0))
w.add_float32("fastconformer.encoder.ff_factor", float(model.encoder.layers[0].fc_factor))
# Whether the linear layers of the feed-forward modules and the attention and the pointwise convolutions have biases;
# the depthwise convolution has one either way once the batch norm is folded into it.
use_bias = bool(model.encoder.layers[0].feed_forward1.use_bias)
assert all(bool(m.use_bias) == use_bias for layer in model.encoder.layers
           for m in (layer.feed_forward1, layer.feed_forward2, layer.self_attn, layer.conv))
w.add_bool("fastconformer.encoder.use_bias", use_bias)
w.add_string("fastconformer.encoder.attention", attention)
if attention == "rel_pos_local_attn":
    w.add_uint32("fastconformer.encoder.attention_context", int(context[0]))
    w.add_uint32("fastconformer.encoder.global_tokens", int(encoder.global_tokens))

# The joint's classes are the tokenizer's ids, a blank after them, then one per duration.
tokenizer = model.tokenizer
assert not tokenizer.legacy
proto = sentencepiece_model_pb2.ModelProto()
proto.ParseFromString(tokenizer.tokenizer.serialized_model_proto())
# Normal, user-defined, unknown and byte pieces only: SentencePiece writes a user-defined piece (parakeet-tdt-0.6b-v3's
# <|en|> and the other tags) as it writes a normal one, and the C++ detokenizer handles no control pieces. A byte piece
# <0xXX>, which byte fallback adds, stands for its byte, and the bytes of consecutive byte pieces are read as UTF-8.
byte_fallback = bool(proto.trainer_spec.byte_fallback)
allowed = (proto.SentencePiece.NORMAL, proto.SentencePiece.USER_DEFINED, proto.SentencePiece.UNKNOWN)
allowed += (proto.SentencePiece.BYTE,) if byte_fallback else ()
assert all(p.type in allowed for p in proto.pieces)
assert all(p.piece == f"<0x{i:02X}>" for i, p in enumerate(p for p in proto.pieces if p.type == proto.SentencePiece.BYTE))
assert not proto.trainer_spec.treat_whitespace_as_suffix
blank = int(decoding.blank_id)
assert blank == len(proto.pieces) == tokenizer.vocab_size
w.add_string("fastconformer.decoder.kind", decoder)
w.add_uint32("fastconformer.decoder.blank_id", blank)
w.add_uint32("fastconformer.decoder.prediction_layers", int(dec.pred_rnn_layers))
w.add_string("fastconformer.decoder.prediction", prediction)
if prediction == "kda":
    kda = dec_rnn.layers[0]
    w.add_uint32("fastconformer.decoder.kda.num_heads", int(kda.num_heads))
    w.add_uint32("fastconformer.decoder.kda.head_dim", int(kda.head_k_dim))
    w.add_uint32("fastconformer.decoder.kda.conv_size", int(kda.conv_size))
    # The eps of the blocks' and the final LayerNorm, and of the RMSNorm of each head's output.
    w.add_float32("fastconformer.decoder.kda.norm_eps", float(dec_rnn.norms[0].eps))
    w.add_float32("fastconformer.decoder.kda.output_norm_eps", float(kda.o_norm.eps))
if decoder == "tdt":
    add_array("fastconformer.decoder.tdt.durations", durations, GGUFValueType.INT32)
    # The most tokens emitted on one frame before the decoding moves to the next.
    w.add_uint32("fastconformer.decoder.tdt.max_symbols", int(decoding.decoding.max_symbols))
else:
    w.add_uint32("fastconformer.decoder.rnnt.beam_size", int(beam.beam_size))
    # Whether the best of the finished hypotheses is the one with the highest score per label, the blank it starts
    # with counted.
    w.add_bool("fastconformer.decoder.rnnt.score_norm", bool(beam.score_norm))
    # The most labels a hypothesis takes, as a multiple of the encoder's frames.
    w.add_float32("fastconformer.decoder.rnnt.max_target_ratio", float(beam.alsd_max_target_length))
    # The most tokens greedy decoding emits on one frame before it moves to the next.
    w.add_uint32("fastconformer.decoder.rnnt.max_symbols", greedy_max_symbols)
# The marks that end a segment where a word ends: the checkpoint's decoding segment_seperators, or NeMo's default
# (".", "?", "!") when it sets none, as the decoding object takes them.
add_array("fastconformer.segment.separators", list(decoding.segment_seperators), GGUFValueType.STRING)
unspaced = all(language in UNSPACED_LANGUAGES for language in LANGUAGES[args.model])
add_array("fastconformer.segment.breaks", UNSPACED_BREAKS if unspaced else [], GGUFValueType.STRING)
add_array("fastconformer.tokenizer.tokens", [p.piece for p in proto.pieces], GGUFValueType.STRING)
w.add_uint32("fastconformer.tokenizer.unknown_id", int(tokenizer.tokenizer.unk_id()))
w.add_string("fastconformer.tokenizer.unknown_surface", proto.trainer_spec.unk_surface)
# Whether SentencePiece's decoder drops the leading "▁" of each piece until the text is no longer empty, which it
# does when either normalizer option is set.
w.add_bool("fastconformer.tokenizer.byte_fallback", byte_fallback)
w.add_bool("fastconformer.tokenizer.strip_leading_space",
           bool(proto.normalizer_spec.add_dummy_prefix or proto.normalizer_spec.remove_extra_whitespaces))
# The marks before which the decoding removes one whitespace character. The C++ looks for a space only,
# the one whitespace character a decoded text can hold when no piece holds another and the unknown surface
# holds only spaces.
assert not any(c.isspace() for p in proto.pieces for c in p.piece)
assert all(c == " " or not c.isspace() for c in proto.trainer_spec.unk_surface)
add_array("fastconformer.tokenizer.punctuation", sorted(decoding.supported_punctuation or []), GGUFValueType.STRING)
# The decoding strips no language tags from the text (strip_lang_tags, for models that write them), which the C++
# does not do; NeMo before 3.0, which Haas-ja runs on, has no such option.
assert not getattr(decoding, "strip_lang_tags", False)

sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}


def add(name, data):
    w.add_tensor(name, np.ascontiguousarray(data, dtype=np.float32))


add("frontend.window", sd["preprocessor.featurizer.window"])
add("frontend.filterbank", sd["preprocessor.featurizer.fb"][0])

# The subsampling's nn.Sequential: Conv2d, ReLU, then (depthwise Conv2d, pointwise Conv2d, ReLU) per halving.
s = "encoder.pre_encode."
add("sub.conv.0.weight", sd[s + "conv.0.weight"])
add("sub.conv.0.bias", sd[s + "conv.0.bias"])
for i in range(1, int(np.log2(enc.subsampling_factor))):
    dw, pw = 3 * i - 1, 3 * i
    add(f"sub.conv.{i}.dw.weight", sd[s + f"conv.{dw}.weight"])
    add(f"sub.conv.{i}.dw.bias", sd[s + f"conv.{dw}.bias"])
    add(f"sub.conv.{i}.pw.weight", sd[s + f"conv.{pw}.weight"][:, :, 0, 0])
    add(f"sub.conv.{i}.pw.bias", sd[s + f"conv.{pw}.bias"])
add("sub.out.weight", sd[s + "out.weight"])
add("sub.out.bias", sd[s + "out.bias"])

d = int(enc.d_model)
for l in range(int(enc.n_layers)):
    a, o = f"encoder.layers.{l}.", f"blk.{l}."

    def linear(src, dst, bias=use_bias):
        add(dst + ".weight", sd[a + src + ".weight"])
        if bias:
            add(dst + ".bias", sd[a + src + ".bias"])

    def norm(src, dst):
        add(dst + ".weight", sd[a + src + ".weight"])
        add(dst + ".bias", sd[a + src + ".bias"])

    for ff in ("1", "2"):
        norm(f"norm_feed_forward{ff}", o + f"ff{ff}_norm")
        linear(f"feed_forward{ff}.linear1", o + f"ff{ff}_up")
        linear(f"feed_forward{ff}.linear2", o + f"ff{ff}_down")
    norm("norm_self_att", o + "attn_norm")
    for x in ("q", "k", "v", "out"):
        linear(f"self_attn.linear_{x}", o + f"attn_{x}")
    linear("self_attn.linear_pos", o + "attn_pos", bias=False)
    add(o + "attn_pos_bias_u", sd[a + "self_attn.pos_bias_u"])
    add(o + "attn_pos_bias_v", sd[a + "self_attn.pos_bias_v"])

    norm("norm_conv", o + "conv_norm")
    # GLU keeps the first half of the pointwise convolution's channels and gates them with the second.
    pw1 = sd[a + "conv.pointwise_conv1.weight"][:, :, 0]
    add(o + "conv_pw1_a.weight", pw1[:d])
    add(o + "conv_pw1_gate.weight", pw1[d:])
    if use_bias:
        add(o + "conv_pw1_a.bias", sd[a + "conv.pointwise_conv1.bias"][:d])
        add(o + "conv_pw1_gate.bias", sd[a + "conv.pointwise_conv1.bias"][d:])
    bn = a + "conv.batch_norm."
    scale = sd[bn + "weight"] / np.sqrt(sd[bn + "running_var"] + model.encoder.layers[l].conv.batch_norm.eps)
    dw_bias = sd[a + "conv.depthwise_conv.bias"] if use_bias else 0
    add(o + "conv_dw.weight", sd[a + "conv.depthwise_conv.weight"][:, 0, :] * scale[:, None])
    add(o + "conv_dw.bias", (dw_bias - sd[bn + "running_mean"]) * scale + sd[bn + "bias"])
    add(o + "conv_pw2.weight", sd[a + "conv.pointwise_conv2.weight"][:, :, 0])
    if use_bias:
        add(o + "conv_pw2.bias", sd[a + "conv.pointwise_conv2.bias"])
    norm("norm_out", o + "out_norm")

# The prediction network: an embedding of the tokens and the blank, then LSTM layers whose gates are stacked as
# PyTorch stacks them, input, forget, cell and output, or KDA blocks: each block's LayerNorm, its projections, its
# short convolutions' kernels [channels, width], the low-rank projections of the decay and of the output gate, A_log,
# dt_bias, the RMSNorm of the heads' output and the output projection, then the final LayerNorm.
add("pred.embed.weight", sd["decoder.prediction.embed.weight"])
for l in range(int(dec.pred_rnn_layers)):
    if prediction == "lstm":
        r = "decoder.prediction.dec_rnn.lstm."
        add(f"pred.lstm.{l}.ih.weight", sd[r + f"weight_ih_l{l}"])
        add(f"pred.lstm.{l}.hh.weight", sd[r + f"weight_hh_l{l}"])
        add(f"pred.lstm.{l}.bias", sd[r + f"bias_ih_l{l}"] + sd[r + f"bias_hh_l{l}"])
        continue
    r, o = f"decoder.prediction.dec_rnn.layers.{l}.", f"pred.kda.{l}."
    add(o + "norm.weight", sd[f"decoder.prediction.dec_rnn.norms.{l}.weight"])
    add(o + "norm.bias", sd[f"decoder.prediction.dec_rnn.norms.{l}.bias"])
    for x in ("q", "k", "v"):
        add(o + f"{x}.weight", sd[r + f"{x}_proj.weight"])
        add(o + f"{x}_conv.weight", sd[r + f"{x}_conv1d.weight"][:, 0, :])
    add(o + "f_down.weight", sd[r + "f_proj.0.weight"])
    add(o + "f_up.weight", sd[r + "f_proj.1.weight"])
    add(o + "b.weight", sd[r + "b_proj.weight"])
    add(o + "g_down.weight", sd[r + "g_proj.0.weight"])
    add(o + "g_up.weight", sd[r + "g_proj.1.weight"])
    add(o + "g_up.bias", sd[r + "g_proj.1.bias"])
    add(o + "a_log", sd[r + "A_log"])
    add(o + "dt_bias", sd[r + "dt_bias"])
    add(o + "out_norm.weight", sd[r + "o_norm.weight"])
    add(o + "out.weight", sd[r + "o_proj.weight"])
if prediction == "kda":
    add("pred.final_norm.weight", sd["decoder.prediction.dec_rnn.final_norm.weight"])
    add("pred.final_norm.bias", sd["decoder.prediction.dec_rnn.final_norm.bias"])
for x in ("enc", "pred"):
    add(f"joint.{x}.weight", sd[f"joint.{x}.weight"])
    add(f"joint.{x}.bias", sd[f"joint.{x}.bias"])
add("joint.out.weight", sd["joint.joint_net.2.weight"])
add("joint.out.bias", sd["joint.joint_net.2.bias"])

# The model's identity and languages in the GGUF specification's general keys, which name the file. They follow the
# tensors, whose parameters give a size label the name does not, and then go first after general.architecture, since
# GGUFWriter writes the keys in the order they were added.
parts = NAMES[args.model]
label = parts["size_label"] or size_label(*w.get_total_parameter_count())
repository = f"https://huggingface.co/{pin['repository']}"
w.add_name(pin["repository"].split("/")[1])
w.add_organization(pin["repository"].split("/")[0])
w.add_basename(parts["basename"])
w.add_size_label(label)
if parts["finetune"]:
    w.add_finetune(parts["finetune"])
if parts["version"]:
    w.add_version(parts["version"])
w.add_license(LICENSES[args.model])
w.add_source_url(f"{repository}/tree/{pin['revision']}")
w.add_source_repo_url(repository)
w.add_file_type(LlamaFileType.ALL_F32)
w.add_languages(sorted(LANGUAGES[args.model]))
w.kv_data[0] = dict(sorted(w.kv_data[0].items(), key=lambda item: not item[0].startswith("general.")))

path = os.path.join(args.out_dir, naming_convention(None, parts["basename"], parts["finetune"], parts["version"], label, "f32") + ".gguf")
w.write_header_to_file(path)
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", path)
