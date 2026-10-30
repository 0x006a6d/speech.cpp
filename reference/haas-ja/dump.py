"""Runs the pinned Haas-ja on CUDA in float32 and saves the tensors the C++ port is checked against.

usage: uv run python dump.py <model> <out dir> <audio.wav>...

Each WAVE file, mono at 16 kHz, goes to <out dir>/<model>/<file name without .wav>/, with the stages that
reference/fastconformer/dump.py saves for a model it runs, in the same files and shapes, so that the FastConformer
checks read them alike: audio, features, subsampled, layers, encoded, pred_labels, pred_parents, pred_output,
joint_frames, joint_steps, joint_output, the log-softmax the joint returns on the CPU, and ids, and text.txt and
meta.json. The model runs on CUDA rather than the CPU,
since the prediction network's Kimi Delta Attention runs on fla's Triton kernels alone, with TF32 off (pins.restore()).
The text of the stages is asserted to equal what the official transcribe() returns with the model's default decoding,
greedy TDT, here without CUDA graphs.

Then each file goes through transcribe(timestamps=True), whose ids and text are asserted to be the stages', and the
times it gives are saved beside them as reference/fastconformer/dump.py saves them: token_timestep, token_duration,
token_offsets, token_seconds, segment_offsets, segment_seconds, segments.txt and times.json.
"""

import argparse
import copy
import json
import os

import numpy as np
import soundfile
import torch
from omegaconf import open_dict

from pins import FLA, MODELS, NEMO, restore

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
parser.add_argument("audio", nargs="+")
args = parser.parse_args()

pin = MODELS[args.model]
device = "cuda"
model = restore(pin, device)
sample_rate = int(model.cfg.preprocessor.sample_rate)
# What the checkpoint itself sets of the decoding's segments, read before transcribe(timestamps=True) merges every default
# of the decoding's configuration into the checkpoint's.
set_by_checkpoint = {key: key in model.cfg.decoding for key in ("segment_seperators", "word_seperator", "segment_gap_threshold")}
# On CUDA, greedy TDT decoding runs its label looping in captured CUDA graphs, which call the prediction network and the
# joint from inside the graphs, past the wrappers below; without them it runs the same label looping step by step.
decoding = copy.deepcopy(model.cfg.decoding)
with open_dict(decoding):
    decoding.greedy.use_cuda_graph_decoder = False
model.change_decoding_strategy(decoding, verbose=False)

# The decoding calls the prediction network on the label each hypothesis ends with, from the state its labels before
# left, projects the outputs and the encoder's frames for the joint, and evaluates the joint on pairs of them. These
# wrappers record every row. Rows are told apart by their bytes, which equal tensors share and which the decoding copies
# without change.
calls = {}
predict, project_encoder, project_prednet, joint_after_projection = (
    model.decoder.predict, model.joint.project_encoder, model.joint.project_prednet, model.joint.joint_after_projection)


def row_key(x):
    return x.detach().contiguous().cpu().numpy().tobytes()


def recording_predict(y, state, *a, **k):
    assert y.shape[1] == 1
    # The state a row continues: none or zeros at the start of the sequence, else what an earlier row left. It is read
    # before the step, since fla's kernel may write the next state over the one it is given.
    parents = []
    for r in range(y.shape[0]):
        parent = -1
        if state is not None and any(bool(s[:, r].any()) for s in state):
            parent = calls["states"][row_key(torch.cat([s[:, r] for s in state]))]
        parents.append(parent)
    out = predict(y, state, *a, **k)
    for r, parent in enumerate(parents):
        calls["states"].setdefault(row_key(torch.cat([s[:, r] for s in out[1]])), len(calls["pred"]))
        calls["pred"].append((int(y[r, 0]), parent, out[0][r, 0].clone()))
        calls["outputs"].setdefault(row_key(out[0][r, 0]), len(calls["pred"]) - 1)
    return out


def recording_project_encoder(f):
    out = project_encoder(f)
    for x, p in zip(f.reshape(-1, f.shape[-1]), out.reshape(-1, out.shape[-1])):
        calls["projected_frames"][row_key(p)] = calls["frames"][row_key(x)]
    return out


def recording_project_prednet(g):
    out = project_prednet(g)
    for x, p in zip(g.reshape(-1, g.shape[-1]), out.reshape(-1, out.shape[-1])):
        calls["projected_steps"][row_key(p)] = calls["outputs"][row_key(x)]
    return out


def recording_joint(f, g):
    out = joint_after_projection(f, g)
    f, g, o = f.reshape(-1, f.shape[-1]), g.reshape(-1, g.shape[-1]), out.reshape(-1, out.shape[-1])
    assert f.shape[0] == g.shape[0] == o.shape[0]
    for r in range(o.shape[0]):
        calls["joint"].append((calls["projected_frames"][row_key(f[r])], calls["projected_steps"][row_key(g[r])], o[r].clone()))
    return out


model.decoder.predict, model.joint.project_encoder, model.joint.project_prednet, model.joint.joint_after_projection = (
    recording_predict, recording_project_encoder, recording_project_prednet, recording_joint)


def dump_stages(path):
    """Saves the stages of the WAVE file at `path` and returns the folder they went to."""
    samples, rate = soundfile.read(path, dtype="float32")
    assert rate == sample_rate and samples.ndim == 1, f"{path} is not mono at {sample_rate} Hz"
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(args.out_dir, args.model, name)
    os.makedirs(out, exist_ok=True)
    saved = {"audio": samples}
    # transcribe() ends with the encoder's unfreeze(), which puts it in training: dropout would run, and every
    # batch norm would update its statistics, changing the model for all that follows.
    model.eval()

    signal = torch.from_numpy(samples)[None].to(device)
    length = torch.tensor([signal.shape[1]], device=device)
    features, frames = model.preprocessor(input_signal=signal, length=length)
    frames = int(frames[0])
    saved["features"] = features[0, :, :frames].T

    hooked = {"layers": []}
    hooks = [model.encoder.pre_encode.register_forward_hook(lambda m, a, o: hooked.update(subsampled=o[0]))]
    hooks += [layer.register_forward_hook(lambda m, a, o: hooked["layers"].append(o)) for layer in model.encoder.layers]
    encoded, encoded_length = model.encoder(audio_signal=features, length=torch.tensor([frames], device=device))
    for h in hooks:
        h.remove()
    t = int(encoded_length[0])
    saved["subsampled"] = hooked["subsampled"][0, :t]
    saved["layers"] = torch.stack([layer[0, :t] for layer in hooked["layers"]])
    saved["encoded"] = encoded[0, :, :t].T
    assert torch.equal(saved["layers"][-1], saved["encoded"])

    # The decoding projects every frame of the encoder's output for the joint, those past the length too, which the
    # encoder's output on CUDA holds; the joint reads none of them.
    frames_of = {}
    for i in range(encoded.shape[2]):
        frames_of.setdefault(row_key(encoded[0, :, i]), i)
    calls.update(pred=[], states={}, outputs={}, frames=frames_of, projected_frames={}, projected_steps={}, joint=[])
    hypothesis = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=encoded, encoded_lengths=encoded_length,
                                                                return_hypotheses=True)[0]
    saved["pred_labels"] = np.array([label for label, _, _ in calls["pred"]], dtype=np.int32)
    saved["pred_parents"] = np.array([parent for _, parent, _ in calls["pred"]], dtype=np.int32)
    saved["pred_output"] = torch.stack([g for _, _, g in calls["pred"]])
    saved["joint_frames"] = np.array([frame for frame, _, _ in calls["joint"]], dtype=np.int32)
    saved["joint_steps"] = np.array([step for _, step, _ in calls["joint"]], dtype=np.int32)
    # The joint applies its log-softmax over every output, the tokens, the blank and the durations together, on the
    # CPU alone, which the checks compare with; on CUDA it returns the logits, which the dump turns into the CPU's.
    saved["joint_output"] = torch.log_softmax(torch.stack([o for _, _, o in calls["joint"]]), dim=-1)
    ids = [int(i) for i in hypothesis.y_sequence.tolist() if i != model.decoding.blank_id]
    saved["ids"] = np.array(ids, dtype=np.int32)
    text = hypothesis.text
    assert model.decoding.decode_tokens_to_str_with_strip_punctuation(ids) == text
    # transcribe() computes the encoder again, which on CUDA need not give the same bytes, so it runs unrecorded.
    model.decoder.predict, model.joint.project_encoder, model.joint.project_prednet, model.joint.joint_after_projection = (
        predict, project_encoder, project_prednet, joint_after_projection)
    official = model.transcribe([path], batch_size=1, verbose=False)[0].text
    model.decoder.predict, model.joint.project_encoder, model.joint.project_prednet, model.joint.joint_after_projection = (
        recording_predict, recording_project_encoder, recording_project_prednet, recording_joint)
    assert text == official, f"the stages give {text!r}, transcribe() gives {official!r}"

    for key, value in saved.items():
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        np.save(os.path.join(out, f"{key}.npy"), np.ascontiguousarray(array).astype(np.int32 if array.dtype.kind in "iu" else np.float32))
    meta = {
        "nemo": NEMO, "fla": FLA, "model": pin, "device": torch.cuda.get_device_name(0), "audio": os.path.basename(path),
        "samples": int(samples.shape[0]), "seconds": samples.shape[0] / sample_rate, "frames": frames, "encoded_frames": t,
        "prediction_steps": len(saved["pred_labels"]), "joint_evaluations": len(saved["joint_frames"]), "text": text,
    }
    with open(os.path.join(out, "text.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(text)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("nemo", "fla", "model")}, ensure_ascii=False))
    return out


def dump_times(path, out):
    """Saves the times transcribe(timestamps=True) gives for the WAVE file at `path` into the dump folder `out`."""
    model.decoder.predict, model.joint.project_encoder, model.joint.project_prednet, model.joint.joint_after_projection = (
        predict, project_encoder, project_prednet, joint_after_projection)
    hypothesis = model.transcribe([path], batch_size=1, timestamps=True, verbose=False)[0]
    model.decoder.predict, model.joint.project_encoder, model.joint.project_prednet, model.joint.joint_after_projection = (
        recording_predict, recording_project_encoder, recording_project_prednet, recording_joint)
    ids = [int(i) for i in hypothesis.y_sequence.tolist()]
    assert model.decoding.blank_id not in ids
    assert ids == np.load(os.path.join(out, "ids.npy")).tolist(), f"transcribe(timestamps=True) gives other ids for {path}"
    with open(os.path.join(out, "text.txt"), encoding="utf-8", newline="") as f:
        assert hypothesis.text == f.read(), f"transcribe(timestamps=True) gives another text for {path}"
    stamps = hypothesis.timestamp
    timestep = [int(t) for t in stamps["timestep"]]
    duration = [int(d) for d in hypothesis.token_duration]
    chars, words, segments = stamps["char"], stamps["word"], stamps["segment"]
    assert len(timestep) == len(chars) == len(ids) == len(duration)

    def pairs(entries, *keys):
        return [[entry[k] for k in keys] for entry in entries]

    arrays = {
        "token_timestep": np.array(timestep, dtype=np.int32),
        "token_duration": np.array(duration, dtype=np.int32),
        "token_offsets": np.array(pairs(chars, "start_offset", "end_offset"), dtype=np.int32).reshape(-1, 2),
        "token_seconds": np.array(pairs(chars, "start", "end"), dtype=np.float64).reshape(-1, 2),
        "segment_offsets": np.array(pairs(segments, "start_offset", "end_offset"), dtype=np.int32).reshape(-1, 2),
        "segment_seconds": np.array(pairs(segments, "start", "end"), dtype=np.float64).reshape(-1, 2),
    }
    for key, array in arrays.items():
        np.save(os.path.join(out, f"{key}.npy"), np.ascontiguousarray(array))
    assert all("\n" not in s["segment"] for s in segments)
    with open(os.path.join(out, "segments.txt"), "w", encoding="utf-8", newline="") as f:
        f.write("".join(s["segment"] + "\n" for s in segments))
    times = {
        "nemo": NEMO, "model": pin,
        "segment_seperators": list(model.decoding.segment_seperators), "word_seperator": model.decoding.word_seperator,
        "segment_gap_threshold": model.decoding.segment_gap_threshold, "set_by_checkpoint": set_by_checkpoint,
        "supported_punctuation": sorted(model.decoding.supported_punctuation),
        "tokenizer_type": model.decoding.tokenizer_type,
        "subsampling_factor": int(model.encoder.subsampling_factor),
        "window_stride": float(model.cfg.preprocessor.window_stride),
        "timestep": timestep, "token_duration": duration,
        "pieces": model.decoding.decode_ids_to_tokens(ids),
        "char": [dict(c, char=list(c["char"])) for c in chars], "word": words, "segment": segments,
    }
    with open(os.path.join(out, "times.json"), "w", encoding="utf-8") as f:
        json.dump(times, f, ensure_ascii=False, indent=1)
    print(json.dumps({"dump": out, "tokens": len(ids), "segments": len(segments)}, ensure_ascii=False))


for path in args.audio:
    dump_times(path, dump_stages(path))
