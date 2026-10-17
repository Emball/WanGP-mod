"""Qwen3-VL conditioning used by MiniMax H3."""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer

from models.ideogram4.qwen3_vl_configuration import Qwen3VLConfig
from models.ideogram4.qwen3_vl_transformers import Qwen3VLTextModel, Qwen3VLVisionModel

from .interrupt import GenerationInterrupted
from shared.utils.phase_progress import text_encoding_progress

VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655
VIDEO_PAD = 151656
TEXT_PAD = 151643


class _RawHiddenState(nn.Module):
    def forward(self, hidden_states):
        return hidden_states[0] if isinstance(hidden_states, list) else hidden_states


def load_h3_qwen_config(config_path):
    config = Qwen3VLConfig.from_json_file(config_path)
    config.text_config.num_hidden_layers = 50
    config.text_config.use_cache = False
    config.vision_config.out_hidden_size = config.text_config.hidden_size
    return config


def load_h3_qwen4b_config(config_path):
    """Load the stock Qwen3-VL-4B config used by MiniMax-H3 ClipProj.

    The 4B model is a normal 36-layer Qwen3-VL text/vision model. Unlike the
    official H3 encoder it is not truncated to 50 layers; ClipProj reads an
    intermediate hidden state (normally layer 24) and maps its 2560 channels
    into H3's 5120-channel conditioning space.
    """
    config = Qwen3VLConfig.from_json_file(config_path)
    config.text_config.use_cache = False
    config.vision_config.out_hidden_size = config.text_config.hidden_size
    if config.text_config.hidden_size != 2560 or config.text_config.num_hidden_layers != 36:
        raise ValueError(
            "The bundled H3 ClipProj config is expected to be Qwen3-VL-4B "
            "(hidden_size=2560, num_hidden_layers=36), but loaded "
            f"hidden_size={config.text_config.hidden_size}, "
            f"num_hidden_layers={config.text_config.num_hidden_layers}."
        )
    return config


_PROJECTION_META_TYPES = {
    "tap": int,
    "d_in": int,
    "d_out": int,
}


def _load_clipproj_projection(filename):
    """Load a ClipProj .safetensors matrix into CPU tensors.

    Supports both the original linear+MLP files and newer residual-only MLP
    files. Metadata values are stored as strings by safetensors, so restore
    the small set of numeric fields that affect execution.
    """
    filename = os.path.abspath(filename)
    data = load_safetensors(filename, device="cpu")
    with safe_open(filename, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    for key, value in metadata.items():
        caster = _PROJECTION_META_TYPES.get(key)
        if caster is not None:
            try:
                data[key] = caster(value)
            except (TypeError, ValueError):
                pass

    for key in ("mean_in", "std_in", "mean_out", "std_out", "tap"):
        if key not in data:
            raise KeyError(f"ClipProj file '{filename}' is missing required key '{key}'")
    for key in ("W", "mean_in", "std_in", "mean_out", "std_out"):
        if key in data and torch.is_tensor(data[key]):
            data[key] = data[key].float()
    return data


def _build_clipproj_residual(proj, device):
    layers = sorted({
        int(key.split(".")[1])
        for key in proj
        if key.startswith("mlp.") and len(key.split(".")) >= 3 and key.split(".")[1].isdigit()
    })
    if not layers:
        return None
    modules = []
    for pos, index in enumerate(layers):
        weight = proj[f"mlp.{index}.weight"]
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=f"mlp.{index}.bias" in proj)
        linear.weight.data.copy_(weight.to(device=device))
        if linear.bias is not None:
            linear.bias.data.copy_(proj[f"mlp.{index}.bias"].to(device=device))
        modules.append(linear)
        if pos < len(layers) - 1:
            modules.append(nn.GELU())
    network = nn.Sequential(*modules).to(device=device)
    network.eval().requires_grad_(False)
    return network


def _apply_clipproj(proj, hidden):
    """Map Qwen3-VL-4B activations into MiniMax-H3 conditioning space."""
    if hidden.ndim != 3:
        raise ValueError(f"ClipProj expected [batch, seq, hidden], got {tuple(hidden.shape)}")
    d_in = int(proj["mean_in"].shape[0])
    d_out = int(proj["mean_out"].shape[0])
    if hidden.shape[-1] != d_in:
        raise ValueError(f"ClipProj input mismatch: encoder produced {hidden.shape[-1]} channels, projection expects {d_in}")
    if d_out != 5120:
        raise ValueError(f"MiniMax H3 ClipProj must produce 5120 channels, projection produces {d_out}")

    device = hidden.device
    x = hidden.float()
    mean_in = proj["mean_in"].to(device=device)
    std_in = proj["std_in"].to(device=device)
    mean_out = proj["mean_out"].to(device=device)
    std_out = proj["std_out"].to(device=device)
    normalized = (x - mean_in) / std_in

    if "W" in proj:
        projected = normalized @ proj["W"].to(device=device)
    else:
        # v3/v3.1 residual-only matrices deliberately omit W.
        projected = torch.zeros((*x.shape[:-1], d_out), device=device, dtype=torch.float32)

    residual = _build_clipproj_residual(proj, device)
    if residual is not None:
        residual_dtype = next(residual.parameters()).dtype
        projected = projected + residual(normalized.to(dtype=residual_dtype)).float()

    return (projected * std_out + mean_out).to(dtype=hidden.dtype)


def _visual_patches(frames, video=False, patch_size=16, temporal_patch_size=2, merge_size=2):
    """Convert exactly two THWC RGB frames into the Qwen3-VL patch layout."""
    if frames.shape[0] == 1:
        frames = frames.repeat(2, 1, 1, 1)
    if frames.shape[0] != temporal_patch_size:
        raise ValueError(f"Qwen visual blocks require {temporal_patch_size} frames, got {frames.shape[0]}")
    _, height, width, channels = frames.shape
    if channels != 3:
        raise ValueError(f"Qwen visual input must be RGB, got {channels} channels")
    factor = patch_size * merge_size
    min_pixels, max_pixels = ((4096, 25165824) if video else (65536, 16777216))
    target_h = max(factor, round(height / factor) * factor)
    target_w = max(factor, round(width / factor) * factor)
    if target_h * target_w > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        target_h = max(factor, math.floor(height / scale / factor) * factor)
        target_w = max(factor, math.floor(width / scale / factor) * factor)
    elif target_h * target_w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        target_h = math.ceil(height * scale / factor) * factor
        target_w = math.ceil(width * scale / factor) * factor

    images = F.interpolate(frames.permute(0, 3, 1, 2), size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True)
    images = images.mul(2.0).sub_(1.0)
    grid_h, grid_w = target_h // patch_size, target_w // patch_size
    patches = images.reshape(1, temporal_patch_size, 3, grid_h // merge_size, merge_size, patch_size,
                             grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flattened = patches.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    grid = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long, device=frames.device)
    return flattened, grid


def _mrope_positions(visuals, seq_len, device):
    if not visuals:
        return None
    positions = torch.zeros((3, seq_len), dtype=torch.long, device=device)
    cursor = offset = 0
    for visual in visuals:
        start, size, grid = visual["index"], visual["size"], visual["grid"]
        if cursor < start:
            positions[:, cursor:start] = torch.arange(cursor + offset, start + offset, device=device)
        end = start + size
        max_grid = int(grid.max()) // 2
        positions[0, start:end] = start + offset
        rows = int(grid[0, 1]) // 2
        cols = int(grid[0, 2]) // 2
        positions[1, start:end] = torch.arange(start + offset, start + rows + offset, device=device).unsqueeze(1).expand(rows, cols).reshape(-1)[:size]
        positions[2, start:end] = torch.arange(start + offset, start + cols + offset, device=device).unsqueeze(0).expand(rows, cols).reshape(-1)[:size]
        offset += max_grid - size
        cursor = end
    if cursor < seq_len:
        positions[:, cursor:] = torch.arange(cursor + offset, seq_len + offset, device=device)
    return positions.unsqueeze(1)


class MiniMaxH3TextEncoder(nn.Module):
    """Qwen3-VL conditioning used by MiniMax H3.

    The same class handles the stock 32B/50-layer encoder and the optional
    4B+ClipProj path. The latter captures an intermediate 4B hidden state and
    projects it into the 5120-dimensional H3 conditioning space.
    """

    def __init__(self, config, clipproj_filename=None, clipproj_tap=24):
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.language_model = Qwen3VLTextModel(config.text_config)
        self.language_model.norm = _RawHiddenState()
        self.tokenizer = None
        self._interrupt = False
        self.clipproj_filename = clipproj_filename
        self._clipproj = _load_clipproj_projection(clipproj_filename) if clipproj_filename is not None else None
        self.clipproj_tap = int(self._clipproj.get("tap", clipproj_tap)) if self._clipproj is not None else None
        if self._clipproj is not None:
            print(
                f"MiniMax H3 ClipProj: 4B encoder -> 5120 conditioning, tap={self.clipproj_tap}, "
                f"d_in={self._clipproj['mean_in'].shape[0]}, d_out={self._clipproj['mean_out'].shape[0]}, "
                f"linear={'yes' if 'W' in self._clipproj else 'no'}, "
                f"residual={'yes' if any(k.startswith('mlp.') for k in self._clipproj) else 'no'}"
            )
        if self._clipproj is not None and self.clipproj_tap < 0:
            raise ValueError(f"Invalid ClipProj tap {self.clipproj_tap}")

    @property
    def _interrupt(self):
        return getattr(self, "_abort", False)

    @_interrupt.setter
    def _interrupt(self, value):
        self._abort = bool(value)
        if hasattr(self, "visual"):
            self.visual._interrupt = self._abort
        if hasattr(self, "language_model"):
            self.language_model._interrupt = self._abort

    def set_tokenizer(self, tokenizer_path):
        # Qwen3-VL uses the Qwen2 tokenizer backend. A complete local model
        # folder is preferred, but a standalone tokenizer.json is also enough
        # for this H3 path because we only need text -> input_ids.
        if os.path.isfile(tokenizer_path) and tokenizer_path.lower().endswith(".json"):
            from transformers import Qwen2TokenizerFast
            self.tokenizer = Qwen2TokenizerFast(
                tokenizer_file=tokenizer_path,
                unk_token="<|endoftext|>",
                eos_token="<|im_end|>",
                pad_token="<|endoftext|>",
                bos_token=None,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, local_files_only=True)

    def _token_ids(self, text):
        if self.tokenizer is None:
            raise RuntimeError("The Qwen tokenizer has not been initialized")
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _presentation(self, prompt, items):
        entries = []
        counters = {"image": 0, "audio": 0, "video": 0}

        def add_text(text):
            entries.extend(self._token_ids(text))

        def add_visual(frames, video=False):
            entries.extend((VISION_START, {"frames": frames, "video": video}, VISION_END))

        for item in items:
            kind = item["type"]
            counters[kind] += 1
            if kind == "image":
                add_text(f"<Picture {counters[kind]}>: ")
                add_visual(item["frames"])
            elif kind == "audio":
                add_text(f"<Audio {counters[kind]}>: ")
            elif kind == "video":
                add_text(f"<Video {counters[kind]}>: ")
                frames = item["frames"]
                timestamps = list(item.get("timestamps", [i / 2.0 for i in range(frames.shape[0])]))
                if frames.shape[0] % 2:
                    frames = torch.cat((frames, frames[-1:]), dim=0)
                    timestamps.append(timestamps[-1])
                for index in range(0, frames.shape[0], 2):
                    add_text(f"<{(timestamps[index] + timestamps[index + 1]) / 2.0:.1f} seconds>")
                    add_visual(frames[index:index + 2], video=True)
            else:
                raise ValueError(f"Unsupported H3 presentation item: {kind}")
        add_text(prompt)
        return entries or [TEXT_PAD]

    def encode(self, prompt, items, device, dtype):
        entries = self._presentation(prompt, items)
        token_ids = []
        visual_specs = []
        for entry in entries:
            if isinstance(entry, int):
                token_ids.append(entry)
            else:
                visual_specs.append({"placeholder": len(token_ids), "frames": entry["frames"], "video": entry["video"]})
                token_ids.append(VIDEO_PAD if entry["video"] else IMAGE_PAD)

        encoded_visuals = []
        for spec in visual_specs:
            if self._interrupt:
                raise GenerationInterrupted
            frames = spec["frames"].to(device=device, dtype=torch.float32)
            flattened, grid = _visual_patches(frames, video=spec["video"])
            self.visual._interrupt = self._interrupt
            merged, deepstack = self.visual(flattened, grid)
            if merged is None:
                raise GenerationInterrupted
            encoded_visuals.append({"placeholder": spec["placeholder"], "merged": merged, "deepstack": deepstack,
                                    "grid": grid, "video": spec["video"]})

        expanded_ids = []
        visuals = []
        visual_by_placeholder = {v["placeholder"]: v for v in encoded_visuals}
        for index, token_id in enumerate(token_ids):
            visual = visual_by_placeholder.get(index)
            if visual is None:
                expanded_ids.append(token_id)
                continue
            start = len(expanded_ids)
            size = visual["merged"].shape[0]
            expanded_ids.extend([VIDEO_PAD if visual["video"] else IMAGE_PAD] * size)
            visual["index"] = start
            visual["size"] = size
            visuals.append(visual)

        input_ids = torch.tensor(expanded_ids, dtype=torch.long, device=device).unsqueeze(0)
        embeds = self.language_model.embed_tokens(input_ids)
        visual_mask = torch.zeros((1, embeds.shape[1]), dtype=torch.bool, device=device)
        tags = torch.ones(embeds.shape[1], dtype=torch.long, device=device)
        deepstack = None
        for visual in visuals:
            start, end = visual["index"], visual["index"] + visual["size"]
            embeds[0, start:end] = visual["merged"].to(dtype=embeds.dtype)
            visual_mask[0, start:end] = True
            tags[max(0, start - 1):min(tags.shape[0], end + 1)] = 0
            deepstack = visual["deepstack"] if deepstack is None else [torch.cat((deepstack[i], visual["deepstack"][i]), dim=0) for i in range(len(deepstack))]

        positions = _mrope_positions(visuals, embeds.shape[1], device)
        self.language_model._interrupt = self._interrupt
        return_layers = [self.clipproj_tap] if self._clipproj is not None else None
        if return_layers is not None and self.clipproj_tap >= len(self.language_model.layers):
            raise ValueError(
                f"ClipProj tap {self.clipproj_tap} is out of range for a {len(self.language_model.layers)}-layer encoder"
            )
        with text_encoding_progress(self.language_model.layers):
            output = self.language_model(inputs_embeds=embeds, position_ids=positions, visual_pos_masks=visual_mask if visuals else None,
                                         deepstack_visual_embeds=deepstack, use_cache=False,
                                         return_mid_results_layers=return_layers)
        if output.last_hidden_state is None:
            raise GenerationInterrupted
        if self._clipproj is not None:
            if not getattr(output, "mid_results", None):
                raise RuntimeError(f"Qwen3-VL did not return ClipProj tap {self.clipproj_tap}")
            hidden = output.mid_results[0]
            return _apply_clipproj(self._clipproj, hidden), tags
        return output.last_hidden_state.to(dtype), tags


__all__ = ["MiniMaxH3TextEncoder", "load_h3_qwen_config", "load_h3_qwen4b_config"]
