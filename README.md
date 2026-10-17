# Wan2GP-mod

A fork of [deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP) (WanGP) that extends the
**MiniMax H3** model family with ComfyUI-format checkpoint loading, a Qwen3-VL-4B text-encoder path,
LoRA key compatibility and RefMods. Everything else is stock upstream, kept in sync by merge.

For WanGP itself (features, supported models, usage) see the upstream README.

## What this fork adds

| ID | Feature |
|----|---------|
| M1 | Loads ComfyUI-style single-file Qwen3-VL text-encoder checkpoints (key normalization) |
| M2 | Qwen3-VL-4B + ClipProj conditioning: a 4B hidden-state tap projected into H3's 5120-d space |
| M3 | Tokenizer resolved from local files only (folder or standalone `tokenizer.json`), never downloaded |
| M4 | ClipProj models skip the automatic Hugging Face downloads |
| M5 | Audio VAEs with fused weight-norm weights load correctly |
| M6 | LoRAs with flat `final_layer_audio_out` / `final_layer_video_out` keys load correctly |
| M7 | RefMods: inject precomputed reference latents (visual/audio, with strength) via a settings field |

Exact definitions, formats and code locations: [docs/MODS.md](docs/MODS.md).

## Using the 4B + ClipProj model

1. Put the weights in `ckpts/`: the H3 transformer, `qwen3vl_4b_fp8_scaled.safetensors`, the ClipProj
   `.safetensors`, and the H3 video and audio VAEs.
2. Make sure a Qwen3-VL tokenizer exists locally (for example `ckpts/Qwen3-VL-4B-Instruct/tokenizer.json`),
   or set `tokenizer_folder` in the model definition.
3. Copy `examples/finetune_defs/minimax_h3_fl2va_pruned_4b_clipproj.json` into `finetunes/` and adjust filenames.

## Install

Same as upstream (`scripts/install.bat` / `install.sh`). Windows users can use `scripts/install_uv.bat`,
which bootstraps Python 3.11 through `uv`.

Clone with `git config core.autocrlf input` on Windows so files stay LF in the repository.

## Staying in sync with upstream

```
python tools/sync_upstream.py
```

Merges `upstream/main`, keeps this README, lists any conflicts, and verifies every mod is still present.
`python tools/sync_upstream.py --check` runs the verification alone.

See [AGENTS.md](AGENTS.md) for the full workflow and project layout.

## License

Upstream license applies: see [LICENSE.txt](LICENSE.txt).
