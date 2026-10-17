# Fork modifications

Authoritative definition of everything this fork changes relative to upstream
[deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP). All code changes live in
`models/minimax_h3/`, except the three-line M8 hook in `wgp.py`, which is inert unless a model def opts in.

Baseline: upstream commit `6479db36` (2026-10-07, WanGP 17.17). The original mod set was
diffed against that commit: 5 upstream files modified, 2 files added.

Every mod is opt-in by data (a model-def key, a checkpoint layout, or a setting value) except
where marked **shared path**.

| ID | Mod | Files |
|----|-----|-------|
| M1 | ComfyUI / single-file Qwen3-VL key normalization | `minimax_h3_main.py` |
| M2 | Qwen3-VL-4B + ClipProj conditioning | `text_encoder.py`, `minimax_h3_main.py`, `minimax_h3_handler.py` |
| M3 | Local-only tokenizer resolution | `text_encoder.py`, `minimax_h3_main.py` |
| M4 | Download gating for ClipProj models | `minimax_h3_handler.py` |
| M5 | Audio VAE fused weight-norm re-parametrization | `minimax_h3_main.py` |
| M6 | LoRA `final_layer` key normalization | `transformer.py` |
| M7 | RefMods (precomputed reference latents) | `refmods.py`, `pipeline.py`, `minimax_h3_handler.py` |
| M8 | RefMods tab beside LoRAs (dropdowns + weights) | `refmods_ui.py`, `refmods.py`, `minimax_h3_handler.py`, `wgp.py` |

---

## M1. Single-file Qwen3-VL key normalization

`_strip_text_encoder_wrapper(state_dict, quantization_map, tied_weights_map)` in `minimax_h3_main.py`.
Used as `preprocess_sd` when loading the text encoder.

Upstream picked one root (`language_model` or empty) from the sentinel key
`model.embed_tokens.weight` and applied `offload.map_state_dict(rules={"model": root})`.

Fork: each key of all three mappings is rewritten independently, first match wins:

| Input prefix | Output prefix |
|---|---|
| `model.language_model.` | `language_model.` |
| `language_model.model.` | `language_model.` |
| `model.layers.` / `model.embed_tokens.` / `model.norm.` | `language_model.` + rest |
| `model.visual.` | `visual.` |
| anything else | unchanged |

Purpose: load ComfyUI-style single-file Qwen3-VL checkpoints (plain `model.layers.*` root).
**Shared path:** also runs for the stock 32B encoder.

Equivalence with upstream: for the official layout (`model.language_model.*`, `model.visual.*`) the output
is identical. For the plain-LM layout, `model.visual.*` now maps to `visual.*` (upstream produced
`language_model.visual.*`). Other `model.*` keys are left unchanged (upstream rewrote every `model.` prefix).

Known gap: upstream's `map_state_dict` also rewrites key names held in list *values*, which is how
`tied_weights_map` records tied parameters. `normalize` renames dict keys only, so tied-parameter names
inside `tied_weights_map` values are not remapped.

## M2. Qwen3-VL-4B + ClipProj conditioning

Replaces the 32B/50-layer encoder output with a projection of a 4B hidden state into H3's
5120-channel conditioning space. Enabled per model by the model-def key `clipproj_file`.

Model-def keys (read in `minimax_h3_handler.py`, passed through `load_h3_pipeline`):
- `clipproj_file` — ClipProj `.safetensors`, resolved with `fl.locate_file` (bare name searches `ckpts/`).
- `tokenizer_folder` — optional, see M3.
- `text_encoder_URLs` — the 4B single-file checkpoint (e.g. `qwen3vl_4b_fp8_scaled.safetensors`).

Encoder (`text_encoder.py`):
- `load_h3_qwen4b_config(path)` loads `models/minimax_h3/qwen3_vl_4b_config.json` and asserts
  `hidden_size == 2560` and `num_hidden_layers == 36`; sets `vision_config.out_hidden_size = hidden_size`.
- `MiniMaxH3TextEncoder(config, clipproj_filename=None, clipproj_tap=24)`. With a ClipProj file,
  the forward pass requests `return_mid_results_layers=[tap]`, takes `mid_results[0]`, and returns
  `_apply_clipproj(...)` instead of `last_hidden_state`. Tap is the file's `tap` value (safetensors
  metadata or tensor), default 24; it must be less than the layer count.
- `_load_clipproj_projection(path)` loads tensors on CPU; metadata keys `tap`, `d_in`, `d_out` are cast
  to int. Required keys: `mean_in`, `std_in`, `mean_out`, `std_out`, `tap`. `W`, means and stds are cast to float32.
- `_apply_clipproj(proj, hidden)` computes, in float32:
  1. `n = (hidden - mean_in) / std_in`
  2. `y = n @ W` if `W` exists, else zeros of width `d_out`
  3. `y += residual(n)` if keys `mlp.<i>.weight[/bias]` exist
  4. `out = (y * std_out + mean_out)` cast back to `hidden.dtype`
- `_build_clipproj_residual`: `nn.Linear` per `mlp.<i>` index in ascending order, `GELU` between
  layers, none after the last. Supports both linear+MLP files and residual-only (v3/v3.1) files.
- Validation: input must be `[batch, seq, hidden]`, `hidden == d_in` (2560 for 4B), `d_out == 5120`.

## M3. Local-only tokenizer resolution

Applies to every H3 text-encoder load (**shared path**).

`MiniMaxH3TextEncoder.set_tokenizer(path)`:
- A `.json` file path → `Qwen2TokenizerFast(tokenizer_file=...)` with `unk/pad=<|endoftext|>`, `eos=<|im_end|>`, `bos=None`.
- A folder → `AutoTokenizer.from_pretrained(..., trust_remote_code=False, local_files_only=True)`.

`_find_local_qwen_tokenizer(explicit_folder)` in `minimax_h3_main.py` never downloads. Order:
1. `explicit_folder` (model-def `tokenizer_folder`; absolute, or resolved via `fl.locate_folder`) — must contain
   `tokenizer.json`, else `FileNotFoundError`.
2. For each root in [`fl.get_smart_download_root()`, `$HF_HOME`, `$TRANSFORMERS_CACHE`, `~/.cache/huggingface/hub`]:
   `<root>/Qwen3-VL-4B-Instruct`, then `<root>/Qwen3-VL-32B-Instruct`, then the root itself, then a
   depth-limited (5) walk for a `tokenizer.json` whose path contains `qwen3-vl`.
3. Otherwise `FileNotFoundError`.

## M4. Download gating

`minimax_h3_handler.py`, download-list builder. If the model def has `clipproj_file`, the Hugging Face
downloads for the text-encoder assets (`TEXT_ENCODER_FOLDER`) and the latent upscaler are skipped.
All ClipProj assets, the tokenizer and (for two-phase generation) the latent upscaler must already be local.

## M5. Audio VAE fused weight-norm

`_load_audio_vae` in `minimax_h3_main.py`. Before loading, `expected_shapes` is captured from `vae.state_dict()`.
After the existing `weight_g`/`weight_v` renames, for every expected
`<prefix>.parametrizations.weight.original0` missing from the checkpoint:
- pop `<prefix>.weight` (skip if absent),
- `dim` = first axis of the expected shape whose size is not 1 (fallback 0),
- `original0 = torch.norm_except_dim(weight_fp32, 2, dim)`, `original1 = weight_fp32`.

Purpose: load audio VAEs saved with weight-norm already fused. Logs the converted count.

## M6. LoRA `final_layer` key normalization

`MiniMaxH3Model.preprocess_loras` in `transformer.py`, applied right after the `.lora_*.turbo.weight` rename.
Keys starting with `[""|"diffusion_model."|"transformer."] + final_layer_audio_out.` are rewritten to
`...final_layer.audio_out.`; likewise `final_layer_video_out.` → `final_layer.video_out.`.

`tools/fix_lora_keys.py <lora.safetensors> [--drop]` applies the same rename offline by rewriting the
safetensors header (`*_fixed.safetensors`), or with `--drop` removes those tensors (`*_nofinal.safetensors`).
Tensor data is copied unchanged.

## M7. RefMods

Injects precomputed reference latents (`.safetensors`) as extra visual/audio references.

UI: a text custom setting `h3_refmods` ("RefMods") defined in `minimax_h3_handler.py`. It is hidden in the form
and edited through the M8 tab. It is stored and saved like any custom setting.

Spec grammar (`parse_refmod_spec`): entries separated by `,` `;` or newline;
`name[#visual|#audio|#all][:weight]`. Weight is a float in [0, 10], default 1. `#part` filters which
member kinds are kept (default `all`). Weight 0 drops the entry.

Weight semantics (`load_refmods`): up to 1 the weight is the strength below. Above 1 the entry is repeated:
`floor(w)` full-strength copies, plus one copy at strength `w - floor(w)` when that is non-zero
(2.7 -> 1.0, 1.0, 0.7). Each copy adds its tokens to every attention block.

File lookup (`resolve_refmod_path`): `.safetensors` is appended if missing; absolute paths are used as-is;
otherwise searched under `refmods/`, `ckpts/refmods/`, `loras/refmods/`, `loras/minimax_h3/refmods/`,
`loras/refmods_plugin/minimax_h3/` (the community WanGP plugin's folder) and the repo root, then by
case-insensitive filename across the first five folders (recursive).

File formats (`_read_file`):
- single tensor: key `latent` (or the only tensor in the file); optional metadata JSON from safetensors
  metadata key `refmod_meta` / `audio_refmod_meta`, or a sidecar `<name>.json`.
- bundle: metadata `{"kind": "bundle", "members": [...]}` with tensors `ref_0 … ref_n`; each member dict supplies `kind`/`name`.

Latent shapes (`_validate`): audio `[1, 32, 2, T]`; image `[1, 24, 1, H, W]`; video `[1, 24, T, H, W]`;
`H` and `W` even. Kind is read from metadata, else inferred from rank (4 → audio, 5 → image/video by `T == 1`).

Strength (`_apply_strength`): `s * x + (1 - s) * blur(x)`, where `blur` is average-pool downsample by 8
(per spatial axis for visual, time axis for audio) followed by linear/trilinear upsample.

Pipeline (`pipeline.py`):
- Visual mods are appended to `visual_latents` and `refs` after the image/video references.
- Audio mods are appended after the audio references, with presentation entry `{"type": "audio"}`.
- Phase 2 (two-phase generation): the same mods are re-added to the phase-2 reference lists
  (visual before audio refs when phase-2 refs are rebuilt from phase-1 visual refs; both otherwise).
- `ref_entry(mod)` builds the ref descriptor (`kind`, `latent_h/w`, and `latent_t`/`ref_audio_t` as applicable).

## M8. RefMods tab

A "RefMods" tab placed directly after the LoRAs tab in the generation settings, built by
`build_refmods_tab(spec_box)` in `refmods_ui.py`.

Widgets: 8 rows of `[RefMod dropdown | Applies To (all/visual/audio) | Weight slider 0-10, step 0.05]`,
plus Refresh List and Clear All buttons. The dropdown lists `list_refmod_choices()`: every `.safetensors`
under the first five `SEARCH_DIRS` folders, as a folder-relative name without extension. Names containing
`, ; # :` are hidden (they cannot be written in the spec). The dropdown accepts custom values, so a saved
spec naming an unlisted file is preserved. The list refreshes on tab select and on Refresh List.

Sync with the setting: the tab does not hold state of its own. The hidden `h3_refmods` textbox is the single
source of truth.
- Widget `.input` (user edits only) -> `serialize_refmod_spec` -> textbox. Empty rows are skipped; part `all`
  and weight 1 are omitted from the string.
- Textbox `.change` (loading settings, switching models, Clear All) -> `parse_refmod_spec` -> widgets.
  An unreadable spec blanks the rows and is logged. Only the first 8 entries are shown (logged when more exist).

Hook in `wgp.py` (generic, opt-in by model def):
- `custom_setting_visible(..., ignore_hidden=False)`: a setting def with `"hidden": True` is not shown in the form.
  The generation-info summary passes `ignore_hidden=True`, so a hidden setting still appears there.
- After the LoRAs tab, for each custom setting def with `"tab": "refmods"`, `build_refmods_tab` is called with that
  setting's textbox.

---

## Not part of the fork

The following existed in the original working copy and are intentionally not tracked here:
runtime state (`wgp_config.json`, `settings/`, `workspaces/`, `envs.json`, `loras_url_cache_v2.json`,
`error_queue.zip`, `finetunes/`), a machine-specific `start.bat`, `transformer.py.bak`, `apply_lora_fix.py`
(superseded by M6), and the generated `scripts/uv-shim/py.cmd` (written by `scripts/install_uv.bat`).
The finetune definition is kept as `examples/finetune_defs/minimax_h3_fl2va_pruned_4b_clipproj.json`.

`scripts/install_uv.bat` is a Windows installer variant that bootstraps Python 3.11 through `uv`.
