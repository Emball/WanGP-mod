# AGENTS.md

## Purpose
Fork of deepbeepmeep/Wan2GP (WanGP, a Gradio video/audio generation UI). The fork adds MiniMax H3
loader and conditioning features on top of stock upstream. Spec of every change: `docs/MODS.md` (read it first).
Version: `VERSION`.

## Remotes and branches
- `upstream` = deepbeepmeep/Wan2GP, `origin` = this fork. `main` = upstream merged with the fork mods.
- Upstream sync is by merge, never rebase. Upstream is LF; keep LF.

## Fork surface
All behavioural changes live in `models/minimax_h3/`:
`minimax_h3_main.py` (loaders), `minimax_h3_handler.py` (model-def keys, UI settings, downloads),
`text_encoder.py` (ClipProj), `pipeline.py` (RefMods injection), `transformer.py` (LoRA keys),
`refmods.py` and `qwen3_vl_4b_config.json` (new).
Also fork-owned: `README.md`, `AGENTS.md`, `VERSION`, `docs/MODS.md`, `tools/`, `examples/`, `scripts/install_uv.bat`.

Keep upstream files untouched outside this list. A new mod gets an ID and section in `docs/MODS.md`
and a marker in `MOD_MARKERS` in `tools/sync_upstream.py`, in the same commit.

## Sync workflow
1. `python tools/sync_upstream.py` — fetches, merges without committing, keeps the fork `README.md`, lists conflicts, verifies markers.
2. Resolve conflicts in the fork-surface files by re-applying the mod's intent from `docs/MODS.md` onto upstream's new code.
3. Bump `VERSION`, update `docs/MODS.md` if a mod's code changed shape, commit.

## Non-obvious
- Model-def keys the fork adds: `clipproj_file`, `tokenizer_folder`. Custom setting id: `h3_refmods`.
- Upstream `.gitignore` excludes `finetunes/`, `settings/`, `wgp_config.json`, `loras_url_cache_v2.json`, `*.zip`, `*.safetensors`; that is why examples live in `examples/finetune_defs/`.
- Local-only text-encoder assets: ClipProj models never trigger Hugging Face downloads (M4).
- Fork has no dependency changes; `requirements.txt` is upstream's.
