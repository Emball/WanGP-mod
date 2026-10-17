"""Merge upstream Wan2GP into this fork and verify every fork mod survived.

Usage:
  python tools/sync_upstream.py          fetch + merge (no commit) + verify
  python tools/sync_upstream.py --check  verify mod markers only
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "upstream/main"
KEEP_OURS = ("README.md",)

MOD_MARKERS = {
    "M1": ("models/minimax_h3/minimax_h3_main.py", "model.language_model."),
    "M2": ("models/minimax_h3/text_encoder.py", "_apply_clipproj"),
    "M2 loader": ("models/minimax_h3/minimax_h3_main.py", "clipproj_filename"),
    "M2 handler": ("models/minimax_h3/minimax_h3_handler.py", 'model_def.get("clipproj_file")'),
    "M3": ("models/minimax_h3/minimax_h3_main.py", "_find_local_qwen_tokenizer"),
    "M4": ("models/minimax_h3/minimax_h3_handler.py", 'not model_def.get("clipproj_file")'),
    "M5": ("models/minimax_h3/minimax_h3_main.py", "norm_except_dim"),
    "M6": ("models/minimax_h3/transformer.py", "final_layer_audio_out."),
    "M7 module": ("models/minimax_h3/refmods.py", "def load_refmods"),
    "M7 pipeline": ("models/minimax_h3/pipeline.py", "split_mods(load_refmods"),
    "M7 handler": ("models/minimax_h3/minimax_h3_handler.py", "H3_REFMODS_SETTING"),
    "M7 copies": ("models/minimax_h3/refmods.py", "[1.0] * whole"),
    "M8 tab": ("models/minimax_h3/refmods_ui.py", "def build_refmods_tab"),
    "M8 handler": ("models/minimax_h3/minimax_h3_handler.py", '"tab": "refmods"'),
    "M8 wgp hook": ("wgp.py", "build_refmods_tab(custom_setting_text_inputs"),
    "M8 wgp hidden": ("wgp.py", "ignore_hidden=True"),
}


def git(*args, check=True):
    print("$ git " + " ".join(args))
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    if check and result.returncode:
        sys.exit(f"git {args[0]} failed ({result.returncode})")
    return result


def verify():
    failed = []
    for mod, (path, marker) in MOD_MARKERS.items():
        file = ROOT / path
        ok = file.is_file() and marker in file.read_text(encoding="utf-8", errors="replace")
        print(f"[{'ok' if ok else 'MISSING'}] {mod}: {path} :: {marker}")
        if not ok:
            failed.append(mod)
    return failed


def sync():
    if git("status", "--porcelain").stdout.strip():
        sys.exit("Working tree is not clean; commit or stash first.")
    git("fetch", "upstream")
    result = git("merge", "--no-commit", "--no-ff", UPSTREAM, check=False)
    conflicts = git("diff", "--name-only", "--diff-filter=U").stdout.split()
    for name in KEEP_OURS:
        if name in conflicts:
            print(f"Keeping fork version of {name}")
            git("checkout", "--ours", name)
            git("add", name)
            conflicts.remove(name)
    if conflicts:
        print("\nUnresolved conflicts (resolve, re-check mods against docs/MODS.md, then commit):")
        for name in conflicts:
            print("  " + name)
    elif result.returncode:
        print("Merge reported a problem but no conflicted files remain; inspect `git status`.")
    failed = verify()
    if failed:
        print("\nMods missing after merge: " + ", ".join(failed))
    if conflicts or failed:
        sys.exit(1)
    print("\nMerge staged cleanly. Bump VERSION, update AGENTS.md/docs/MODS.md if needed, then commit.")


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        sys.exit(1 if verify() else 0)
    sync()
