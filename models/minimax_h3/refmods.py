import json
import re
from pathlib import Path

H3_REFMODS_SETTING = "h3_refmods"

ROOT = Path(__file__).resolve().parents[2]
SEARCH_DIRS = (ROOT / "refmods", ROOT / "ckpts" / "refmods", ROOT / "loras" / "refmods", ROOT / "loras" / "minimax_h3" / "refmods", ROOT / "loras" / "refmods_plugin" / "minimax_h3", ROOT)
META_KEYS = ("refmod_meta", "audio_refmod_meta")
VISUAL_KINDS = ("image", "video")
PARTS = ("all", "visual", "audio")
MAX_WEIGHT = 10.0
RESERVED_NAME_CHARS = ",;#:\n"

_TOKEN = re.compile(r"^(?P<name>.+?)(?:#(?P<part>visual|audio|all))?(?::(?P<strength>\d*\.?\d+))?$", re.IGNORECASE)


def parse_refmod_spec(text):
    entries = []
    for raw in re.split(r"[,;\n]+", text or ""):
        raw = raw.strip().strip('"')
        if not raw:
            continue
        match = _TOKEN.match(raw)
        if match is None:
            raise ValueError(f"RefMods: cannot read '{raw}'. Use name, name:0.8 or name#audio:0.8")
        strength = float(match.group("strength")) if match.group("strength") else 1.0
        if not 0.0 <= strength <= MAX_WEIGHT:
            raise ValueError(f"RefMods: weight for '{match.group('name')}' must be between 0 and {MAX_WEIGHT:g}")
        entries.append((match.group("name").strip(), (match.group("part") or "all").lower(), strength))
    return entries


def serialize_refmod_spec(entries):
    parts = []
    for name, part, weight in entries:
        name = (name or "").strip()
        if not name or any(char in name for char in RESERVED_NAME_CHARS):
            continue
        parts.append(name + (f"#{part}" if part in PARTS and part != "all" else "") + (f":{min(max(float(weight), 0.0), MAX_WEIGHT):g}" if float(weight) != 1.0 else ""))
    return ", ".join(parts)


def _available_files():
    seen = {}
    for folder in SEARCH_DIRS[:-1]:
        if folder.is_dir():
            for path in folder.rglob("*.safetensors"):
                seen.setdefault(str(path), path)
    return sorted(seen.values())


def list_refmod_choices():
    names = []
    for folder in SEARCH_DIRS[:-1]:
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.safetensors")):
            name = path.relative_to(folder).with_suffix("").as_posix()
            if any(char in name for char in RESERVED_NAME_CHARS):
                print(f"[RefMods] '{name}' is hidden from the selector: names cannot contain , ; # or :")
            elif name not in names:
                names.append(name)
    return names


def resolve_refmod_path(name):
    wanted = Path(name)
    names = [wanted] if wanted.suffix.lower() == ".safetensors" else [wanted.with_name(wanted.name + ".safetensors")]
    candidates = [n for n in names if n.is_absolute()]
    if not candidates:
        candidates = [folder / n for folder in SEARCH_DIRS for n in names]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if not wanted.is_absolute():
        target = names[0].name.lower()
        for path in _available_files():
            if path.name.lower() == target:
                return path
    available = [str(p.relative_to(ROOT)) if ROOT in p.parents else str(p) for p in _available_files()][:20]
    folders = ", ".join(str(f.relative_to(ROOT)) for f in SEARCH_DIRS[:-1])
    raise FileNotFoundError(f"RefMod '{name}' was not found. Put it in one of: {folders} (inside {ROOT}). Found: {available or 'no RefMod files'}")


def _read_meta(handle, path):
    metadata = handle.metadata() or {}
    for key in META_KEYS:
        if key in metadata:
            try:
                return json.loads(metadata[key])
            except ValueError:
                return None
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except ValueError:
            return None
    return None


def _infer_kind(latent):
    if latent.ndim == 4:
        return "audio"
    if latent.ndim == 5:
        return "image" if latent.shape[2] == 1 else "video"
    raise ValueError(f"unsupported latent shape {tuple(latent.shape)}")


def _validate(latent, kind, label):
    if kind == "audio":
        valid = latent.ndim == 4 and tuple(latent.shape[:3]) == (1, 32, 2) and latent.shape[-1] > 0
    elif kind in VISUAL_KINDS:
        valid = (latent.ndim == 5 and tuple(latent.shape[:2]) == (1, 24) and all(n > 0 for n in latent.shape[2:])
                 and all(n % 2 == 0 for n in latent.shape[-2:]) and (kind != "image" or latent.shape[2] == 1))
    else:
        valid = False
    if not valid:
        raise ValueError(f"RefMod {label}: unexpected {kind} latent shape {tuple(latent.shape)}")


def _blur(latent, factor=8):
    import torch.nn.functional as F
    if latent.ndim == 4:
        b, c, stereo, t = latent.shape
        if t <= 1:
            return latent
        flat = latent.reshape(b * c * stereo, 1, t).float()
        down = F.adaptive_avg_pool1d(flat, max(1, t // factor))
        return F.interpolate(down, size=t, mode="linear", align_corners=False).reshape_as(latent).to(latent.dtype)
    t, h, w = latent.shape[2:]
    down = F.adaptive_avg_pool3d(latent.float(), (t, max(1, h // factor), max(1, w // factor)))
    return F.interpolate(down, size=(t, h, w), mode="trilinear", align_corners=False).to(latent.dtype)


def _apply_strength(latent, strength):
    if strength >= 1.0:
        return latent
    return strength * latent + (1.0 - strength) * _blur(latent)


def _read_file(path, framework):
    from safetensors import safe_open
    with safe_open(str(path), framework=framework, device="cpu") as handle:
        meta = _read_meta(handle, path)
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    members = []
    if isinstance(meta, dict) and meta.get("kind") == "bundle":
        for index, member in enumerate(meta.get("members") or []):
            latent = tensors.get(f"ref_{index}")
            if latent is None:
                raise ValueError(f"RefMod {path.name}: bundle tensor ref_{index} is missing")
            members.append((member, latent))
    else:
        latent = tensors.get("latent")
        if latent is None:
            if len(tensors) != 1:
                raise ValueError(f"RefMod {path.name}: no 'latent' tensor (found {sorted(tensors)})")
            latent = next(iter(tensors.values()))
        members.append((meta if isinstance(meta, dict) else {}, latent))
    return members


def load_refmods(text, framework="pt"):
    mods = []
    for name, part, strength in parse_refmod_spec(text):
        path = resolve_refmod_path(name)
        for index, (meta, latent) in enumerate(_read_file(path, framework)):
            kind = meta.get("kind") or _infer_kind(latent)
            label = f"{path.name}[{index}]"
            _validate(latent, kind, label)
            if (part == "visual" and kind == "audio") or (part == "audio" and kind != "audio") or strength <= 0.0:
                continue
            whole = int(strength)
            for level in [1.0] * whole + ([strength - whole] if strength - whole > 1e-6 else []):
                mods.append({"name": meta.get("name") or path.stem, "kind": kind, "latent": _apply_strength(latent, level), "strength": level, "file": path.name})
    if mods:
        summary = ", ".join(f"{m['name']} ({m['kind']} {'x'.join(str(n) for n in m['latent'].shape[2:])}, {m['strength']:g})" for m in mods)
        print(f"[RefMods] loaded {len(mods)}: {summary}")
    return mods


def ref_entry(mod):
    shape = mod["latent"].shape
    if mod["kind"] == "audio":
        return {"kind": "audio", "ref_audio_t": shape[-1]}
    entry = {"kind": mod["kind"], "latent_h": shape[-2], "latent_w": shape[-1]}
    if mod["kind"] == "video":
        entry.update({"latent_t": shape[2], "ref_audio_t": 0})
    return entry


def split_mods(mods):
    return [m for m in mods if m["kind"] != "audio"], [m for m in mods if m["kind"] == "audio"]
