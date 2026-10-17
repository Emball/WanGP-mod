import json, struct, sys, pathlib

RENAMES = (("final_layer_audio_out", "final_layer.audio_out"),
           ("final_layer_video_out", "final_layer.video_out"))

args = [a for a in sys.argv[1:] if a != "--drop"]
src = pathlib.Path(args[0] if args else r"E:\Wan2GP\loras\minimax_h3\NSFW_ANIME_H3_V16-step00002250.safetensors")
drop = "--drop" in sys.argv[1:]
dst = src.with_name(src.stem + ("_nofinal" if drop else "_fixed") + src.suffix)

with open(src, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    data_start = 8 + n

print("Keys containing 'final':")
for k in header:
    if "final" in k:
        print("   ", k)

new_header = {}
for k, v in header.items():
    if k == "__metadata__":
        new_header[k] = v
        continue
    if "final_layer_audio_out" in k or "final_layer_video_out" in k:
        if drop:
            continue
        for old, new in RENAMES:
            k = k.replace(old, new)
    new_header[k] = v

blob = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
blob += b" " * ((8 - len(blob) % 8) % 8)

with open(src, "rb") as f, open(dst, "wb") as out:
    f.seek(data_start)
    out.write(struct.pack("<Q", len(blob)))
    out.write(blob)
    while chunk := f.read(64 * 1024 * 1024):
        out.write(chunk)

print("Wrote", dst)
print("Keys containing 'final' in the new file:")
for k in new_header:
    if "final" in k:
        print("   ", k)
