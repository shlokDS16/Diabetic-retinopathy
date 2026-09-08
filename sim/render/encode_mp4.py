"""Encode a rendered PNG sequence to H.264 MP4 (imageio-ffmpeg).
Run (nrdi-env): python sim/render/encode_mp4.py sim/render/out/explode_seq_cycles explode_cycles.mp4 [fps]"""
import sys, pathlib, imageio.v3 as iio, imageio
seq = pathlib.Path(sys.argv[1]); out = seq.parent / sys.argv[2]; fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
frames = sorted(seq.glob("frame_*.png"))
w = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8)
for f in frames:
    w.append_data(iio.imread(f)[..., :3])
w.close(); print(f"wrote {out} ({len(frames)} frames @ {fps} fps)")
