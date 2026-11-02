#!/usr/bin/env python
"""Contact sheet of every generated bridge, labelled with its parameters."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default="Dataset/RendersNoGround/side")
    ap.add_argument("--summary", default="Dataset/bridge_summary.json")
    ap.add_argument("--out", default="Dataset/Figures/contact_sheet.png")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--cell", type=int, default=420)
    ap.add_argument("--title", default="V3 - 100 generated bridges (broadside view)")
    args = ap.parse_args()

    summary = {d["bridge_id"]: d for d in json.loads(Path(args.summary).read_text())}
    files = sorted(Path(args.renders).glob("*.png"),
                   key=lambda f: int(f.stem.split("_")[-1]))
    if not files:
        raise SystemExit(f"no renders in {args.renders}")

    cw = args.cell
    ch = int(cw * 0.64) + 44          # render + caption strip
    cols = args.cols
    rows = math.ceil(len(files) / cols)
    pad, top = 10, 64

    sheet = Image.new("RGB", (cols * (cw + pad) + pad, top + rows * (ch + pad) + pad),
                      (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((pad + 4, 18), args.title, fill=(20, 20, 20), font=font(28))

    f_id, f_p = font(17), font(14)
    for i, f in enumerate(files):
        r, c = divmod(i, cols)
        x = pad + c * (cw + pad)
        y = top + r * (ch + pad)
        im = Image.open(f).convert("RGB").resize((cw, int(cw * 0.64)), Image.LANCZOS)
        sheet.paste(im, (x, y))

        cfg = summary.get(f.stem, {})
        d.rectangle([x, y + im.height, x + cw, y + ch], fill=(246, 246, 248))
        d.text((x + 6, y + im.height + 4),
               f"{f.stem}   {cfg.get('family','')}", fill=(15, 15, 15), font=f_id)
        pier = cfg.get("pier_type") or "no piers"
        d.text((x + 6, y + im.height + 24),
               f"L {cfg.get('total_length_m','?')} m   W {cfg.get('width_m','?')} m   "
               f"{cfg.get('num_spans','?')} span   {pier}",
               fill=(70, 70, 75), font=f_p)
        d.rectangle([x, y, x + cw, y + ch], outline=(214, 214, 218), width=1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, optimize=True)
    print(f"wrote {args.out}  ({sheet.width}x{sheet.height}, {len(files)} bridges)")


if __name__ == "__main__":
    main()
