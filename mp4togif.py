"""
Batch convert all MP4 files in a folder to GIFs.

Examples
--------
python mp4togif.py --input ./videos --output ./gifs --fps 12 --resize 0.5
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable

from moviepy.editor import VideoFileClip


def find_mp4s(root: Path, recursive: bool) -> Iterable[Path]:
    """Yield mp4 files under root."""
    pattern = "**/*.mp4" if recursive else "*.mp4"
    return sorted(root.glob(pattern))


def convert_mp4_to_gif(
    src: Path,
    dest: Path,
    fps: int | None,
    resize: float | None,
    overwrite: bool,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        print(f"skip (exists): {dest}")
        return

    with VideoFileClip(str(src)) as clip:
        if resize is not None:
            clip = clip.resize(resize)
        clip.write_gif(str(dest), fps=fps)
    print(f"created: {dest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all MP4 files in a folder to GIFs."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Folder containing MP4 files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination folder for GIF files.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second for the GIF (defaults to source).",
    )
    parser.add_argument(
        "--resize",
        type=float,
        default=None,
        help="Scale factor for resizing (e.g., 0.5 halves width/height).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subfolders for MP4 files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite GIFs if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input
    output_dir: Path = args.output

    if not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    mp4_files = list(find_mp4s(input_dir, args.recursive))
    if not mp4_files:
        print("No MP4 files found.")
        return

    for mp4_path in mp4_files:
        rel = mp4_path.relative_to(input_dir)
        gif_dest = output_dir / rel.with_suffix(".gif")
        try:
            convert_mp4_to_gif(
                src=mp4_path,
                dest=gif_dest,
                fps=args.fps,
                resize=args.resize,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001 - keep simple for CLI
            print(f"failed: {mp4_path} -> {gif_dest} ({exc})")


if __name__ == "__main__":
    main()
