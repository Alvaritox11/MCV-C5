import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pycocotools.mask as mask_utils
from PIL import Image


DEFAULT_TRAIN_SEQS = ["0000", "0001", "0003", "0004", "0005", "0009", "0011", "0012", "0015", "0017", "0019", "0020"]
DEFAULT_VAL_SEQS   = ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]

# KITTI-MOTS class IDs in instances_txt:
# 1 = Car, 2 = Pedestrian
KITTI_TO_YOLO = {
    2: 0,  # Pedestrian -> person (0)
    1: 1,  # Car -> car (1)
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def xyxy_to_yolo_norm(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[float, float, float, float]:
    # Clamp to bounds
    x1 = clamp(x1, 0.0, w - 1.0)
    y1 = clamp(y1, 0.0, h - 1.0)
    x2 = clamp(x2, 0.0, w * 1.0)
    y2 = clamp(y2, 0.0, h * 1.0)

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 1.0 or bh <= 1.0:
        return None  # too small / invalid

    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    # Normalize
    return (xc / w, yc / h, bw / w, bh / h)


def parse_instances_txt(txt_path: Path) -> Dict[int, List[Tuple[int, dict]]]:
    """
    Returns: frame_id -> list of (yolo_cls, rle_obj)
    instances_txt line:
      frame_id track_id class_id img_height img_width rle...
    """
    annos: Dict[int, List[Tuple[int, dict]]] = {}
    if not txt_path.exists():
        return annos

    with txt_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue

            frame_id = int(parts[0])
            class_id = int(parts[2])
            height = int(parts[3])
            width = int(parts[4])

            if class_id not in KITTI_TO_YOLO:
                continue

            rle_str = " ".join(parts[5:])
            rle_obj = {"counts": rle_str.encode("utf-8"), "size": [height, width]}
            yolo_cls = KITTI_TO_YOLO[class_id]

            annos.setdefault(frame_id, []).append((yolo_cls, rle_obj))

    return annos


def symlink_or_copy(src: Path, dst: Path, use_symlinks: bool) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    if use_symlinks:
        # Use relative symlink when possible (nicer for portability)
        try:
            rel = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel)
        except Exception:
            dst.symlink_to(src)
    else:
        # Copy file contents
        import shutil
        shutil.copy2(src, dst)


def export_split(
    kitti_root: Path,
    out_root: Path,
    split: str,
    seqs: List[str],
    use_symlinks: bool,
) -> None:
    img_root = kitti_root / "training" / "image_02"
    inst_txt_root = kitti_root / "instances_txt"

    out_img_dir = out_root / "images" / split
    out_lbl_dir = out_root / "labels" / split
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    for seq in seqs:
        seq_img_dir = img_root / seq
        seq_txt = inst_txt_root / f"{seq}.txt"
        if not seq_img_dir.exists():
            print(f"[WARN] Missing images dir: {seq_img_dir}")
            continue
        if not seq_txt.exists():
            print(f"[WARN] Missing txt: {seq_txt}")
            continue

        annos_by_frame = parse_instances_txt(seq_txt)

        frames = sorted([p for p in seq_img_dir.iterdir() if p.suffix.lower() == ".png"])
        for frame_path in frames:
            frame_id = int(frame_path.stem)

            # Output file names: <seq>_<frame>.png / .txt
            out_stem = f"{seq}_{frame_path.stem}"
            out_img = out_img_dir / f"{out_stem}.png"
            out_lbl = out_lbl_dir / f"{out_stem}.txt"

            # Link/copy image
            symlink_or_copy(frame_path, out_img, use_symlinks=use_symlinks)

            # Get image size (prefer reading once from PIL to avoid trusting txt size)
            with Image.open(frame_path) as im:
                w, h = im.size

            objs = annos_by_frame.get(frame_id, [])
            lines = []
            for yolo_cls, rle_obj in objs:
                # bbox from mask rle (xywh)
                x, y, bw, bh = mask_utils.toBbox(rle_obj).tolist()
                x1, y1 = float(x), float(y)
                x2, y2 = float(x + bw), float(y + bh)

                yolo = xyxy_to_yolo_norm(x1, y1, x2, y2, w=w, h=h)
                if yolo is None:
                    continue
                xc, yc, nw, nh = yolo
                lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            # Write label file (empty file is OK for images with no objects)
            out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_data_yaml(out_root: Path) -> None:
    yaml_path = out_root / "data.yaml"
    content = f"""# KITTI-MOTS exported to YOLO detection format
path: {out_root}
train: images/train
val: images/val
names:
  0: person
  1: car
"""
    yaml_path.write_text(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", type=str, required=True, help="Path to KITTI-MOTS root (contains training/, instances_txt/)")
    ap.add_argument("--out_root", type=str, required=True, help="Output YOLO dataset root to create")
    ap.add_argument("--no_symlinks", action="store_true", help="Copy images instead of symlinking")
    ap.add_argument("--train_seqs", type=str, default=",".join(DEFAULT_TRAIN_SEQS))
    ap.add_argument("--val_seqs", type=str, default=",".join(DEFAULT_VAL_SEQS))
    args = ap.parse_args()

    kitti_root = Path(args.kitti_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    use_symlinks = not args.no_symlinks

    train_seqs = [s.strip() for s in args.train_seqs.split(",") if s.strip()]
    val_seqs = [s.strip() for s in args.val_seqs.split(",") if s.strip()]

    ensure_dir(out_root)
    export_split(kitti_root, out_root, "train", train_seqs, use_symlinks)
    export_split(kitti_root, out_root, "val", val_seqs, use_symlinks)
    write_data_yaml(out_root)

    print("\nDone!")
    print(f"YOLO dataset: {out_root}")
    print(f"data.yaml: {out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()