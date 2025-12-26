import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

@dataclass
class Denom:
    label: str
    diameter_mm: float
    value: float


DENOMS: List[Denom] = [
    Denom("1TL", 26.15, 1.00),
    Denom("50kr", 23.85, 0.50),
    Denom("25kr", 20.50, 0.25),
    Denom("10kr", 18.50, 0.10),
    Denom("5kr", 17.50, 0.05),
]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def match_denom(diam_mm: float, tol: float) -> Tuple[str, float]:
    for d in DENOMS:
        lo = d.diameter_mm * (1 - tol / 100)
        hi = d.diameter_mm * (1 + tol / 100)
        if lo <= diam_mm <= hi:
            return d.label, d.value
    return "UNKNOWN", 0.0


def build_mask_a4(img_bgr: np.ndarray) -> np.ndarray:
    """Assumes coins on a bright (white A4) background."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    _, mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    return mask


def detect_coins_from_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        circles.append([x, y, r])

    if not circles:
        return np.zeros((0, 3), dtype=np.float32)

    circles = np.array(circles, dtype=np.float32)
    circles = circles[np.argsort(circles[:, 0])]  
    return circles


def draw_overlay(img, circles, labels, diam_mm, ref_idx):
    out = img.copy()
    for i, (x, y, r) in enumerate(circles):
        cx, cy, rr = int(x), int(y), int(r)

        color = (0, 165, 255) if i == ref_idx else (0, 255, 0)
        thick = 3 if i == ref_idx else 2

        cv2.circle(out, (cx, cy), rr, color, thick)
        cv2.circle(out, (cx, cy), 2, (0, 0, 255), 3)

        text = f"{i}: {labels[i]} ({diam_mm[i]:.1f}mm)"
        cv2.putText(out, text, (cx - 70, cy - rr - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return out


def main():
    ap = argparse.ArgumentParser("Coin Counter (A4 background)")
    ap.add_argument("--input", required=True, help="Path to input image")
    ap.add_argument("--outdir", default="out", help="Base output directory")
    ap.add_argument("--tol", type=float, default=2.5, help="Denomination tolerance percent")
    ap.add_argument("--min_area", type=int, default=2000, help="Min contour area to keep")
    args = ap.parse_args()

    base = os.path.splitext(os.path.basename(args.input))[0]
    outdir = os.path.join(args.outdir, base)                  
    ensure_dir(outdir)

    img = cv2.imread(args.input)
    if img is None:
        raise SystemExit(f"Image not found: {args.input}")

    mask = build_mask_a4(img)
    cv2.imwrite(os.path.join(outdir, "_mask.png"), mask)

    circles = detect_coins_from_mask(mask, args.min_area)
    if circles.shape[0] == 0:
        raise SystemExit("No coins detected (try lowering --min_area)")

    ref_idx = int(np.argmax(circles[:, 2]))
    one_tl = next(d for d in DENOMS if d.label == "1TL")
    mm_per_px = one_tl.diameter_mm / (2 * circles[ref_idx][2])

    labels, values, diam_mm = [], [], []
    total = 0.0

    for (_, _, r) in circles:
        dmm = 2 * r * mm_per_px
        label, val = match_denom(dmm, args.tol)
        labels.append(label)
        values.append(val)
        diam_mm.append(dmm)
        total += val

    overlay = draw_overlay(img, circles, labels, diam_mm, ref_idx)

    overlay_path = os.path.join(outdir, f"{base}_overlay.png")
    csv_path = os.path.join(outdir, f"{base}_coins.csv")

    cv2.imwrite(overlay_path, overlay)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "cx", "cy", "radius_px", "diameter_mm", "label", "value"])
        for i, (x, y, r) in enumerate(circles):
            w.writerow([i, f"{x:.1f}", f"{y:.1f}", f"{r:.1f}", f"{diam_mm[i]:.2f}", labels[i], values[i]])

    print(f"Detected coins: {len(circles)}")
    print(f"Total value: {total:.2f}")
    print(f"Saved to: {outdir}/")
    print(f"  {os.path.basename(overlay_path)}")
    print(f"  {os.path.basename(csv_path)}")
    print("  _mask.png")


if __name__ == "__main__":
    main()