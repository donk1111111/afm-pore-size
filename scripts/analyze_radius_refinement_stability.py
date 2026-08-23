#!/usr/bin/env python3
"""Analyze experimental radius-refinement stability without changing detections."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def gray_from_rgb(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(float)


def sample_gray(gray: np.ndarray, x: float, y: float) -> float | None:
    if x < 0 or y < 0 or x >= gray.shape[1] - 1 or y >= gray.shape[0] - 1:
        return None
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    dx = x - x0
    dy = y - y0
    top = (1.0 - dx) * gray[y0, x0] + dx * gray[y0, x0 + 1]
    bottom = (1.0 - dx) * gray[y0 + 1, x0] + dx * gray[y0 + 1, x0 + 1]
    return float((1.0 - dy) * top + dy * bottom)


def radial_gradient_candidates(gray: np.ndarray, center_x: float, center_y: float, old_radius: float) -> list[dict[str, float | bool]]:
    candidates: list[dict[str, float | bool]] = []
    for index in range(72):
        theta = 2.0 * math.pi * index / 72.0
        radii = np.linspace(max(1.0, 0.35 * old_radius), max(2.0, 1.80 * old_radius), 80)
        values: list[float] = []
        valid_radii: list[float] = []
        for radius in radii:
            value = sample_gray(gray, center_x + math.cos(theta) * radius, center_y + math.sin(theta) * radius)
            if value is None:
                continue
            valid_radii.append(float(radius))
            values.append(value)
        if len(values) < 8:
            candidates.append({"theta": theta, "radius": float("nan"), "x": float("nan"), "y": float("nan"), "gradient": 0.0, "valid": False})
            continue
        smoothed = np.convolve(np.array(values, dtype=float), np.ones(5) / 5.0, mode="same")
        gradients = np.gradient(smoothed)
        search_start = max(1, int(0.15 * len(gradients)))
        search_end = max(search_start + 1, int(0.95 * len(gradients)))
        local_index = int(np.argmax(gradients[search_start:search_end]) + search_start)
        radius = valid_radii[local_index]
        valid = bool(gradients[local_index] > 0)
        candidates.append(
            {
                "theta": theta,
                "radius": radius,
                "x": center_x + math.cos(theta) * radius,
                "y": center_y + math.sin(theta) * radius,
                "gradient": float(gradients[local_index]),
                "valid": valid,
            }
        )
    return candidates


def category(confidence: float, radius_std: float, args: argparse.Namespace) -> str:
    if confidence < args.low_confidence_threshold or radius_std > args.low_radius_std_threshold:
        return "low"
    if confidence < args.high_confidence_threshold or radius_std > args.high_radius_std_threshold:
        return "medium"
    return "high"


def distribution_rows(values: np.ndarray, metric: str) -> list[dict[str, object]]:
    return [
        {
            "metric": metric,
            "count": int(values.size),
            "min": float(np.min(values)),
            "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)),
            "median": float(np.median(values)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    ]


def draw_debug_image(
    rgb: np.ndarray,
    row: dict[str, str],
    refinement: dict[str, str],
    radial_candidates: list[dict[str, float | bool]],
    output_path: Path,
) -> None:
    center_x = as_float(row, "CenterX_px")
    center_y = as_float(row, "CenterY_px")
    old_radius = as_float(refinement, "old_radius_px")
    new_radius = as_float(refinement, "new_radius_px")
    pad = int(max(45, 2.4 * max(old_radius, new_radius)))
    x0 = max(0, int(round(center_x - pad)))
    y0 = max(0, int(round(center_y - pad)))
    x1 = min(rgb.shape[1], int(round(center_x + pad)))
    y1 = min(rgb.shape[0], int(round(center_y + pad)))
    crop = Image.fromarray(rgb[y0:y1, x0:x1].astype(np.uint8)).convert("RGB")
    scale = 4
    canvas = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(canvas)

    def point(x: float, y: float) -> tuple[float, float]:
        return (x - x0) * scale, (y - y0) * scale

    cx, cy = point(center_x, center_y)
    for candidate in radial_candidates:
        if not candidate["valid"]:
            continue
        px, py = point(float(candidate["x"]), float(candidate["y"]))
        draw.line((cx, cy, px, py), fill=(255, 220, 0), width=1)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), outline=(0, 255, 255), width=1)

    old_r = old_radius * scale
    new_r = new_radius * scale
    draw.ellipse((cx - old_r, cy - old_r, cx + old_r, cy + old_r), outline=(0, 255, 0), width=2)
    draw.ellipse((cx - new_r, cy - new_r, cx + new_r, cy + new_r), outline=(255, 0, 0), width=2)
    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 255, 255))
    text = (
        f"label {refinement['label']} | old green | refined red | samples yellow\n"
        f"std={float(refinement['radius_std']):.3f}, confidence={float(refinement['confidence']):.3f}"
    )
    draw.rectangle((4, 4, min(canvas.width - 4, 560), 42), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    refinement_rows = read_csv(Path(args.radius_refinement))
    pore_rows = {int(float(row["Label"])): row for row in read_csv(Path(args.pores_detail))}

    confidence_values = np.array([as_float(row, "confidence") for row in refinement_rows], dtype=float)
    radius_std_values = np.array([as_float(row, "radius_std") for row in refinement_rows], dtype=float)

    classified_rows: list[dict[str, object]] = []
    counts = {"high": 0, "medium": 0, "low": 0}
    for row in refinement_rows:
        label = int(float(row["label"]))
        confidence = as_float(row, "confidence")
        radius_std = as_float(row, "radius_std")
        level = category(confidence, radius_std, args)
        counts[level] += 1
        if level == "low":
            classified_rows.append(
                {
                    "label": label,
                    "old_radius_px": row["old_radius_px"],
                    "new_radius_px": row["new_radius_px"],
                    "radius_std": row["radius_std"],
                    "confidence": row["confidence"],
                }
            )

    summary_rows = [
        {
            "high_confidence_count": counts["high"],
            "medium_confidence_count": counts["medium"],
            "low_confidence_count": counts["low"],
            "low_rule": f"confidence < {args.low_confidence_threshold} OR radius_std > {args.low_radius_std_threshold}",
            "medium_rule": f"otherwise confidence < {args.high_confidence_threshold} OR radius_std > {args.high_radius_std_threshold}",
            "high_rule": "remaining pores",
        }
    ]
    distribution = distribution_rows(confidence_values, "confidence") + distribution_rows(radius_std_values, "radius_std")

    write_csv(output_dir / "low_confidence_pores.csv", classified_rows)
    write_csv(output_dir / "radius_refinement_confidence_counts.csv", summary_rows)
    write_csv(output_dir / "radius_refinement_distribution.csv", distribution)

    if args.image:
        rgb = np.array(Image.open(args.image).convert("RGB"))
        gray = gray_from_rgb(rgb)
        for low_row in classified_rows:
            label = int(low_row["label"])
            pore_row = pore_rows[label]
            refinement_row = next(row for row in refinement_rows if int(float(row["label"])) == label)
            radial_candidates = radial_gradient_candidates(
                gray,
                as_float(pore_row, "CenterX_px"),
                as_float(pore_row, "CenterY_px"),
                as_float(refinement_row, "old_radius_px"),
            )
            draw_debug_image(
                rgb,
                pore_row,
                refinement_row,
                radial_candidates,
                output_dir / "debug_low_confidence" / f"label_{label:03d}.png",
            )

    print(f"high={counts['high']} medium={counts['medium']} low={counts['low']}")
    print(f"Wrote results to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze radius-refinement stability.")
    parser.add_argument("--radius-refinement", required=True, help="Path to radius_refinement_results.csv")
    parser.add_argument("--pores-detail", required=True, help="Path to pores_detail.csv from the same run")
    parser.add_argument("--image", help="Original image used to create debug images")
    parser.add_argument("--output", default="radius_refinement_stability", help="Output directory")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.88)
    parser.add_argument("--low-radius-std-threshold", type=float, default=2.5)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.93)
    parser.add_argument("--high-radius-std-threshold", type=float, default=1.5)
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
