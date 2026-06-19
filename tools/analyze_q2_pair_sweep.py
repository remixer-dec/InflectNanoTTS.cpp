#!/usr/bin/env python3
"""Analyze pairwise acoustic Q2 sweep results and emit reusable layer lists."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_layers(value):
    return [item for item in value.split("|") if item]


def load_rows(path):
    path = Path(path)
    rows = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["layers"] = parse_layers(row["layers"])
            for key in ("cosine", "mean_abs", "max_abs", "convert_seconds", "synth_seconds", "compare_seconds", "total_seconds"):
                if row.get(key) not in (None, ""):
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def is_valid(row):
    return not row.get("error") and "cosine" in row and "mean_abs" in row and "max_abs" in row


def passes(row, min_cosine, max_mean_abs, max_max_abs):
    if not is_valid(row):
        return False
    if row["cosine"] < min_cosine:
        return False
    if row["mean_abs"] > max_mean_abs:
        return False
    if max_max_abs is not None and row["max_abs"] > max_max_abs:
        return False
    return True


def layer_stats(rows, min_cosine, max_mean_abs, max_max_abs):
    stats = defaultdict(lambda: {
        "pairs": 0,
        "passed": 0,
        "failed": 0,
        "worst_cosine": 1.0,
        "worst_mean_abs": 0.0,
        "worst_max_abs": 0.0,
        "worst_pair": "",
    })

    for row in rows:
        if not is_valid(row):
            continue
        ok = passes(row, min_cosine, max_mean_abs, max_max_abs)
        pair_name = "|".join(row["layers"])
        for layer in row["layers"]:
            s = stats[layer]
            s["pairs"] += 1
            if ok:
                s["passed"] += 1
            else:
                s["failed"] += 1

            worse = False
            if row["cosine"] < s["worst_cosine"]:
                worse = True
            if row["mean_abs"] > s["worst_mean_abs"] and row["cosine"] <= s["worst_cosine"] + 0.01:
                worse = True
            if worse:
                s["worst_cosine"] = row["cosine"]
                s["worst_mean_abs"] = row["mean_abs"]
                s["worst_max_abs"] = row["max_abs"]
                s["worst_pair"] = pair_name
            else:
                s["worst_cosine"] = min(s["worst_cosine"], row["cosine"])
                s["worst_mean_abs"] = max(s["worst_mean_abs"], row["mean_abs"])
                s["worst_max_abs"] = max(s["worst_max_abs"], row["max_abs"])

    for layer, s in stats.items():
        s["pass_rate"] = s["passed"] / s["pairs"] if s["pairs"] else 0.0
        # Higher means more sensitive. This intentionally emphasizes complete
        # failures while still ranking smaller degradations.
        s["risk_score"] = (
            (1.0 - s["worst_cosine"]) * 100.0
            + s["worst_mean_abs"] * 10.0
            + s["worst_max_abs"]
            + s["failed"] * 0.01
        )

    return dict(stats)


def max_compatible_layers(rows, min_cosine, max_mean_abs, max_max_abs, eligible_layers):
    """Find the largest layer set where every observed internal pair passes."""
    layers = sorted(eligible_layers)
    index = {layer: i for i, layer in enumerate(layers)}
    n = len(layers)
    pass_adj = [0] * n
    pass_pairs = set()
    fail_pairs = set()

    for row in rows:
        if not is_valid(row) or len(row["layers"]) != 2:
            continue
        a, b = row["layers"]
        if a not in index or b not in index:
            continue
        i, j = index[a], index[b]
        if i == j:
            continue
        key = tuple(sorted((i, j)))
        if passes(row, min_cosine, max_mean_abs, max_max_abs):
            pass_pairs.add(key)
        else:
            fail_pairs.add(key)

    # A failed observed pair wins over any duplicate passing row.
    for i, j in pass_pairs - fail_pairs:
        pass_adj[i] |= 1 << j
        pass_adj[j] |= 1 << i

    best = 0

    def color_sort(candidates):
        order = []
        colors = []
        remaining = candidates
        color = 0
        while remaining:
            color += 1
            available = remaining
            while available:
                bit = available & -available
                v = bit.bit_length() - 1
                order.append(v)
                colors.append(color)
                remaining &= ~bit
                available &= ~bit
                available &= ~pass_adj[v]
        return order, colors

    def expand(clique, candidates):
        nonlocal best
        if not candidates:
            if clique.bit_count() > best.bit_count():
                best = clique
            return

        order, colors = color_sort(candidates)
        for pos in range(len(order) - 1, -1, -1):
            if clique.bit_count() + colors[pos] <= best.bit_count():
                return
            v = order[pos]
            bit = 1 << v
            if not (candidates & bit):
                continue
            expand(clique | bit, candidates & pass_adj[v])
            candidates &= ~bit

    expand(0, (1 << n) - 1 if n else 0)

    compatible = [layers[i] for i in range(n) if best & (1 << i)]
    bad_pairs = [(layers[i], layers[j]) for i, j in sorted(fail_pairs)]
    return compatible, bad_pairs


def write_lines(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def write_override_file(path, layers, quant):
    lines = [f"--acoustic-quantize-override '{layer}={quant}'" for layer in layers]
    write_lines(path, lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze Q2 pair-sweep results into robust/sensitive layer lists.")
    parser.add_argument("--results", required=True, type=Path, help="Path to results.jsonl or results.csv.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--max-mean-abs", type=float, default=0.005)
    parser.add_argument("--max-max-abs", type=float, default=None)
    parser.add_argument("--min-pairs", type=int, default=1, help="Minimum observed pairs required for a layer to be classified.")
    parser.add_argument("--good-pass-rate", type=float, default=1.0, help="Layer is Q2-safe if at least this fraction of its pairs pass.")
    parser.add_argument("--special-quant", default="f16", help="Quant override to emit for sensitive layers, e.g. f16/q8_0/q4_k.")
    parser.add_argument("--good-quant", default="q2_k", help="Quant override to emit for robust layers.")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args(argv)

    rows = load_rows(args.results)
    valid_rows = [row for row in rows if is_valid(row)]
    if not valid_rows:
        raise SystemExit(f"no valid rows found in {args.results}")

    stats = layer_stats(valid_rows, args.min_cosine, args.max_mean_abs, args.max_max_abs)
    layers = sorted(stats)

    good_layers = [
        layer for layer in layers
        if stats[layer]["pairs"] >= args.min_pairs and stats[layer]["pass_rate"] >= args.good_pass_rate
    ]
    sensitive_layers = [
        layer for layer in layers
        if stats[layer]["pairs"] >= args.min_pairs and stats[layer]["pass_rate"] < args.good_pass_rate
    ]
    sensitive_layers.sort(key=lambda layer: stats[layer]["risk_score"], reverse=True)

    out_dir = args.out_dir or args.results.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    eligible_layers = [layer for layer in layers if stats[layer]["pairs"] >= args.min_pairs]
    compatible_layers, bad_pairs = max_compatible_layers(
        valid_rows,
        args.min_cosine,
        args.max_mean_abs,
        args.max_max_abs,
        eligible_layers,
    )
    compatible_set = set(compatible_layers)
    incompatible_layers = [layer for layer in eligible_layers if layer not in compatible_set]

    summary_path = out_dir / "layer_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "layer",
            "pairs",
            "passed",
            "failed",
            "pass_rate",
            "worst_cosine",
            "worst_mean_abs",
            "worst_max_abs",
            "risk_score",
            "worst_pair",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for layer in sorted(layers, key=lambda x: stats[x]["risk_score"], reverse=True):
            row = {"layer": layer, **stats[layer]}
            writer.writerow(row)

    write_lines(out_dir / "q2_good_layers.txt", good_layers)
    write_lines(out_dir / "q2_compatible_layers.txt", compatible_layers)
    write_lines(out_dir / "q2_incompatible_layers.txt", incompatible_layers)
    write_lines(out_dir / "sensitive_layers.txt", sensitive_layers)
    write_override_file(out_dir / "q2_good_overrides.txt", good_layers, args.good_quant)
    write_override_file(out_dir / "q2_compatible_overrides.txt", compatible_layers, args.good_quant)
    write_override_file(out_dir / "q2_incompatible_overrides.txt", incompatible_layers, args.special_quant)
    mixed_lines = [
        *(f"--acoustic-quantize-override '{layer}={args.special_quant}'" for layer in incompatible_layers),
        *(f"--acoustic-quantize-override '{layer}={args.good_quant}'" for layer in compatible_layers),
    ]
    write_lines(out_dir / "q2_compatible_mixed_overrides.txt", mixed_lines)
    write_override_file(out_dir / "sensitive_overrides.txt", sensitive_layers, args.special_quant)
    write_lines(out_dir / "bad_pairs.txt", ["|".join(pair) for pair in bad_pairs])

    print(f"Rows: {len(rows)} total, {len(valid_rows)} valid")
    print(f"Thresholds: cosine>={args.min_cosine}, mean_abs<={args.max_mean_abs}, max_abs<={args.max_max_abs}")
    print(f"Q2-good layers: {len(good_layers)}")
    print(f"Q2-compatible layer set: {len(compatible_layers)}")
    print(f"Q2-incompatible layer set: {len(incompatible_layers)}")
    print(f"Bad pair edges: {len(bad_pairs)}")
    print(f"Sensitive/special layers: {len(sensitive_layers)}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {out_dir / 'q2_good_layers.txt'}")
    print(f"Wrote {out_dir / 'q2_compatible_layers.txt'}")
    print(f"Wrote {out_dir / 'q2_incompatible_layers.txt'}")
    print(f"Wrote {out_dir / 'sensitive_layers.txt'}")
    print(f"Wrote {out_dir / 'q2_good_overrides.txt'}")
    print(f"Wrote {out_dir / 'q2_compatible_overrides.txt'}")
    print(f"Wrote {out_dir / 'q2_incompatible_overrides.txt'}")
    print(f"Wrote {out_dir / 'q2_compatible_mixed_overrides.txt'}")
    print(f"Wrote {out_dir / 'sensitive_overrides.txt'}")
    print(f"Wrote {out_dir / 'bad_pairs.txt'}")

    print("\nMost sensitive:")
    for layer in sensitive_layers[: args.top]:
        s = stats[layer]
        print(
            f"{layer}: pass_rate={s['pass_rate']:.3f} "
            f"worst_cosine={s['worst_cosine']:.6f} "
            f"worst_mean_abs={s['worst_mean_abs']:.6f} "
            f"worst_pair={s['worst_pair']}"
        )

    print("\nMost robust:")
    robust = sorted(good_layers, key=lambda layer: (stats[layer]["worst_cosine"], -stats[layer]["worst_mean_abs"]), reverse=True)
    for layer in robust[: args.top]:
        s = stats[layer]
        print(
            f"{layer}: pass_rate={s['pass_rate']:.3f} "
            f"worst_cosine={s['worst_cosine']:.6f} "
            f"worst_mean_abs={s['worst_mean_abs']:.6f}"
        )

    print("\nLargest mutually-compatible Q2 set:")
    for layer in compatible_layers[: args.top]:
        s = stats[layer]
        print(
            f"{layer}: pass_rate={s['pass_rate']:.3f} "
            f"worst_cosine={s['worst_cosine']:.6f} "
            f"worst_mean_abs={s['worst_mean_abs']:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
