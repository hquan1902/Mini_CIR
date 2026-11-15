"""Export a CSV and plot from one or more cir.py results.json files."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Paths to results.json")
    parser.add_argument("--output", default="Mini-CIR/comparison")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for result_path in args.results:
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
        for method, values in payload["methods"].items():
            rows.append({"run": Path(result_path).parent.name, "method": method,
                         "gallery_size": payload["config"]["gallery_size"],
                         "queries": payload["config"]["query_count"], **values["metrics"]})
    if not rows:
        raise SystemExit("No method metrics found in input results")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("CSV saved; install matplotlib to also create the comparison plot") from exc
    names = [f"{r['run']}\n{r['method']}" for r in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.15), 5))
    ax.bar([i - .18 for i in x], [100 * r["mAP@5"] for r in rows], width=.36, label="mAP@5")
    ax.bar([i + .18 for i in x], [100 * r["Recall@10"] for r in rows], width=.36, label="Recall@10")
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("CIRCO validation results")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "metrics.png", dpi=180)
    plt.close(fig)
    print(f"Saved {out / 'metrics.csv'} and {out / 'metrics.png'}")


if __name__ == "__main__":
    main()
