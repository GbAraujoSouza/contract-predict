"""Gera comparação com/sem split temporal reutilizando TSCV já calculado."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_split_protocols import (  # noqa: E402
    MODEL_NAMES,
    load_xy,
    protocol_chrono_8020,
    protocol_random_shuffle,
    protocol_temporal,
)


def load_cached_temporal() -> dict | None:
    path = Path("reports/model_report_data.json")
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    cv = {}
    holdout = {}
    for name in MODEL_NAMES:
        t = raw["tscv"].get(name, {})
        h = raw["holdout"].get(name, {})
        if t:
            cv[name] = {
                "mean": t["summary"]["auc"]["mean"],
                "std": t["summary"]["auc"]["std"],
            }
        if h:
            holdout[name] = {
                "roc_auc": h["roc_auc"],
                "f1": h["f1"],
                "precision": h["precision"],
                "recall": h["recall"],
                "accuracy": h["accuracy"],
            }
    return {
        "holdout": holdout,
        "cv": cv,
        "meta": {
            "train_rows": raw["data"]["train_rows"],
            "test_rows": raw["data"]["test_rows"],
            "test_period": "2026-03-10 até 2026-03-11",
            "source": "reports/model_report_data.json (cache)",
        },
    }


def main() -> None:
    x, y, bars, _ = load_xy()
    cached = load_cached_temporal()
    if cached:
        print("Reutilizando métricas temporais de reports/model_report_data.json")
        temporal = cached
    else:
        print("Calculando split temporal (sem CV)...")
        temporal = protocol_temporal(x, y, bars, run_cv=False)

    print("Calculando 80/20 cronológico...")
    chrono = protocol_chrono_8020(x, y)
    print("Calculando 80/20 aleatório...")
    random_p = protocol_random_shuffle(x, y)

    report = {
        "com_split_temporal": temporal,
        "sem_split_cronologico_8020": chrono,
        "sem_split_aleatorio": random_p,
    }
    out = Path("reports/split_comparison.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSalvo: {out}")

    for key, label in [
        ("com_split_temporal", "COM split temporal (holdout por data)"),
        ("sem_split_cronologico_8020", "SEM split temporal (80/20 cronológico)"),
        ("sem_split_aleatorio", "SEM split temporal (80/20 aleatório)"),
    ]:
        print(f"\n{label}:")
        for name in MODEL_NAMES:
            h = report[key]["holdout"][name]
            delta = ""
            if key != "com_split_temporal":
                ref = report["com_split_temporal"]["holdout"][name]["roc_auc"]
                delta = f"  (ΔAUC vs temporal: {h['roc_auc'] - ref:+.4f})"
            print(
                f"  {name:16s} AUC={h['roc_auc']:.4f}  F1={h['f1']:.4f}  "
                f"Prec={h['precision']:.4f}  Rec={h['recall']:.4f}{delta}"
            )


if __name__ == "__main__":
    main()
