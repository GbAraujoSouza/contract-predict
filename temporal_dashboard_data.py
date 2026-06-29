from __future__ import annotations

import base64
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

NOTEBOOK_PATH = Path(__file__).resolve().parent / "07_treinamento_serie_temporal.ipynb"

FEATURES = [
    "close_mid",
    "depth_imbalance",
    "mean_spread",
    "close_spread",
    "bar_volatility",
    "ofi_corrected",
    "close_mid_lag1",
    "depth_imbalance_lag1",
    "mean_spread_lag1",
    "close_spread_lag1",
    "bar_volatility_lag1",
    "ofi_corrected_lag1",
]

PERIOD_DF = pd.DataFrame([
    {"Bloco": "Dataset completo", "Início": "2026-03-06 00:01 UTC", "Fim": "2026-03-11 23:59 UTC", "Linhas": "5.582.837"},
    {"Bloco": "Pré-teste", "Início": "2026-03-06 00:01 UTC", "Fim": "2026-03-09 23:44 UTC", "Linhas": "4.936.768"},
    {"Bloco": "Holdout final", "Início": "2026-03-10 00:00 UTC", "Fim": "2026-03-11 23:59 UTC", "Linhas": "638.504"},
])

TIMING_DF = pd.DataFrame([
    {"Protocolo": "TimeSeriesSplit", "Tempo (s)": 1246.241991, "Combinações por modelo": 4, "Fits totais": 80},
    {"Protocolo": "Walk-forward", "Tempo (s)": 619.428919, "Combinações por modelo": 4, "Fits totais": 32},
])

TSCV_BEST_DF = pd.DataFrame([
    {
        "Modelo": "Random Forest",
        "Folds": 5,
        "Combinações": 4,
        "Melhor ROC-AUC": 0.914666,
        "Std": 0.009551,
        "Melhores hiperparâmetros": "{'max_depth': 10, 'min_samples_leaf': 5, 'n_estimators': 100}",
    },
    {
        "Modelo": "LightGBM",
        "Folds": 5,
        "Combinações": 4,
        "Melhor ROC-AUC": 0.912963,
        "Std": 0.009887,
        "Melhores hiperparâmetros": "{'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 200}",
    },
    {
        "Modelo": "Decision Tree",
        "Folds": 5,
        "Combinações": 4,
        "Melhor ROC-AUC": 0.901458,
        "Std": 0.014369,
        "Melhores hiperparâmetros": "{'max_depth': 10, 'min_samples_leaf': 10}",
    },
    {
        "Modelo": "Logistic Regression",
        "Folds": 5,
        "Combinações": 4,
        "Melhor ROC-AUC": 0.893422,
        "Std": 0.012094,
        "Melhores hiperparâmetros": "{'C': 0.1}",
    },
]).sort_values("Melhor ROC-AUC", ascending=False)

WALK_BEST_DF = pd.DataFrame([
    {
        "Modelo": "Random Forest",
        "Folds": 2,
        "Combinações": 4,
        "Melhor ROC-AUC": 0.922445,
        "Std": 0.000628,
        "Melhores hiperparâmetros": "{'max_depth': 10, 'min_samples_leaf': 5, 'n_estimators': 100}",
    },
    {
        "Modelo": "LightGBM",
        "Folds": 2,
        "Combinações": 4,
        "Melhor ROC-AUC": 0.921792,
        "Std": 0.000799,
        "Melhores hiperparâmetros": "{'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 200}",
    },
    {
        "Modelo": "Decision Tree",
        "Folds": 2,
        "Combinações": 4,
        "Melhor ROC-AUC": 0.914585,
        "Std": 0.003041,
        "Melhores hiperparâmetros": "{'max_depth': 10, 'min_samples_leaf': 10}",
    },
    {
        "Modelo": "Logistic Regression",
        "Folds": 2,
        "Combinações": 4,
        "Melhor ROC-AUC": 0.905235,
        "Std": 0.007178,
        "Melhores hiperparâmetros": "{'C': 0.01}",
    },
]).sort_values("Melhor ROC-AUC", ascending=False)

VALIDATION_DF = pd.DataFrame([
    {
        "Modelo": "Random Forest",
        "ROC-AUC TSCV": 0.914666,
        "ROC-AUC Walk-forward": 0.922445,
        "Melhor params TSCV": "{'max_depth': 10, 'min_samples_leaf': 5, 'n_estimators': 100}",
        "Melhor params Walk-forward": "{'max_depth': 10, 'min_samples_leaf': 5, 'n_estimators': 100}",
    },
    {
        "Modelo": "LightGBM",
        "ROC-AUC TSCV": 0.912963,
        "ROC-AUC Walk-forward": 0.921792,
        "Melhor params TSCV": "{'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 200}",
        "Melhor params Walk-forward": "{'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 200}",
    },
    {
        "Modelo": "Decision Tree",
        "ROC-AUC TSCV": 0.901458,
        "ROC-AUC Walk-forward": 0.914585,
        "Melhor params TSCV": "{'max_depth': 10, 'min_samples_leaf': 10}",
        "Melhor params Walk-forward": "{'max_depth': 10, 'min_samples_leaf': 10}",
    },
    {
        "Modelo": "Logistic Regression",
        "ROC-AUC TSCV": 0.893422,
        "ROC-AUC Walk-forward": 0.905235,
        "Melhor params TSCV": "{'C': 0.1}",
        "Melhor params Walk-forward": "{'C': 0.01}",
    },
]).sort_values("ROC-AUC Walk-forward", ascending=False)
VALIDATION_DF["Ganho Walk-forward"] = (
    VALIDATION_DF["ROC-AUC Walk-forward"] - VALIDATION_DF["ROC-AUC TSCV"]
)

HOLDOUT_WALK_DF = pd.DataFrame([
    {
        "Modelo": "Random Forest",
        "ROC-AUC": 0.909877,
        "PR-AUC": 0.818909,
        "Accuracy": 0.752700,
        "Precision": 0.521392,
        "Recall": 0.930808,
        "F1-Score": 0.668387,
    },
    {
        "Modelo": "LightGBM",
        "ROC-AUC": 0.905172,
        "PR-AUC": 0.810032,
        "Accuracy": 0.755317,
        "Precision": 0.524268,
        "Recall": 0.930598,
        "F1-Score": 0.670691,
    },
    {
        "Modelo": "Decision Tree",
        "ROC-AUC": 0.902120,
        "PR-AUC": 0.797864,
        "Accuracy": 0.759112,
        "Precision": 0.528778,
        "Recall": 0.921719,
        "F1-Score": 0.672025,
    },
    {
        "Modelo": "Logistic Regression",
        "ROC-AUC": 0.893813,
        "PR-AUC": 0.790383,
        "Accuracy": 0.770352,
        "Precision": 0.544421,
        "Recall": 0.872064,
        "F1-Score": 0.670349,
    },
]).sort_values("ROC-AUC", ascending=False)

HOLDOUT_TSCV_DF = pd.DataFrame([
    {
        "Modelo": "Random Forest",
        "ROC-AUC": 0.909877,
        "PR-AUC": 0.818909,
        "Accuracy": 0.752700,
        "Precision": 0.521392,
        "Recall": 0.930808,
        "F1-Score": 0.668387,
    },
    {
        "Modelo": "LightGBM",
        "ROC-AUC": 0.905172,
        "PR-AUC": 0.810032,
        "Accuracy": 0.755317,
        "Precision": 0.524268,
        "Recall": 0.930598,
        "F1-Score": 0.670691,
    },
    {
        "Modelo": "Decision Tree",
        "ROC-AUC": 0.902120,
        "PR-AUC": 0.797864,
        "Accuracy": 0.759112,
        "Precision": 0.528778,
        "Recall": 0.921719,
        "F1-Score": 0.672025,
    },
    {
        "Modelo": "Logistic Regression",
        "ROC-AUC": 0.893812,
        "PR-AUC": 0.790382,
        "Accuracy": 0.770348,
        "Precision": 0.544415,
        "Recall": 0.872064,
        "F1-Score": 0.670345,
    },
]).sort_values("ROC-AUC", ascending=False)

WALK_FOLDS_DF = pd.DataFrame([
    {
        "Fold": "WF-1",
        "Treino": "2026-03-06 00:00 -> 2026-03-07 23:45 UTC",
        "Validação": "2026-03-08 00:00 -> 2026-03-08 23:59 UTC",
    },
    {
        "Fold": "WF-2",
        "Treino": "2026-03-07 00:00 -> 2026-03-08 23:45 UTC",
        "Validação": "2026-03-09 00:00 -> 2026-03-09 23:59 UTC",
    },
])

TSCV_FOLDS_DF = pd.DataFrame([
    {
        "Fold": "TSCV-1",
        "Treino termina em": "2026-03-06 12:25 UTC",
        "Validação começa em": "2026-03-06 12:25 UTC",
        "Validação termina em": "2026-03-07 00:14 UTC",
    },
    {
        "Fold": "TSCV-2",
        "Treino termina em": "2026-03-07 00:14 UTC",
        "Validação começa em": "2026-03-07 00:14 UTC",
        "Validação termina em": "2026-03-07 12:28 UTC",
    },
    {
        "Fold": "TSCV-3",
        "Treino termina em": "2026-03-07 12:27 UTC",
        "Validação começa em": "2026-03-07 12:28 UTC",
        "Validação termina em": "2026-03-08 01:28 UTC",
    },
    {
        "Fold": "TSCV-4",
        "Treino termina em": "2026-03-08 01:28 UTC",
        "Validação começa em": "2026-03-08 01:28 UTC",
        "Validação termina em": "2026-03-08 16:12 UTC",
    },
    {
        "Fold": "TSCV-5",
        "Treino termina em": "2026-03-08 16:12 UTC",
        "Validação começa em": "2026-03-08 16:12 UTC",
        "Validação termina em": "2026-03-09 23:44 UTC",
    },
])

INTERPRETATION_MD = """
- `y_proba`: probabilidade estimada da classe `1`.
- `y_pred`: classe final após threshold `0.5`.
- `ROC-AUC`: mede capacidade de ranking usando `y_proba`.
- `PR-AUC`: resume trade-off entre `precision` e `recall`.
- `Accuracy`, `Precision`, `Recall` e `F1-Score` dependem do threshold.
- `baseline_acc = 0.7322` vale só para `accuracy`; baseline natural da ROC segue `0.5`.
"""

SYNTHESIS_MD = """
- `TimeSeriesSplit` segue útil, mas agora com dados ordenados e `gap`.
- `Walk-forward` entrou como protocolo separado: treino fixo de `2 dias`, validação de `1 dia`.
- Hiperparâmetros foram escolhidos por `ROC-AUC` apenas no pré-teste.
- Holdout final ficou intocado até a última etapa.
- `ROC-AUC` usa `y_proba`; métricas thresholdadas usam `y_pred`.
"""


@lru_cache(maxsize=1)
def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text())


def load_notebook_png(cell_index: int) -> bytes:
    notebook = _load_notebook()
    outputs = notebook["cells"][cell_index]["outputs"]
    for output in outputs:
        png_data = output.get("data", {}).get("image/png")
        if png_data:
            encoded = png_data if isinstance(png_data, str) else "".join(png_data)
            return base64.b64decode(encoded)
    raise ValueError(f"Cell {cell_index} has no PNG output.")
