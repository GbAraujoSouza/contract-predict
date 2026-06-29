import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Walk-Forward", page_icon="🚀", layout="wide")

st.title("🚀 Walk-Forward com Tuning por ROC-AUC")
st.caption("Resultados reais do notebook `07_treinamento_serie_temporal.ipynb`.")

validation_df = pd.DataFrame([
    {
        "Modelo": "Random Forest",
        "ROC-AUC TSCV": 0.914666,
        "ROC-AUC Walk-forward": 0.922445,
        "Std TSCV": 0.009551,
        "Std Walk-forward": 0.000628,
    },
    {
        "Modelo": "LightGBM",
        "ROC-AUC TSCV": 0.912963,
        "ROC-AUC Walk-forward": 0.921792,
        "Std TSCV": 0.009887,
        "Std Walk-forward": 0.000799,
    },
    {
        "Modelo": "Decision Tree",
        "ROC-AUC TSCV": 0.901458,
        "ROC-AUC Walk-forward": 0.914585,
        "Std TSCV": 0.014369,
        "Std Walk-forward": 0.003041,
    },
    {
        "Modelo": "Logistic Regression",
        "ROC-AUC TSCV": 0.893422,
        "ROC-AUC Walk-forward": 0.905235,
        "Std TSCV": 0.012094,
        "Std Walk-forward": 0.007178,
    },
]).sort_values("ROC-AUC Walk-forward", ascending=False)

holdout_df = pd.DataFrame([
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

timing_df = pd.DataFrame([
    {"Protocolo": "TimeSeriesSplit", "Tempo (s)": 1246.241991, "Combinações por modelo": 4, "Fits totais": 80},
    {"Protocolo": "Walk-forward", "Tempo (s)": 619.428919, "Combinações por modelo": 4, "Fits totais": 32},
])

walk_folds_df = pd.DataFrame([
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

tscv_summary_df = pd.DataFrame([
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

feature_df = pd.DataFrame({
    "Feature base": [
        "close_mid",
        "depth_imbalance",
        "mean_spread",
        "close_spread",
        "bar_volatility",
        "ofi_corrected",
    ],
    "Lag t-1": [
        "close_mid_lag1",
        "depth_imbalance_lag1",
        "mean_spread_lag1",
        "close_spread_lag1",
        "bar_volatility_lag1",
        "ofi_corrected_lag1",
    ],
})

grid_df = pd.DataFrame([
    {"Modelo": "LightGBM", "Combinações": 4, "Grade": "n_estimators in {100, 200}; max_depth in {4, 6}; learning_rate = 0.05"},
    {"Modelo": "Decision Tree", "Combinações": 4, "Grade": "max_depth in {4, 6, 10}; min_samples_leaf in {1, 5, 10}"},
    {"Modelo": "Logistic Regression", "Combinações": 4, "Grade": "C in {0.01, 0.1, 1.0, 10.0}"},
    {"Modelo": "Random Forest", "Combinações": 4, "Grade": "n_estimators in {50, 100}; max_depth in {6, 10}; min_samples_leaf in {1, 5}"},
])

st.markdown(
    """
Objetivo: prever se o preço do contrato sobe nos próximos `15 min` (`target = 1`).
Todo tuning acontece só no período pré-teste; o holdout final fica intocado até fim do fluxo.
"""
)

st.divider()

st.subheader("⚙️ Configuração")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Horizonte do target", "15 min")
c2.metric("Purge gap", "15 min")
c3.metric("Walk-forward", "2 dias treino / 1 dia validação")
c4.metric("TimeSeriesSplit", "5 folds")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Linhas totais", "5,582,837")
c6.metric("Pré-teste", "4,936,768")
c7.metric("Holdout final", "638,504")
c8.metric("Baseline accuracy", "0.7322")

with st.expander("📋 Features usadas (12)"):
    st.dataframe(feature_df, use_container_width=True, hide_index=True)
    st.caption(
        "`ofi_corrected = (buy_volume - sell_volume) / total_volume` quando `total_volume > 0`; "
        "caso contrário, `0`."
    )

st.divider()

st.subheader("📅 Divisão temporal")
period_df = pd.DataFrame([
    {"Bloco": "Dataset completo", "Início": "2026-03-06 00:01 UTC", "Fim": "2026-03-11 23:59 UTC"},
    {"Bloco": "Pré-teste", "Início": "2026-03-06 00:01 UTC", "Fim": "2026-03-09 23:44 UTC"},
    {"Bloco": "Holdout final", "Início": "2026-03-10 00:00 UTC", "Fim": "2026-03-11 23:59 UTC"},
])
st.dataframe(period_df, use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.markdown(
        "- `Purge gap = 15 min` antes do holdout final.\n"
        "- `Walk-forward`: 2 folds com janela fixa de treino.\n"
        "- `TimeSeriesSplit`: 5 folds expansivos com `gap = 15` observações."
    )
with right:
    st.dataframe(timing_df.style.format({"Tempo (s)": "{:.1f}"}), use_container_width=True, hide_index=True)

st.divider()

st.subheader("🧪 Folds temporais")
tab1, tab2 = st.tabs(["Walk-forward", "TimeSeriesSplit"])
with tab1:
    st.dataframe(walk_folds_df, use_container_width=True, hide_index=True)
with tab2:
    st.dataframe(tscv_summary_df, use_container_width=True, hide_index=True)

st.divider()

st.subheader("🎛️ Grade de hiperparâmetros")
st.dataframe(grid_df, use_container_width=True, hide_index=True)

st.divider()

st.subheader("🏆 Melhor ROC-AUC na validação temporal")
st.dataframe(
    validation_df.style.format({
        "ROC-AUC TSCV": "{:.4f}",
        "ROC-AUC Walk-forward": "{:.4f}",
        "Std TSCV": "{:.4f}",
        "Std Walk-forward": "{:.4f}",
    }),
    use_container_width=True,
    hide_index=True,
)

cmp_long = validation_df.melt(
    id_vars="Modelo",
    value_vars=["ROC-AUC TSCV", "ROC-AUC Walk-forward"],
    var_name="Protocolo",
    value_name="ROC-AUC",
)
fig = px.bar(
    cmp_long,
    x="Modelo",
    y="ROC-AUC",
    color="Protocolo",
    barmode="group",
    text="ROC-AUC",
    range_y=[0.88, 0.93],
    title="Validação: TimeSeriesSplit vs Walk-forward",
)
fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
st.plotly_chart(fig, use_container_width=True)

st.info(
    "Walk-forward venceu em todos os modelos e ainda rodou mais rápido: "
    "619.4 s contra 1246.2 s."
)

st.divider()

st.subheader("🎯 Holdout final cego")
st.caption("Modelos re-treinados no pré-teste inteiro com melhores hiperparâmetros do walk-forward.")
st.dataframe(
    holdout_df.style.format({col: "{:.4f}" for col in holdout_df.columns if col != "Modelo"}),
    use_container_width=True,
    hide_index=True,
)

holdout_long = holdout_df.melt(
    id_vars="Modelo",
    value_vars=["ROC-AUC", "PR-AUC", "Accuracy", "F1-Score"],
    var_name="Métrica",
    value_name="Valor",
)
fig2 = px.bar(
    holdout_long,
    x="Modelo",
    y="Valor",
    color="Métrica",
    barmode="group",
    text="Valor",
    range_y=[0.65, 0.95],
    title="Métricas no holdout final",
)
fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    "- `Random Forest` fecha melhor `ROC-AUC` (`0.9099`) e melhor `PR-AUC` (`0.8189`).\n"
    "- `Logistic Regression` fecha melhor `Accuracy` (`0.7704`).\n"
    "- `Decision Tree` fecha melhor `F1-Score` (`0.6720`)."
)

st.divider()

st.subheader("📚 Interpretação")
st.markdown(
    """
- `y_proba`: score de probabilidade da classe `1`.
- `y_pred`: classe final depois do threshold (`0.5` no notebook).
- `ROC-AUC`: mede capacidade de ranking usando `y_proba`.
- `Accuracy`, `Precision`, `Recall` e `F1-Score`: dependem de threshold e usam `y_pred`.
- `baseline_acc = 0.7322` vale só para `accuracy`; baseline natural da ROC segue `0.5`.
"""
)

st.divider()

st.subheader("🧾 Síntese")
st.success(
    """
- Dados ordenados no tempo e avaliados com `gap`.
- `Walk-forward` entrou como protocolo separado, com treino fixo de `2 dias` e validação de `1 dia`.
- Tuning feito por `ROC-AUC` apenas no pré-teste.
- Holdout final preservado até última etapa.
- Melhor resultado final em ranking: `Random Forest`.
"""
)
