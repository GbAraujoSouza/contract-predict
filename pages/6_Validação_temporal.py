import plotly.express as px
import streamlit as st

from temporal_dashboard_data import (
    FEATURES,
    PERIOD_DF,
    TIMING_DF,
    TSCV_BEST_DF,
    TSCV_FOLDS_DF,
    VALIDATION_DF,
    WALK_BEST_DF,
    WALK_FOLDS_DF,
)

st.set_page_config(page_title="Validação temporal", page_icon="⏱️", layout="wide")

st.title("⏱️ Validação temporal")
st.caption("Comparação entre `TimeSeriesSplit` e `Walk-forward` no notebook `07_treinamento_serie_temporal.ipynb`.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Horizonte do target", "15 min")
c2.metric("Purge gap", "15 min")
c3.metric("TimeSeriesSplit", "5 folds")
c4.metric("Walk-forward", "2 dias treino / 1 dia validação")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Features do modelo", str(len(FEATURES)))
c6.metric("Pré-teste", "4.936.768")
c7.metric("Holdout final", "638.504")
c8.metric("Baseline accuracy", "0.7322")

st.divider()

st.subheader("📅 Divisão temporal")
st.dataframe(PERIOD_DF, use_container_width=True, hide_index=True)
st.info("Todo tuning acontece no pré-teste. Holdout final fica intocado até a etapa final.")

st.divider()

st.subheader("🧪 Folds por protocolo")
tab1, tab2 = st.tabs(["Walk-forward", "TimeSeriesSplit"])
with tab1:
    st.dataframe(WALK_FOLDS_DF, use_container_width=True, hide_index=True)
with tab2:
    st.dataframe(TSCV_FOLDS_DF, use_container_width=True, hide_index=True)

st.divider()

st.subheader("⏱️ Custo computacional")
st.dataframe(
    TIMING_DF.style.format({"Tempo (s)": "{:.1f}"}),
    use_container_width=True,
    hide_index=True,
)

st.divider()

st.subheader("🏆 Melhores combinações por protocolo")
best_tab1, best_tab2 = st.tabs(["TimeSeriesSplit", "Walk-forward"])
with best_tab1:
    st.dataframe(
        TSCV_BEST_DF.style.format({"Melhor ROC-AUC": "{:.4f}", "Std": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
with best_tab2:
    st.dataframe(
        WALK_BEST_DF.style.format({"Melhor ROC-AUC": "{:.4f}", "Std": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

st.subheader("⚖️ Comparação direta entre protocolos")
st.dataframe(
    VALIDATION_DF[
        [
            "Modelo",
            "ROC-AUC TSCV",
            "ROC-AUC Walk-forward",
            "Ganho Walk-forward",
            "Melhor params TSCV",
            "Melhor params Walk-forward",
        ]
    ].style.format({
        "ROC-AUC TSCV": "{:.4f}",
        "ROC-AUC Walk-forward": "{:.4f}",
        "Ganho Walk-forward": "{:+.4f}",
    }),
    use_container_width=True,
    hide_index=True,
)

cmp_long = VALIDATION_DF.melt(
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
    title="ROC-AUC médio na validação temporal",
)
fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
st.plotly_chart(fig, use_container_width=True)

fig_gain = px.bar(
    VALIDATION_DF,
    x="Modelo",
    y="Ganho Walk-forward",
    color="Ganho Walk-forward",
    text="Ganho Walk-forward",
    color_continuous_scale="Blues",
    title="Ganho do walk-forward sobre o TimeSeriesSplit",
)
fig_gain.update_traces(texttemplate="%{text:+.4f}", textposition="outside")
st.plotly_chart(fig_gain, use_container_width=True)

st.success(
    """
- `Walk-forward` vence em todos os modelos.
- Também custa menos computação: `619.4 s` e `32 fits` vs `1246.2 s` e `80 fits`.
- Melhor ROC-AUC de validação: `Random Forest` com `0.9224`.
"""
)
