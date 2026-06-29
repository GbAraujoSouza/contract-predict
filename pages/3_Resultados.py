import pandas as pd
import plotly.express as px
import streamlit as st

from temporal_dashboard_data import (
    HOLDOUT_TSCV_DF,
    HOLDOUT_WALK_DF,
    INTERPRETATION_MD,
    SYNTHESIS_MD,
    VALIDATION_DF,
    load_notebook_png,
)

st.set_page_config(page_title="Resultados", page_icon="📑", layout="wide")

st.title("📑 Resultados")
st.caption("Resultados e análises do notebook `07_treinamento_serie_temporal.ipynb`.")

summary_df = VALIDATION_DF[["Modelo", "ROC-AUC Walk-forward"]].merge(
    HOLDOUT_WALK_DF, on="Modelo"
).rename(columns={"ROC-AUC": "ROC-AUC Holdout"})
summary_df["Delta Validação->Holdout"] = (
    summary_df["ROC-AUC Holdout"] - summary_df["ROC-AUC Walk-forward"]
)
summary_df = summary_df.sort_values("ROC-AUC Holdout", ascending=False)

best_roc = HOLDOUT_WALK_DF.loc[HOLDOUT_WALK_DF["ROC-AUC"].idxmax()]
best_pr = HOLDOUT_WALK_DF.loc[HOLDOUT_WALK_DF["PR-AUC"].idxmax()]
best_acc = HOLDOUT_WALK_DF.loc[HOLDOUT_WALK_DF["Accuracy"].idxmax()]
best_f1 = HOLDOUT_WALK_DF.loc[HOLDOUT_WALK_DF["F1-Score"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Melhor ROC-AUC", f"{best_roc['Modelo']} ({best_roc['ROC-AUC']:.4f})")
c2.metric("Melhor PR-AUC", f"{best_pr['Modelo']} ({best_pr['PR-AUC']:.4f})")
c3.metric("Melhor Accuracy", f"{best_acc['Modelo']} ({best_acc['Accuracy']:.4f})")
c4.metric("Melhor F1", f"{best_f1['Modelo']} ({best_f1['F1-Score']:.4f})")

st.divider()

st.subheader("🎯 Holdout final com melhores hiperparâmetros do walk-forward")
st.dataframe(
    HOLDOUT_WALK_DF.style.format({col: "{:.4f}" for col in HOLDOUT_WALK_DF.columns if col != "Modelo"}),
    use_container_width=True,
    hide_index=True,
)

holdout_long = HOLDOUT_WALK_DF.melt(
    id_vars="Modelo",
    value_vars=["ROC-AUC", "PR-AUC", "Accuracy", "F1-Score"],
    var_name="Métrica",
    value_name="Valor",
)
fig = px.bar(
    holdout_long,
    x="Modelo",
    y="Valor",
    color="Métrica",
    barmode="group",
    text="Valor",
    range_y=[0.65, 0.95],
    title="Métricas no holdout final",
)
fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "- `Random Forest` lidera em `ROC-AUC` e `PR-AUC`.\n"
    "- `Logistic Regression` lidera em `Accuracy`.\n"
    "- `Decision Tree` lidera em `F1-Score`."
)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Validação vs holdout",
    "📈 ROC / PR",
    "🧩 Matrizes",
    "📚 Interpretação",
])

with tab1:
    st.dataframe(
        summary_df[
            [
                "Modelo",
                "ROC-AUC Walk-forward",
                "ROC-AUC Holdout",
                "Delta Validação->Holdout",
                "PR-AUC",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
            ]
        ].style.format({
            "ROC-AUC Walk-forward": "{:.4f}",
            "ROC-AUC Holdout": "{:.4f}",
            "Delta Validação->Holdout": "{:+.4f}",
            "PR-AUC": "{:.4f}",
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    compare_long = summary_df.melt(
        id_vars="Modelo",
        value_vars=["ROC-AUC Walk-forward", "ROC-AUC Holdout"],
        var_name="Etapa",
        value_name="ROC-AUC",
    )
    fig_compare = px.bar(
        compare_long,
        x="Modelo",
        y="ROC-AUC",
        color="Etapa",
        barmode="group",
        text="ROC-AUC",
        range_y=[0.88, 0.93],
        title="Queda da validação para o holdout",
    )
    fig_compare.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig_compare, use_container_width=True)

    delta_df = summary_df[["Modelo", "Delta Validação->Holdout"]]
    fig_delta = px.bar(
        delta_df,
        x="Modelo",
        y="Delta Validação->Holdout",
        color="Delta Validação->Holdout",
        text="Delta Validação->Holdout",
        color_continuous_scale="RdYlGn",
        title="Delta de generalização em ROC-AUC",
    )
    fig_delta.update_traces(texttemplate="%{text:+.4f}", textposition="outside")
    st.plotly_chart(fig_delta, use_container_width=True)

    with st.expander("Comparação extra: holdout usando melhores params do TSCV"):
        compare_tuning_df = HOLDOUT_WALK_DF.merge(
            HOLDOUT_TSCV_DF,
            on="Modelo",
            suffixes=(" Walk-forward", " TSCV"),
        )
        st.dataframe(
            compare_tuning_df.style.format({
                col: "{:.4f}" for col in compare_tuning_df.columns if col != "Modelo"
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("No notebook 07, os resultados finais com params de TSCV e walk-forward ficam quase idênticos.")

with tab2:
    st.image(
        load_notebook_png(17),
        caption="Curvas ROC e Precision-Recall do holdout final.",
        use_container_width=True,
    )

with tab3:
    st.image(
        load_notebook_png(18),
        caption="Matrizes de confusão do holdout final.",
        use_container_width=True,
    )

with tab4:
    st.markdown(INTERPRETATION_MD)
    st.markdown("**Síntese do notebook 07**")
    st.markdown(SYNTHESIS_MD)

st.markdown("---")
st.caption("Protocolos e tuning ficaram concentrados na página `Validação temporal`.")
