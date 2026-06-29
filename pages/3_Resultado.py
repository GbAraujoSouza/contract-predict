import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Resultados", page_icon="📑", layout="wide")

st.title("📑 Resultados Atualizados")
st.caption("Consolidação do notebook `07_treinamento_serie_temporal.ipynb`.")

validation_df = pd.DataFrame([
    {"Modelo": "Random Forest", "ROC-AUC TSCV": 0.914666, "ROC-AUC Walk-forward": 0.922445},
    {"Modelo": "LightGBM", "ROC-AUC TSCV": 0.912963, "ROC-AUC Walk-forward": 0.921792},
    {"Modelo": "Decision Tree", "ROC-AUC TSCV": 0.901458, "ROC-AUC Walk-forward": 0.914585},
    {"Modelo": "Logistic Regression", "ROC-AUC TSCV": 0.893422, "ROC-AUC Walk-forward": 0.905235},
])
validation_df["Ganho Walk-forward"] = (
    validation_df["ROC-AUC Walk-forward"] - validation_df["ROC-AUC TSCV"]
)

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
    {"Protocolo": "TimeSeriesSplit", "Tempo (s)": 1246.242, "Fits totais": 80},
    {"Protocolo": "Walk-forward", "Tempo (s)": 619.429, "Fits totais": 32},
])

tabs = st.tabs([
    "🧠 Regressão Logística",
    "🌳 Modelos de Árvore",
    "⚖️ Protocolos Temporais",
])

with tabs[0]:
    st.subheader("Logistic Regression no protocolo temporal")
    logistic_validation = validation_df.loc[
        validation_df["Modelo"] == "Logistic Regression"
    ].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC TSCV", f"{logistic_validation['ROC-AUC TSCV']:.4f}")
    c2.metric("ROC-AUC Walk-forward", f"{logistic_validation['ROC-AUC Walk-forward']:.4f}")
    c3.metric("ROC-AUC holdout", "0.8938")
    c4.metric("Accuracy holdout", "0.7704")

    logistic_summary = pd.DataFrame([
        {"Cenário": "Melhor tuning via TSCV", "Valor": "C = 0.1"},
        {"Cenário": "Melhor tuning via Walk-forward", "Valor": "C = 0.01"},
        {"Cenário": "PR-AUC no holdout", "Valor": "0.7904"},
        {"Cenário": "F1 no holdout", "Valor": "0.6703"},
    ])
    st.dataframe(logistic_summary, use_container_width=True, hide_index=True)

    logistic_chart = pd.DataFrame([
        {"Etapa": "Validação TSCV", "ROC-AUC": 0.893422},
        {"Etapa": "Validação Walk-forward", "ROC-AUC": 0.905235},
        {"Etapa": "Holdout final", "ROC-AUC": 0.893813},
    ])
    fig = px.bar(
        logistic_chart,
        x="Etapa",
        y="ROC-AUC",
        text="ROC-AUC",
        range_y=[0.88, 0.91],
        color="Etapa",
        title="Logistic Regression: tuning temporal vs holdout",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Walk-forward melhora o score de validação da regressão logística, "
        "mas no holdout ela fica abaixo dos ensembles em ROC-AUC e PR-AUC."
    )

with tabs[1]:
    st.subheader("Modelos de árvore no holdout final")
    tree_df = holdout_df[holdout_df["Modelo"] != "Logistic Regression"].copy()
    st.dataframe(
        tree_df.style.format({col: "{:.4f}" for col in tree_df.columns if col != "Modelo"}),
        use_container_width=True,
        hide_index=True,
    )

    tree_long = tree_df.melt(
        id_vars="Modelo",
        value_vars=["ROC-AUC", "PR-AUC", "Accuracy", "F1-Score"],
        var_name="Métrica",
        value_name="Valor",
    )
    fig = px.bar(
        tree_long,
        x="Modelo",
        y="Valor",
        color="Métrica",
        barmode="group",
        text="Valor",
        range_y=[0.65, 0.95],
        title="Árvores: comparação no holdout",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "- `Random Forest` lidera em `ROC-AUC` e `PR-AUC`.\n"
        "- `Decision Tree` fica com melhor `Accuracy` e melhor `F1-Score`.\n"
        "- `LightGBM` segue muito próximo, mas não vence nenhuma métrica final."
    )

with tabs[2]:
    st.subheader("TimeSeriesSplit vs Walk-forward")
    c1, c2 = st.columns(2)
    c1.metric("Tempo TSCV", "1246.2 s")
    c2.metric("Tempo Walk-forward", "619.4 s", delta="-626.8 s")

    protocol_df = validation_df.copy()
    protocol_df["Ganho Walk-forward"] = protocol_df["Ganho Walk-forward"].map(lambda x: f"{x:.4f}")
    st.dataframe(
        protocol_df.style.format({
            "ROC-AUC TSCV": "{:.4f}",
            "ROC-AUC Walk-forward": "{:.4f}",
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
        title="ROC-AUC médio na validação temporal",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Neste notebook, `Walk-forward` supera `TimeSeriesSplit` em todos os modelos "
        "e ainda custa menos tempo de execução por usar 2 folds em vez de 5."
    )

st.markdown("---")
st.caption("Trabalho Final · Página atualizada com métricas do notebook 07")
