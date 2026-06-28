import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Modelos de Árvore", page_icon="🌳", layout="wide")
st.title("🌳 LightGBM, Decision Tree e Random Forest")

st.markdown(
    "Mesmo protocolo da Regressão Logística: holdout por data + `TimeSeriesSplit` (5 folds). "
    "Modelos de árvore não precisam de `StandardScaler`."
)

st.header("Validação temporal — médias")
cv = pd.DataFrame({
    "Modelo": ["LightGBM", "Decision Tree", "Random Forest"],
    "AUC médio": [0.8917, 0.8750, 0.8918],
    "AUC std": [0.0062, 0.0108, 0.0050],
    "F1 médio": [0.6676, 0.6576, 0.6694],
    "F1 std": [0.0269, 0.0235, 0.0270],
})
st.dataframe(cv, use_container_width=True, hide_index=True)

fig = px.bar(
    cv.melt(id_vars="Modelo", value_vars=["AUC médio", "F1 médio"],
            var_name="Métrica", value_name="Valor"),
    x="Modelo", y="Valor", color="Métrica", barmode="group", text="Valor",
)
fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig.update_layout(yaxis_range=[0.5, 1.0])
st.plotly_chart(fig, use_container_width=True)

st.header("Holdout final (10–11/03/2026)")
holdout = pd.DataFrame({
    "Modelo": ["LightGBM", "Decision Tree", "Random Forest"],
    "Accuracy": [0.7537, 0.7598, 0.7536],
    "Precision": [0.5226, 0.5295, 0.5224],
    "Recall": [0.9264, 0.9218, 0.9283],
    "F1": [0.6682, 0.6727, 0.6686],
    "ROC-AUC": [0.9057, 0.9025, 0.9112],
})
st.dataframe(
    holdout.style.format({c: "{:.4f}" for c in holdout.columns if c != "Modelo"})
                 .highlight_max(axis=0, subset=holdout.columns[1:], color="#0a5d2c"),
    use_container_width=True, hide_index=True,
)

st.subheader("ROC-AUC no holdout")
fig2 = go.Figure(
    go.Bar(
        x=holdout["Modelo"], y=holdout["ROC-AUC"],
        text=[f"{v:.4f}" for v in holdout["ROC-AUC"]],
        textposition="outside",
        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
)
fig2.update_layout(yaxis_range=[0.85, 0.93], height=400)
st.plotly_chart(fig2, use_container_width=True)

st.header("Trade-off Precision × Recall")
fig3 = px.scatter(
    holdout, x="Precision", y="Recall", color="Modelo", size="ROC-AUC",
    text="Modelo", size_max=40,
)
fig3.update_traces(textposition="top center")
st.plotly_chart(fig3, use_container_width=True)

st.warning(
    "Todos os três modelos priorizam **recall alto (~92%)** em detrimento de precisão (~52%). "
    "Aproximadamente metade dos alertas de subida são falsos positivos — adequado se o custo "
    "de perder uma subida é maior que o de operar em alarme falso."
)

st.header("Conclusões")
st.markdown(
    """
    1. **Random Forest** lidera no holdout (ROC-AUC = 0,9112), seguido por **LightGBM** (0,9057).
    2. **LightGBM** é a escolha técnica definitiva pelo custo computacional muito menor —
       fundamental para operar fluxos massivos de orderbook em alta frequência.
    3. A **Decision Tree** sozinha é competitiva (AUC 0,9025) mas com maior variância entre folds.
    4. O ranking entre modelos é estável: ensembles > árvore única > linear.
    """
)
