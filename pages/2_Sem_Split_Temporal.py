import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Sem Split Temporal", page_icon="📈", layout="wide")
st.title("📈 Modelo inicial — LightGBM sem split temporal (`lightgbm.ipynb`)")

st.markdown(
    """
    Primeira tentativa, antes da adoção de validação temporal. Usa `train_test_split` com
    `shuffle=False` no fim das linhas (80/20 cronológico simples), early stopping em 50 rounds.
    Serve como **ponto de partida** para mostrar o problema que o split temporal resolve.
    """
)

st.header("Resultado no conjunto de validação")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", "0,8446")
c2.metric("Precision", "0,8397")
c3.metric("Recall", "0,4979")
c4.metric("F1-Score", "0,6251")
c5.metric("ROC-AUC", "0,9092")

st.header("Matriz de confusão")
cm = [[799144, 27636], [145971, 144759]]
fig = go.Figure(
    data=go.Heatmap(
        z=cm,
        x=["Pred. 0", "Pred. 1"],
        y=["Real 0", "Real 1"],
        text=cm, texttemplate="%{text}",
        colorscale="Blues",
    )
)
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

st.header("Hiperparâmetros principais")
st.code(
    """
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
}
early_stopping_rounds = 50
best_iteration = 102
    """,
    language="python",
)

st.warning(
    "Este resultado parece bom mas **superestima** o desempenho real: o split 80/20 cronológico "
    "simples mistura observações próximas no tempo dos mesmos mercados. Veja a página "
    "**Comparação de Splits** para o impacto quantitativo."
)
