import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Regressão Logística", page_icon="🧠", layout="wide")
st.title("🧠 Regressão Logística — baseline com split temporal")

st.markdown(
    """
    Baseline linear seguindo o protocolo:
    1. Holdout por data (treino até 09/03, teste 10–11/03)
    2. `TimeSeriesSplit` (5 folds) somente no conjunto de treino
    3. `StandardScaler` ajustado por fold (evita leakage)
    4. Avaliação final no holdout futuro
    """
)

st.header("Validação temporal — TimeSeriesSplit (5 folds)")
folds = pd.DataFrame({
    "Fold": [1, 2, 3, 4, 5],
    "AUC": [0.8811, 0.8790, 0.8926, 0.8890, 0.8921],
    "F1": [0.6388, 0.5954, 0.6426, 0.5975, 0.6243],
    "Precision": [0.7385, 0.7223, 0.7881, 0.6742, 0.7517],
    "Recall": [0.5627, 0.5064, 0.5424, 0.5365, 0.5337],
})
st.dataframe(folds, use_container_width=True, hide_index=True)

fig = px.line(
    folds.melt(id_vars="Fold", var_name="Métrica", value_name="Valor"),
    x="Fold", y="Valor", color="Métrica", markers=True,
)
fig.update_layout(yaxis_range=[0.4, 1.0])
st.plotly_chart(fig, use_container_width=True)

st.subheader("Médias da validação cruzada")
c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC", "0,8868 ± 0,0057")
c2.metric("F1", "0,6197 ± 0,0200")
c3.metric("Precision", "0,7350 ± 0,0374")
c4.metric("Recall", "0,5363 ± 0,0181")

st.header("Holdout final (10–11/03/2026)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC-AUC", "0,8942")
c2.metric("F1", "0,6272")
c3.metric("Precision", "0,6019")
c4.metric("Recall", "0,6546")

st.success(
    "A Regressão Logística cumpre seu papel de baseline com excelência (AUC ≈ 0,89), "
    "validando que a engenharia de features estruturou relações sólidas no dataset."
)
