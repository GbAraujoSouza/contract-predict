import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Comparação de Splits", page_icon="⚖️", layout="wide")
st.title("⚖️ Com vs Sem Split Temporal")

st.markdown(
    """
    Comparação dos **mesmos 3 modelos, 12 features e hiperparâmetros** sob três protocolos:

    | Protocolo | Descrição | Tamanho do teste |
    |-----------|-----------|------------------|
    | **Com split temporal** | Treino até 09/03, holdout 10–11/03 + TSCV | 638k linhas (futuro real) |
    | **80/20 cronológico** | Como `lightdm.ipynb`: últimos 20% das linhas | 1,12M linhas |
    | **80/20 aleatório** | `train_test_split(shuffle=True)` — vaza tempo | 1,12M linhas |
    """
)

st.header("ROC-AUC por modelo e protocolo")
auc_df = pd.DataFrame({
    "Modelo": ["LightGBM", "Random Forest", "Árvore de Decisão"] * 3,
    "Protocolo": (["Com split temporal"] * 3 +
                  ["80/20 cronológico"] * 3 +
                  ["80/20 aleatório"] * 3),
    "ROC-AUC": [0.906, 0.911, 0.902,
                0.894, 0.892, 0.881,
                0.909, 0.923, 0.919],
})
fig = px.bar(
    auc_df, x="Modelo", y="ROC-AUC", color="Protocolo", barmode="group",
    text="ROC-AUC",
    color_discrete_map={
        "Com split temporal": "#2ca02c",
        "80/20 cronológico": "#ff7f0e",
        "80/20 aleatório": "#d62728",
    },
)
fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig.update_layout(yaxis_range=[0.85, 0.94])
st.plotly_chart(fig, use_container_width=True)

st.header("F1-Score por modelo e protocolo")
f1_df = pd.DataFrame({
    "Modelo": ["LightGBM", "Random Forest", "Árvore de Decisão"] * 3,
    "Protocolo": (["Com split temporal"] * 3 +
                  ["80/20 cronológico"] * 3 +
                  ["80/20 aleatório"] * 3),
    "F1": [0.668, 0.669, 0.673,
           0.692, 0.698, 0.689,
           0.685, 0.703, 0.697],
})
fig2 = px.bar(
    f1_df, x="Modelo", y="F1", color="Protocolo", barmode="group", text="F1",
    color_discrete_map={
        "Com split temporal": "#2ca02c",
        "80/20 cronológico": "#ff7f0e",
        "80/20 aleatório": "#d62728",
    },
)
fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig2.update_layout(yaxis_range=[0.6, 0.75])
st.plotly_chart(fig2, use_container_width=True)

st.header("Trade-off Precisão × Recall")
tradeoff = pd.DataFrame({
    "Modelo": ["LightGBM", "LightGBM", "Random Forest", "Random Forest"],
    "Protocolo": ["Temporal", "Aleatório", "Temporal", "Aleatório"],
    "Precisão": [0.523, 0.563, 0.522, 0.576],
    "Recall": [0.926, 0.877, 0.928, 0.902],
})
st.dataframe(tradeoff, use_container_width=True, hide_index=True)

st.header("Leitura principal")
st.error(
    "**Split aleatório infla AUC em ~1–2 p.p.** — o treino vê minutos futuros do mesmo mercado, "
    "métricas ficam otimistas demais."
)
st.success(
    "**Split temporal é mais conservador e honesto** — simula previsão em dias que o modelo nunca "
    "viu, refletindo o cenário real de produção."
)
st.info(
    "**Ranking estável:** Random Forest > LightGBM > Árvore de Decisão nos três protocolos. "
    "A escolha do split não muda o vencedor, mas muda o número que se reporta."
)

st.header("Expansão do dataset via API Polymarket")
st.markdown(
    """
    Tentativa de ampliar a janela com `polymarket_client.py` + `backend/pipelines/01_api_collect.py`.

    **A API pública não expõe histórico de orderbook.** Faltam 6 colunas obrigatórias:
    `mean_spread`, `close_spread`, `bid_depth`, `ask_depth`, `depth_imbalance`, `order_flow_imbalance`.
    """
)
api = pd.DataFrame({
    "Métrica": ["Janela", "Mercados", "Linhas", "Schema completo"],
    "Dataset Kaggle": ["06–11/03/2026 (6 dias)", "4.710", "5,5M", "Sim (16 colunas)"],
    "Coleta API": ["~27/05–27/06/2026 (~31 dias)", "7 de 20 solicitados", "5.068", "Não (10 colunas parciais)"],
})
st.dataframe(api, use_container_width=True, hide_index=True)
