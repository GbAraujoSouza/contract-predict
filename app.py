import streamlit as st

st.set_page_config(
    page_title="Polymarket ML Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Engenharia Financeira de Alta Performance — Parte II")
st.subheader("Previsão de movimento de preço em mercados Polymarket (horizonte 15 min)")

st.markdown(
    """
    Este dashboard consolida os resultados do trabalho final de **Introdução ao Aprendizado de Máquina**.
    O objetivo é prever se o preço de um contrato Polymarket **sobe nos próximos 15 minutos** (`target = 1`)
    a partir de features de microestrutura de mercado em barras de 1 minuto.

    ### Navegação
    - **Dataset** — visão geral dos dados, features e desbalanceamento de classes
    - **Sem Split Temporal (lightdm)** — modelo inicial LightGBM, 80/20 cronológico sem validação temporal
    - **Regressão Logística** — baseline linear com `TimeSeriesSplit` + holdout por data
    - **Modelos de Árvore** — LightGBM, Decision Tree e Random Forest com protocolo temporal
    - **Comparação de Splits** — impacto de embaralhar dados em série temporal
    - **Walk-Forward** — protocolo mais rigoroso com tuning por ROC-AUC e purge gap

    ### Pipeline geral
    """
)

st.graphviz_chart(
    """
    digraph G {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fillcolor="#1f77b4", fontcolor=white, fontname=Helvetica];
        A [label="Coleta Kaggle\\n5,5M linhas\\n4.710 mercados"];
        B [label="Feature engineering\\n16 colunas + lags"];
        C [label="Split temporal\\nTreino 06–09/03\\nTeste 10–11/03"];
        D [label="TimeSeriesSplit\\n5 folds"];
        E [label="Walk-forward\\n2d treino / 1d val"];
        F [label="Holdout final\\nROC-AUC, F1, Prec, Rec"];
        A -> B -> C -> D -> F;
        C -> E -> F;
    }
    """
)

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Linhas totais", "5.587.547")
col2.metric("Mercados", "4.710")
col3.metric("Janela", "06–11/03/2026")
col4.metric("Horizonte alvo", "15 min")

st.info(
    "Dados base: dataset Kaggle de orderbook Polymarket. "
    "A tentativa de expansão via API pública não foi viável: a API não expõe histórico de orderbook "
    "(spread, depth, OFI) retroativamente."
)
