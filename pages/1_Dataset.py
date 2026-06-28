import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dataset", page_icon="📊", layout="wide")
st.title("📊 Dataset e Engenharia de Features")

st.header("Visão geral")
c1, c2, c3 = st.columns(3)
c1.metric("Linhas", "5.587.547")
c2.metric("Colunas", "23")
c3.metric("Mercados únicos", "4.710")

st.header("Features utilizadas nos modelos")
features = pd.DataFrame({
    "Feature": [
        "close_mid", "depth_imbalance", "mean_spread", "close_spread",
        "bar_volatility", "ofi_corrected",
        "close_mid_lag1", "depth_imbalance_lag1", "mean_spread_lag1",
        "close_spread_lag1", "bar_volatility_lag1", "ofi_corrected_lag1",
    ],
    "Tipo": ["preço","liquidez","spread","spread","volatilidade","fluxo de ordens"] * 2,
    "Origem": ["contemporânea"] * 6 + ["lag 1 min"] * 6,
})
st.dataframe(features, use_container_width=True, hide_index=True)

st.caption(
    "Excluídas: `market_id` (identificador), `minute_bar` (timestamp bruto), "
    "`total_volume`, `buy_volume`, `sell_volume`, `trade_count` (baixo poder preditivo na EDA), "
    "`bid_depth`, `ask_depth` (multicolinearidade com `depth_imbalance`)."
)

st.header("Desbalanceamento do target")
target_df = pd.DataFrame({
    "Conjunto": ["Treino", "Treino", "Teste", "Teste"],
    "Classe": ["0 (não sobe)", "1 (sobe)", "0 (não sobe)", "1 (sobe)"],
    "Percentual": [73.37, 26.63, 73.22, 26.78],
})
fig = px.bar(
    target_df, x="Conjunto", y="Percentual", color="Classe",
    barmode="stack", text="Percentual",
    color_discrete_map={"0 (não sobe)": "#ef553b", "1 (sobe)": "#00cc96"},
)
fig.update_traces(texttemplate="%{text:.2f}%", textposition="inside")
st.plotly_chart(fig, use_container_width=True)

st.success(
    "A proporção das classes se manteve praticamente idêntica entre treino e teste (Δ < 0,2 p.p.), "
    "indicando que a divisão temporal preservou o desbalanceamento natural do problema."
)

st.header("Sobreposição de mercados treino × teste")
overlap = pd.DataFrame({
    "Categoria": ["Mercados em comum", "Apenas no teste"],
    "Quantidade": [862, 640],
    "Percentual": [57.39, 42.61],
})
fig2 = px.pie(overlap, values="Quantidade", names="Categoria", hole=0.5)
st.plotly_chart(fig2, use_container_width=True)

st.warning(
    "**640 mercados (42,61%)** aparecem apenas no teste — o modelo precisa generalizar para contratos "
    "nunca vistos no treino, cenário mais próximo de operação real."
)

st.header("Divisão temporal")
st.markdown(
    """
    | Conjunto | Período | Linhas aproximadas |
    |----------|---------|-------------------:|
    | **Treino** | 06/03/2026 → 09/03/2026 | ~4,5M |
    | **Teste (holdout)** | 10/03/2026 → 11/03/2026 23:59 UTC | ~638k |
    """
)
