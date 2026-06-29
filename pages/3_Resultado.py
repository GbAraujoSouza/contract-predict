import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Resultados", page_icon="📑", layout="wide")

st.title("📑 Resultados — Modelos e Protocolos de Validação")
st.caption("Consolidação dos notebooks 04 (Regressão Logística), 05 (Modelos de Árvore) e 06 (Comparação de Splits).")

tab1, tab2, tab3 = st.tabs([
    "🧠 Regressão Logística",
    "🌳 Modelos de Árvore",
    "⚖️ Comparação de Splits",
])

# =====================================================================
with tab1:
    st.subheader("Regressão Logística com TimeSeriesSplit")
    st.markdown(
        "Notebook **04** — baseline linear treinado com validação temporal "
        "(`TimeSeriesSplit`, 5 folds) e avaliado em holdout cronológico."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC (holdout)", "0.8942")
    c2.metric("Accuracy", "0.8123")
    c3.metric("F1", "0.7894")
    c4.metric("PR-AUC", "0.8456")

    folds_lr = pd.DataFrame({
        "Fold": [f"Fold {i}" for i in range(1, 6)],
        "ROC-AUC": [0.8821, 0.8867, 0.8902, 0.8915, 0.8938],
    })
    fig = px.bar(folds_lr, x="Fold", y="ROC-AUC", text="ROC-AUC",
                 range_y=[0.86, 0.90], title="ROC-AUC por fold (TimeSeriesSplit)")
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Observações:** estabilidade alta entre folds (desvio < 0.01). "
        "Serve como piso de comparação para os modelos não lineares."
    )

# =====================================================================
with tab2:
    st.subheader("Modelos baseados em árvore")
    st.markdown("Notebook **05** — comparação entre Decision Tree, Random Forest e LightGBM em split cronológico 80/20.")

    arvore = pd.DataFrame({
        "Modelo": ["Decision Tree", "Random Forest", "LightGBM"],
        "ROC-AUC": [0.8312, 0.8821, 0.9047],
        "Accuracy": [0.7621, 0.8189, 0.8401],
        "F1": [0.7423, 0.7984, 0.8245],
        "PR-AUC": [0.7812, 0.8456, 0.8912],
    })
    st.dataframe(arvore, use_container_width=True, hide_index=True)

    fig = px.bar(arvore.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor"),
                 x="Modelo", y="Valor", color="Métrica", barmode="group",
                 text="Valor", range_y=[0.7, 0.92],
                 title="Métricas por modelo")
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Importância de features (LightGBM)")
    imp = pd.DataFrame({
        "Feature": ["price_lag_1", "volume_lag_1", "price_mean_5", "spread",
                    "price_lag_3", "volume_mean_5", "hour", "weekday",
                    "price_lag_5", "volume_lag_3"],
        "Importance": [0.182, 0.154, 0.121, 0.098, 0.087, 0.076, 0.062, 0.054, 0.048, 0.041],
    })
    fig2 = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature",
                  orientation="h", text="Importance")
    fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, use_container_width=True)

    st.success("LightGBM domina nas 4 métricas e vira o modelo de referência para os protocolos seguintes.")

# =====================================================================
with tab3:
    st.subheader("Como o tipo de split afeta a métrica")
    st.markdown(
        "Notebook **06** — mesmo modelo (LightGBM), mesma feature set, "
        "trocando apenas o protocolo de validação."
    )

    splits = pd.DataFrame({
        "Protocolo": [
            "Split aleatório 80/20",
            "K-Fold (5, shuffle=True)",
            "Split cronológico 80/20",
            "K-Fold (5, shuffle=False)",
            "TimeSeriesSplit (5)",
        ],
        "ROC-AUC": [0.9412, 0.9389, 0.9047, 0.9012, 0.8965],
        "Rigor": [1, 2, 3, 4, 5],
    })

    fig = px.bar(splits.sort_values("Rigor"), x="Protocolo", y="ROC-AUC",
                 text="ROC-AUC", color="ROC-AUC",
                 color_continuous_scale="RdYlGn_r", range_y=[0.86, 0.96])
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(height=460, xaxis_tickangle=-15)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(splits.sort_values(by=["Rigor"]), x="Protocolo", y="ROC-AUC",
                   markers=True, text="ROC-AUC",
                   title="Queda do AUC conforme o protocolo fica mais rigoroso")
    fig2.update_traces(texttemplate="%{text:.4f}", textposition="top center",
                       line=dict(width=3))
    fig2.update_layout(height=420, yaxis_range=[0.88, 0.95], xaxis_tickangle=-15)
    st.plotly_chart(fig2, use_container_width=True)

    st.warning(
        "**Diferença de ~4,5 pontos de AUC** entre o split aleatório e o "
        "TimeSeriesSplit usando exatamente o mesmo modelo — evidência de "
        "vazamento temporal quando se embaralha o tempo."
    )

st.markdown("---")
st.caption("Trabalho Final — Introdução ao Machine Learning · Notebooks 04 + 05 + 06")
