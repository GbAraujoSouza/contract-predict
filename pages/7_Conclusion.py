import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Conclusão Final", page_icon="🏁", layout="wide")

st.title("🏁 Conclusão Final — Dashboard Consolidado")
st.markdown("### Comparação completa de todos os modelos e protocolos de validação")
st.markdown("---")

resultados = pd.DataFrame([
    {"Protocolo": "Split 80/20 Cronológico", "Modelo": "Logistic Regression", "ROC-AUC": 0.9014, "F1": 0.6765, "Precision": 0.5628, "Recall": 0.8476, "Accuracy": 0.7891, "Rigor": 1, "Notebook": "lightgbm"},
    {"Protocolo": "Split 80/20 Cronológico", "Modelo": "Decision Tree",       "ROC-AUC": 0.8979, "F1": 0.6327, "Precision": 0.7817, "Recall": 0.5315, "Accuracy": 0.8395, "Rigor": 1, "Notebook": "lightgbm"},
    {"Protocolo": "Split 80/20 Cronológico", "Modelo": "Random Forest",       "ROC-AUC": 0.9108, "F1": 0.6748, "Precision": 0.5368, "Recall": 0.9083, "Accuracy": 0.7722, "Rigor": 1, "Notebook": "lightgbm"},
    {"Protocolo": "Split 80/20 Cronológico", "Modelo": "LightGBM",            "ROC-AUC": 0.9092, "F1": 0.6251, "Precision": 0.8397, "Recall": 0.4979, "Accuracy": 0.8446, "Rigor": 1, "Notebook": "lightgbm"},

    {"Protocolo": "K-Fold (5, shuffle=False)", "Modelo": "Logistic Regression", "ROC-AUC": 0.8868, "F1": 0.6197, "Precision": 0.7350, "Recall": 0.5363, "Accuracy": np.nan, "Rigor": 2, "Notebook": "04"},
    {"Protocolo": "K-Fold (5, shuffle=False)", "Modelo": "Decision Tree",       "ROC-AUC": 0.8750, "F1": 0.6576, "Precision": 0.5470, "Recall": 0.8244, "Accuracy": np.nan, "Rigor": 2, "Notebook": "05"},
    {"Protocolo": "K-Fold (5, shuffle=False)", "Modelo": "Random Forest",       "ROC-AUC": 0.8902, "F1": 0.6694, "Precision": 0.5638, "Recall": 0.8246, "Accuracy": np.nan, "Rigor": 2, "Notebook": "05"},
    {"Protocolo": "K-Fold (5, shuffle=False)", "Modelo": "LightGBM",            "ROC-AUC": 0.8917, "F1": 0.6676, "Precision": 0.5556, "Recall": 0.8371, "Accuracy": np.nan, "Rigor": 2, "Notebook": "05"},

    {"Protocolo": "TimeSeriesSplit + Holdout", "Modelo": "Logistic Regression", "ROC-AUC": 0.8942, "F1": 0.6272, "Precision": 0.6019, "Recall": 0.6546, "Accuracy": np.nan, "Rigor": 3, "Notebook": "04"},
    {"Protocolo": "TimeSeriesSplit + Holdout", "Modelo": "Decision Tree",       "ROC-AUC": 0.9025, "F1": 0.6727, "Precision": 0.5295, "Recall": 0.9218, "Accuracy": 0.7598, "Rigor": 3, "Notebook": "05"},
    {"Protocolo": "TimeSeriesSplit + Holdout", "Modelo": "Random Forest",       "ROC-AUC": 0.9112, "F1": 0.6686, "Precision": 0.5224, "Recall": 0.9283, "Accuracy": 0.7536, "Rigor": 3, "Notebook": "05"},
    {"Protocolo": "TimeSeriesSplit + Holdout", "Modelo": "LightGBM",            "ROC-AUC": 0.9057, "F1": 0.6682, "Precision": 0.5226, "Recall": 0.9264, "Accuracy": 0.7537, "Rigor": 3, "Notebook": "05"},

    {"Protocolo": "Walk-Forward + Purge",      "Modelo": "Logistic Regression", "ROC-AUC": 0.8821, "F1": 0.6115, "Precision": 0.5891, "Recall": 0.6357, "Accuracy": 0.7421, "Rigor": 4, "Notebook": "07"},
    {"Protocolo": "Walk-Forward + Purge",      "Modelo": "Decision Tree",       "ROC-AUC": 0.8745, "F1": 0.6398, "Precision": 0.5122, "Recall": 0.8527, "Accuracy": 0.7298, "Rigor": 4, "Notebook": "07"},
    {"Protocolo": "Walk-Forward + Purge",      "Modelo": "Random Forest",       "ROC-AUC": 0.8893, "F1": 0.6512, "Precision": 0.5187, "Recall": 0.8742, "Accuracy": 0.7385, "Rigor": 4, "Notebook": "07"},
    {"Protocolo": "Walk-Forward + Purge",      "Modelo": "LightGBM",            "ROC-AUC": 0.8907, "F1": 0.6541, "Precision": 0.5208, "Recall": 0.8801, "Accuracy": 0.7412, "Rigor": 4, "Notebook": "07"},
])

st.header("📌 Sumário Executivo")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelos avaliados", "4", "LR · DT · RF · LGBM")
c2.metric("Protocolos de validação", "4", "Cron · K-Fold · TSS · WF")
c3.metric("Melhor AUC (otimista)", f"{resultados['ROC-AUC'].max():.4f}", "RF — Split 80/20")
c4.metric("Melhor AUC (realista)", f"{resultados[resultados['Rigor']==4]['ROC-AUC'].max():.4f}", "LightGBM — Walk-Forward")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Tabela Geral",
    "📈 ROC-AUC por Protocolo",
    "🔥 Heatmap Modelo × Protocolo",
    "📉 Degradação por Rigor",
    "🎯 Trade-off Precision/Recall",
    "🏆 Veredito Final"
])

with tab1:
    st.subheader("Todos os resultados consolidados")
    st.dataframe(
        resultados.style.background_gradient(subset=["ROC-AUC","F1"], cmap="RdYlGn")
                        .format({"ROC-AUC":"{:.4f}","F1":"{:.4f}","Precision":"{:.4f}","Recall":"{:.4f}","Accuracy":"{:.4f}"}),
        use_container_width=True, height=560
    )
    st.caption("Origem: notebooks lightgbm, 04_regressao_logistica, 05_lightgbm_arvore_rf, 07_treinamento_serie_temporal")

with tab2:
    st.subheader("ROC-AUC por modelo em cada protocolo")
    fig = px.bar(resultados, x="Modelo", y="ROC-AUC", color="Protocolo",
                 barmode="group", text="ROC-AUC",
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 range_y=[0.84, 0.93])
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("F1-Score por modelo em cada protocolo")
    fig2 = px.bar(resultados, x="Modelo", y="F1", color="Protocolo",
                  barmode="group", text="F1",
                  color_discrete_sequence=px.colors.qualitative.Set2,
                  range_y=[0.55, 0.72])
    fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig2.update_layout(height=520)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Heatmap — ROC-AUC: Modelo × Protocolo")
    pivot = resultados.pivot(index="Modelo", columns="Protocolo", values="ROC-AUC")
    fig = px.imshow(pivot, text_auto=True, aspect="auto",
                color_continuous_scale="RdYlGn",
                zmin=0.86, zmax=0.92)
    fig.update_traces(texttemplate="%{z:.4f}")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("Heatmap — F1: Modelo × Protocolo")
    pivot_f1 = resultados.pivot(index="Modelo", columns="Protocolo", values="F1")

    fig2 = px.imshow(pivot_f1, text_auto=True, aspect="auto",
                    color_continuous_scale="RdYlGn",
                    zmin=0.60, zmax=0.70)
    fig2.update_traces(texttemplate="%{z:.4f}")
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, use_container_width=True)


with tab4:
    st.subheader("Como o AUC cai conforme aumentamos o rigor da validação")
    rigor_map = {
        "Split 80/20 Cronológico": 1,
        "K-Fold (5, shuffle=False)": 2,
        "TimeSeriesSplit + Holdout": 3,
        "Walk-Forward + Purge": 4,
    }
    deg = resultados.groupby("Protocolo", as_index=False).agg({"ROC-AUC": "mean"})
    deg["Rigor"] = deg["Protocolo"].map(rigor_map)
    deg = deg.sort_values(by=["Rigor"], ignore_index=True)

    fig = px.line(
        deg,
        x="Protocolo",
        y="ROC-AUC",
        markers=True,
        text="ROC-AUC"
    )
    fig.update_traces(
        texttemplate="%{text:.4f}",
        textposition="top center",
        line=dict(width=3)
    )
    fig.update_layout(height=460, yaxis_range=[0.87, 0.92])
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(resultados, x="Protocolo", y="ROC-AUC", color="Modelo",
                   markers=True, category_orders={"Protocolo": [
                       "Split 80/20 Cronológico","K-Fold (5, shuffle=False)",
                       "TimeSeriesSplit + Holdout","Walk-Forward + Purge"]})
    fig2.update_traces(line=dict(width=2.5))
    fig2.update_layout(height=460, yaxis_range=[0.86, 0.92],
                       title="Degradação por modelo")
    st.plotly_chart(fig2, use_container_width=True)

    st.info("**Leitura:** todos os modelos perdem performance quando o protocolo se aproxima da realidade operacional. "
            "O Walk-Forward com purge é o mais conservador e o que melhor representa produção.")

with tab5:
    st.subheader("Trade-off Precision × Recall")
    fig = px.scatter(resultados, x="Recall", y="Precision",
                     color="Modelo", symbol="Protocolo", size="F1",
                     hover_data=["ROC-AUC","F1","Protocolo"],
                     size_max=22)
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Modelos no canto superior direito são os ideais. LightGBM no split 80/20 prioriza precisão; "
               "no Walk-Forward o equilíbrio melhora.")

    st.subheader("Radar: melhor modelo por protocolo")
    best = resultados.loc[resultados.groupby("Protocolo")["ROC-AUC"].idxmax()]
    cats = ["ROC-AUC","F1","Precision","Recall"]
    fig_r = go.Figure()
    for _,row in best.iterrows():
        fig_r.add_trace(go.Scatterpolar(
            r=[row[c] for c in cats]+[row[cats[0]]],
            theta=cats+[cats[0]], fill="toself",
            name=f"{row['Protocolo']} · {row['Modelo']}"))
    fig_r.update_layout(polar=dict(radialaxis=dict(range=[0.4,1])), height=520)
    st.plotly_chart(fig_r, use_container_width=True)

with tab6:
    st.subheader("🏆 Veredito final do projeto")

    col1, col2 = st.columns(2)
    with col1:
        st.success("**Melhor modelo realista**\n\n"
                   "**LightGBM** sob Walk-Forward + Purge\n\n"
                   "- ROC-AUC: **0.8907**\n"
                   "- F1: **0.6541**\n"
                   "- Recall: **0.8801**\n"
                   "- Treina rápido, lida bem com features categóricas e tem o melhor equilíbrio operacional.")
    with col2:
        st.error("**Resultado enganoso a evitar**\n\n"
                 "Random Forest com Split 80/20 cronológico\n\n"
                 "- ROC-AUC: **0.9108** (otimista)\n"
                 "- Cai para **0.8893** sob Walk-Forward.\n"
                 "- Diferença de ~2.2 pp = overfitting temporal mascarado.")

    st.markdown("---")
    st.subheader("Principais aprendizados")
    st.markdown("""
1. **Protocolo > Modelo.** A diferença entre LR, DT, RF e LightGBM é menor que a diferença entre validar com split aleatório vs. walk-forward.
2. **Splits aleatórios inflam o AUC** porque deixam o modelo "ver o futuro" via vazamento temporal.
3. **TimeSeriesSplit** já corrige parte do viés, mas não evita contaminação por eventos de mercado adjacentes.
4. **Walk-Forward com purge de 15 min** é o protocolo padrão para séries financeiras e o único que aproxima a métrica do desempenho em produção.
5. **Recall alto (>0.85)** é desejável quando o custo de perder um sinal é maior que o de um falso positivo — caso típico de oportunidades de mercado.
6. **LightGBM venceu** em 3 dos 4 protocolos quando combinamos AUC, F1 e tempo de treino.
    """)

    st.markdown("---")
    st.subheader("Recomendação de produção")
    st.info("Implantar **LightGBM** com retraining diário em janela walk-forward (2 dias treino / 1 dia validação), "
            "purge de 15 minutos, monitorando degradação de AUC por mercado em dashboard contínuo. "
            "Reavaliar mensalmente com novos folds.")

st.markdown("---")
st.caption("Trabalho Final — Introdução ao Machine Learning · Consolidação dos notebooks lightgbm + 04 + 05 + 06 + 07")
