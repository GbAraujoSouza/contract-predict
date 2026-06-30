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
    {"Protocolo": "Split 80/20 Cronológico", "Modelo": "LightGBM",            "ROC-AUC": 0.908397, "PR-AUC": np.nan, "F1": 0.626999, "Precision": 0.839801, "Recall": 0.500241, "Accuracy": 0.845158,},
    {"Protocolo": "Split 80/20 Cronológico", "Modelo": "Logistic Regression", "ROC-AUC": 0.900260, "PR-AUC": np.nan, "F1": 0.676551, "Precision": 0.564308, "Recall": 0.844529, "Accuracy": 0.789918,},
    {"Protocolo": "Split 80/20 Cronológico", "Modelo": "Decision Tree",       "ROC-AUC": 0.900333, "PR-AUC": np.nan, "F1": 0.651835, "Precision": 0.714502, "Recall": 0.599274, "Accuracy": 0.833451,},
    {"Protocolo": "Split 80/20 Cronológico", "Modelo": "Random Forest",       "ROC-AUC": 0.908270, "PR-AUC": np.nan, "F1": 0.676912, "Precision": 0.537124, "Recall": 0.915059, "Accuracy": 0.772749,},

    {"Protocolo": "K-Fold (Média 5 Folds)", "Modelo": "LightGBM",            "ROC-AUC": 0.903254, "PR-AUC": np.nan, "F1": 0.675896, "Precision": 0.548209, "Recall": 0.881753, "Accuracy": 0.774290,},
    {"Protocolo": "K-Fold (Média 5 Folds)", "Modelo": "Logistic Regression", "ROC-AUC": 0.888257, "PR-AUC": np.nan, "F1": 0.652531, "Precision": 0.572338, "Recall": 0.762118, "Accuracy": 0.783766,},
    {"Protocolo": "K-Fold (Média 5 Folds)", "Modelo": "Decision Tree",       "ROC-AUC": 0.903234, "PR-AUC": np.nan, "F1": 0.681602, "Precision": 0.555639, "Recall": 0.881759, "Accuracy": 0.780388,},
    {"Protocolo": "K-Fold (Média 5 Folds)", "Modelo": "Random Forest",       "ROC-AUC": 0.908251, "PR-AUC": np.nan, "F1": 0.684456, "Precision": 0.556820, "Recall": 0.889285, "Accuracy": 0.780768,},

    {"Protocolo": "Holdout final (params Walk-forward)", "Modelo": "Random Forest",       "ROC-AUC": 0.909877, "PR-AUC": 0.818909, "F1": 0.668387, "Precision": 0.521392, "Recall": 0.930808, "Accuracy": 0.752700,},
    {"Protocolo": "Holdout final (params Walk-forward)", "Modelo": "LightGBM",            "ROC-AUC": 0.905172, "PR-AUC": 0.810032, "F1": 0.670691, "Precision": 0.524268, "Recall": 0.930598, "Accuracy": 0.755317,},
    {"Protocolo": "Holdout final (params Walk-forward)", "Modelo": "Decision Tree",       "ROC-AUC": 0.902120, "PR-AUC": 0.797864, "F1": 0.672025, "Precision": 0.528778, "Recall": 0.921719, "Accuracy": 0.759112,},
    {"Protocolo": "Holdout final (params Walk-forward)", "Modelo": "Logistic Regression", "ROC-AUC": 0.893813, "PR-AUC": 0.790383, "F1": 0.670349, "Precision": 0.544421, "Recall": 0.872064, "Accuracy": 0.770352,},

    {"Protocolo": "Holdout final (Time Series)", "Modelo": "Random Forest",       "ROC-AUC": 0.909877, "PR-AUC": 0.818909, "F1": 0.668387, "Precision": 0.521392, "Recall": 0.930808, "Accuracy": 0.752700,},
    {"Protocolo": "Holdout final (Time Series)", "Modelo": "LightGBM",            "ROC-AUC": 0.905172, "PR-AUC": 0.810032, "F1": 0.670691, "Precision": 0.524268, "Recall": 0.930598, "Accuracy": 0.755317,},
    {"Protocolo": "Holdout final (Time Series)", "Modelo": "Decision Tree",       "ROC-AUC": 0.902120, "PR-AUC": 0.797864, "F1": 0.672025, "Precision": 0.528778, "Recall": 0.921719, "Accuracy": 0.759112,},
    {"Protocolo": "Holdout final (Time Series)", "Modelo": "Logistic Regression", "ROC-AUC": 0.893813, "PR-AUC": 0.790383, "F1": 0.670349, "Precision": 0.544421, "Recall": 0.872064, "Accuracy": 0.770352,},
])

best_optimistic = resultados[resultados["Protocolo"] == "Split 80/20 Cronológico"].loc[lambda df: df["ROC-AUC"].idxmax()]
holdout_final = resultados[resultados["Protocolo"] == "Holdout final (params Walk-forward)"]
if holdout_final.empty:
    st.error("Nenhum resultado encontrado para `Holdout final (params Walk-forward)`.")
    st.stop()
best_realistic_auc = holdout_final.loc[holdout_final["ROC-AUC"].idxmax()]
best_realistic_f1 = holdout_final.loc[holdout_final["F1"].idxmax()]

st.header("📌 Sumário Executivo")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelos avaliados", "4", "LR · DT · RF · LGBM")
c2.metric("Blocos de análise", "5", "Cron · K-Fold · TSS · WF · Holdout")
c3.metric("Melhor AUC (split 80/20)", f"{best_optimistic['ROC-AUC']:.4f}", f"{best_optimistic['Modelo']}")
c4.metric("Melhor AUC (holdout final)", f"{best_realistic_auc['ROC-AUC']:.4f}", f"{best_realistic_auc['Modelo']}")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Tabela Geral",
    "📈 ROC-AUC por Protocolo",
    "🔥 Heatmap Modelo × Protocolo",
    "📉 Evolução por Rigor",
    "🎯 Trade-off Precision/Recall",
    "🏆 Veredito Final"
])

with tab1:
    st.subheader("Todos os resultados consolidados")
    st.dataframe(
        resultados.style.background_gradient(subset=["ROC-AUC","F1"], cmap="RdYlGn")
                        .format({"ROC-AUC":"{:.6f}","PR-AUC":"{:.6f}","F1":"{:.6f}","Precision":"{:.6f}","Recall":"{:.6f}","Accuracy":"{:.6f}"}),
        use_container_width=True, height=560
    )
    st.caption("Origem: `modelos_sem_serie_temporal.ipynb` e `07_treinamento_serie_temporal.ipynb`. Métricas ausentes no dataframe original ficam como NaN.")

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
                zmin=0.86, zmax=0.925)
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
    st.subheader("Como o AUC varia conforme mudamos o protocolo")
    rigor_map = {
        "Split 80/20 Cronológico": 1,
        "K-Fold (Média 5 Folds)": 2,
        "TimeSeriesSplit (validação CV)": 3,
        "Walk-forward (validação CV)": 4,
        "Holdout final (params Walk-forward)": 5,
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
    fig.update_layout(height=460, yaxis_range=[0.87, 0.925])
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(resultados, x="Protocolo", y="ROC-AUC", color="Modelo",
                   markers=True, category_orders={"Protocolo": [
                       "Split 80/20 Cronológico","K-Fold (Média 5 Folds)",
                       "TimeSeriesSplit (validação CV)","Walk-forward (validação CV)",
                       "Holdout final (params Walk-forward)"]})
    fig2.update_traces(line=dict(width=2.5))
    fig2.update_layout(height=460, yaxis_range=[0.86, 0.925],
                       title="Evolução por modelo")
    st.plotly_chart(fig2, use_container_width=True)

    st.info("**Leitura:** TSCV e Walk-forward são métricas médias de validação usadas para tuning. "
            "A comparação operacional final é o holdout final treinado com os melhores hiperparâmetros do walk-forward.")

with tab5:
    st.subheader("Trade-off Precision × Recall")
    tradeoff_df = resultados.dropna(subset=["Recall", "Precision", "F1"])
    fig = px.scatter(tradeoff_df, x="Recall", y="Precision",
                     color="Modelo", symbol="Protocolo", size="F1",
                     hover_data=["ROC-AUC","F1","Protocolo"],
                     size_max=22)
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Modelos no canto superior direito são os ideais. TSCV/WF de validação não aparecem aqui porque o notebook 07 não calcula Precision/Recall nesses dataframes.")

    st.subheader("Radar: melhor modelo por protocolo")
    best = tradeoff_df.loc[tradeoff_df.groupby("Protocolo")["ROC-AUC"].idxmax()]
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
        st.success("**Melhor AUC no holdout final**\n\n"
                   f"**{best_realistic_auc['Modelo']}** com params Walk-forward\n\n"
                   f"- ROC-AUC: **{best_realistic_auc['ROC-AUC']:.6f}**\n"
                   f"- PR-AUC: **{best_realistic_auc['PR-AUC']:.6f}**\n"
                   f"- F1: **{best_realistic_auc['F1']:.6f}**\n"
                   f"- Recall: **{best_realistic_auc['Recall']:.6f}**")
    with col2:
        st.info("**Melhor F1 no holdout final**\n\n"
                f"**{best_realistic_f1['Modelo']}** com params Walk-forward\n\n"
                f"- ROC-AUC: **{best_realistic_f1['ROC-AUC']:.6f}**\n"
                f"- PR-AUC: **{best_realistic_f1['PR-AUC']:.6f}**\n"
                f"- F1: **{best_realistic_f1['F1']:.6f}**\n"
                f"- Precision: **{best_realistic_f1['Precision']:.6f}**")

    st.markdown("---")
    st.subheader("Principais aprendizados")
    st.markdown("""
1. **Protocolo > Modelo.** A diferença entre protocolos é tão relevante quanto a diferença entre LR, DT, RF e LightGBM.
2. **Split 80/20 e K-Fold** vêm do notebook sem série temporal e são comparações menos realistas.
3. **TimeSeriesSplit e Walk-forward** no notebook 07 são usados para tuning por ROC-AUC médio.
4. **Holdout final** é a avaliação operacional: treino no pré-teste inteiro e teste no bloco temporal cego.
5. **Random Forest** teve maior ROC-AUC no holdout final; **Decision Tree** teve maior F1.
6. **PR-AUC** só existe no dataframe de holdout final do notebook 07.
    """)

    st.markdown("---")
    st.subheader("Recomendação de produção")
    st.info(f"Implantar **{best_realistic_auc['Modelo']}** como candidato principal por ROC-AUC no holdout final, "
            "com retraining diário em janela walk-forward (2 dias treino / 1 dia validação), "
            "purge de 15 minutos, monitorando degradação de AUC por mercado em dashboard contínuo. "
            f"Manter **{best_realistic_f1['Modelo']}** como alternativa se F1 for o critério operacional.")

st.markdown("---")
st.caption("Trabalho Final — Introdução ao Machine Learning · Consolidação dos notebooks modelos_sem_serie_temporal + 07")
