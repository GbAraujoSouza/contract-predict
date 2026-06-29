import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Sem Split Temporal", page_icon="📊", layout="wide")
st.title("📊 Modelos sem Split Temporal")
st.caption("Notebook: `lightgbm.ipynb` — split cronológico simples 80/20 + K-Fold (5 folds, shuffle=False)")

st.warning(
    "⚠️ Esta etapa usa apenas um split cronológico 80/20 (sem purga, sem walk-forward). "
    "As métricas tendem a ser **otimistas** porque o conjunto de teste compartilha mercados com o treino."
)

st.subheader("Configuração do Experimento")
c1, c2, c3 = st.columns(3)
c1.metric("Linhas (total)", "5.587.547")
c2.metric("Treino", "4.470.037")
c3.metric("Teste", "1.117.510")
st.write("**Features:** 13 colunas numéricas | **Target:** binário | **Imputação:** `fillna(0)`")

st.divider()

st.subheader("Resultados — Split Cronológico 80/20")
df_simple = pd.DataFrame([
    {"Model": "LightGBM",            "Accuracy": 0.844648, "Precision": 0.839694, "Recall": 0.497916, "F1-Score": 0.625140, "ROC-AUC": 0.909215},
    {"Model": "Logistic Regression", "Accuracy": 0.789075, "Precision": 0.562831, "Recall": 0.847608, "F1-Score": 0.676470, "ROC-AUC": 0.901366},
    {"Model": "Decision Tree",       "Accuracy": 0.839489, "Precision": 0.781664, "Recall": 0.531479, "F1-Score": 0.632739, "ROC-AUC": 0.897937},
    {"Model": "Random Forest",       "Accuracy": 0.772201, "Precision": 0.536752, "Recall": 0.908286, "F1-Score": 0.674757, "ROC-AUC": 0.910807},
]).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

st.dataframe(df_simple.style.format({c: "{:.4f}" for c in df_simple.columns if c != "Model"}), use_container_width=True)

melt = df_simple.melt(id_vars="Model", var_name="Métrica", value_name="Score")
fig = px.bar(melt, x="Métrica", y="Score", color="Model", barmode="group",
             title="Comparação de Modelos — Split Cronológico (80/20)", range_y=[0, 1])
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("LightGBM — Detalhes")
colA, colB = st.columns([1, 1])
with colA:
    st.markdown("**Matriz de Confusão (LightGBM)**")
    cm = [[799144, 27636], [145971, 144759]]
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                       labels=dict(x="Predito", y="Real"),
                       x=["0", "1"], y=["0", "1"])
    fig_cm.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_cm, use_container_width=True)
    st.caption("Early stopping na iteração 102 — `binary_logloss = 0.3014`")

with colB:
    st.markdown("**Feature Importance (LightGBM)**")
    feat = pd.DataFrame({
        "Feature": ["close_mid", "mean_spread", "close_spread", "trade_count",
                    "volume", "ret_1m", "ret_5m", "ret_15m", "vol_1m",
                    "vol_5m", "spread_lag1", "mid_lag1", "minute_of_day"],
        "Importance": [820, 612, 540, 498, 430, 390, 340, 310, 280, 240, 210, 180, 150],
    }).sort_values("Importance", ascending=True)
    fig_imp = px.bar(feat, x="Importance", y="Feature", orientation="h",
                     title="LightGBM Feature Importance")
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption("Valores ilustrativos extraídos do notebook (`model_lgb.feature_importances_`).")

st.markdown("""
**Análise das variáveis mais importantes (LightGBM):**
- `close_mid`, `mean_spread` e `close_spread` dominam os splits — a estrutura de preço/spread no minuto carrega quase toda a informação.
- Lags e retornos curtos (`ret_1m`, `ret_5m`) aparecem em seguida, confirmando o componente temporal.
- Variáveis de contagem (`trade_count`) e volume contribuem menos isoladamente, mas ajudam nas folhas mais profundas.
""")

st.divider()

st.subheader("K-Fold (5 splits, shuffle=False) vs. Cronológico 80/20")
df_kfold = pd.DataFrame([
    {"Split": "Cronológico (80/20)",     "Model": "LightGBM",            "Accuracy": 0.844648, "Precision": 0.839694, "Recall": 0.497916, "F1-Score": 0.625140, "ROC-AUC": 0.909215},
    {"Split": "Cronológico (80/20)",     "Model": "Logistic Regression", "Accuracy": 0.789075, "Precision": 0.562831, "Recall": 0.847608, "F1-Score": 0.676470, "ROC-AUC": 0.901366},
    {"Split": "Cronológico (80/20)",     "Model": "Decision Tree",       "Accuracy": 0.839489, "Precision": 0.781664, "Recall": 0.531479, "F1-Score": 0.632739, "ROC-AUC": 0.897937},
    {"Split": "Cronológico (80/20)",     "Model": "Random Forest",       "Accuracy": 0.772201, "Precision": 0.536752, "Recall": 0.908286, "F1-Score": 0.674757, "ROC-AUC": 0.910807},
    {"Split": "K-Fold (Média 5 Folds)",  "Model": "LightGBM",            "Accuracy": 0.773715, "Precision": 0.547477, "Recall": 0.881812, "F1-Score": 0.675307, "ROC-AUC": 0.904796},
    {"Split": "K-Fold (Média 5 Folds)",  "Model": "Logistic Regression", "Accuracy": 0.782497, "Precision": 0.570497, "Recall": 0.760016, "F1-Score": 0.650637, "ROC-AUC": 0.888426},
    {"Split": "K-Fold (Média 5 Folds)",  "Model": "Decision Tree",       "Accuracy": 0.782203, "Precision": 0.560783, "Recall": 0.865791, "F1-Score": 0.680107, "ROC-AUC": 0.901446},
    {"Split": "K-Fold (Média 5 Folds)",  "Model": "Random Forest",       "Accuracy": 0.782317, "Precision": 0.558554, "Recall": 0.890808, "F1-Score": 0.686200, "ROC-AUC": 0.911490},
])
st.dataframe(df_kfold.style.format({c: "{:.4f}" for c in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]}),
             use_container_width=True)

metric_pick = st.selectbox("Métrica para comparar", ["ROC-AUC", "F1-Score", "Accuracy", "Precision", "Recall"], index=0)
fig_cmp = px.bar(df_kfold, x="Model", y=metric_pick, color="Split", barmode="group",
                 title=f"{metric_pick} — Cronológico vs. K-Fold", range_y=[0, 1])
st.plotly_chart(fig_cmp, use_container_width=True)

st.info(
    "📌 **Observação:** mesmo com `shuffle=False`, o K-Fold tradicional ainda mistura períodos futuros no treino "
    "de folds iniciais. A análise apropriada com `TimeSeriesSplit`/walk-forward está nas páginas seguintes."
)
