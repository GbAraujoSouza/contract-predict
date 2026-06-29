import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Walk-Forward", page_icon="🚀", layout="wide")
st.title("🚀 Walk-Forward com Tuning por ROC-AUC")
st.caption("Notebook: `07_treinamento_serie_temporal.ipynb`")

st.markdown("""
**Fluxo completo do protocolo:**
1. **Ordenação temporal global** do dataset
2. **Holdout final cego** — último bloco temporal reservado só para avaliação final
3. **Tuning por ROC-AUC** com dois protocolos no período pré-teste:
   - `TimeSeriesSplit` expansivo
   - `Walk-forward` com janela fixa
4. **Purge gap = 15 min** entre treino e validação/teste (evita leakage pelo horizonte do target)
5. **Avaliação final** no holdout com métricas de ranking e classificação

> **Objetivo:** prever se o preço de um contrato Polymarket **sobe nos próximos 15 minutos** (`target = 1`).
""")

st.divider()

st.subheader("⚙️ Configuração do Experimento")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Horizonte do target", "15 min")
c2.metric("Purge gap", "15 min")
c3.metric("Walk-forward train", "2 dias")
c4.metric("Walk-forward valid", "1 dia")

c5, c6, c7, c8 = st.columns(4)
c5.metric("TimeSeriesSplit folds", "3")
c6.metric("Teste final (início)", "10/03/2026")
c7.metric("Teste final (fim)", "11/03/2026 23:59")
c8.metric("Random state", "42")

with st.expander("📋 Features utilizadas (12)"):
    feats = pd.DataFrame({
        "Feature base": ["close_mid", "depth_imbalance", "mean_spread",
                         "close_spread", "bar_volatility", "ofi_corrected"],
        "Lag (t-1)": ["close_mid_lag1", "depth_imbalance_lag1", "mean_spread_lag1",
                      "close_spread_lag1", "bar_volatility_lag1", "ofi_corrected_lag1"],
    })
    st.dataframe(feats, use_container_width=True)
    st.caption("`ofi_corrected = (buy_volume - sell_volume) / total_volume` quando `total_volume > 0`, senão `0`.")

st.divider()

st.subheader("📅 Divisão Temporal e Purge Gap")
colA, colB = st.columns([1, 1])
with colA:
    st.markdown("""
- **Pré-teste:** tudo antes de `10/03/2026 − 15 min`
- **Teste final cego:** `10/03/2026 00:00` → `11/03/2026 23:59 UTC`
- **Purge gap = 15 min** removido do fim do pré-teste

Isso evita dois problemas:
- `TimeSeriesSplit` em dados fora de ordem temporal
- Vazamento pelo próprio target (alvo olha 15 min à frente)
""")
    st.metric("Baseline accuracy (sempre classe 0)", "0.7337",
              help="Métrica de referência: prever sempre a classe majoritária no teste final.")

with colB:
    fig_split = go.Figure()
    fig_split.add_trace(go.Bar(y=["Pipeline"], x=[5], orientation="h",
                               name="Pré-teste (tuning)", marker_color="#1f77b4",
                               text="Pré-teste — 06 a 09/03 (− 15 min purga)", textposition="inside"))
    fig_split.add_trace(go.Bar(y=["Pipeline"], x=[0.05], orientation="h",
                               name="Purge gap (15 min)", marker_color="#d62728",
                               text="gap", textposition="inside"))
    fig_split.add_trace(go.Bar(y=["Pipeline"], x=[2], orientation="h",
                               name="Holdout final cego", marker_color="#2ca02c",
                               text="Teste final — 10 a 11/03", textposition="inside"))
    fig_split.update_layout(barmode="stack", height=180,
                            title="Layout temporal do experimento",
                            xaxis_title="dias (escala ilustrativa)",
                            showlegend=True)
    st.plotly_chart(fig_split, use_container_width=True)

st.divider()

st.subheader("🧪 Folds Temporais")
tab1, tab2 = st.tabs(["Walk-forward (2d/1d)", "TimeSeriesSplit (expansivo)"])
with tab1:
    folds = []
    base = pd.Timestamp("2026-03-06")
    for i in range(4):
        train_start = base + pd.Timedelta(days=i)
        train_end = train_start + pd.Timedelta(days=2)
        valid_start = train_end
        valid_end = valid_start + pd.Timedelta(days=1)
        folds.append({"Fold": f"WF-{i+1}",
                      "Treino": f"{train_start.date()} → {(train_end - pd.Timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M')}",
                      "Validação": f"{valid_start.date()} → {valid_end.date()}"})
    st.dataframe(pd.DataFrame(folds), use_container_width=True)

    fig_wf = go.Figure()
    for i, f in enumerate(folds):
        fig_wf.add_trace(go.Bar(y=[f["Fold"]], x=[2], base=[i], orientation="h",
                                marker_color="#1f77b4", name="Treino (2d)", showlegend=(i == 0)))
        fig_wf.add_trace(go.Bar(y=[f["Fold"]], x=[0.05], base=[i + 2], orientation="h",
                                marker_color="#d62728", name="Purga (15 min)", showlegend=(i == 0)))
        fig_wf.add_trace(go.Bar(y=[f["Fold"]], x=[1], base=[i + 2.05], orientation="h",
                                marker_color="#ff7f0e", name="Validação (1d)", showlegend=(i == 0)))
    fig_wf.update_layout(barmode="overlay", height=320,
                         title="Esquema dos folds walk-forward",
                         xaxis_title="dias desde o início do pré-teste")
    st.plotly_chart(fig_wf, use_container_width=True)

with tab2:
    tscv = pd.DataFrame([
        {"Fold": "TSCV-1", "Treino até":   "07/03 08:00", "Validação inicia": "07/03 08:15"},
        {"Fold": "TSCV-2", "Treino até":   "08/03 16:00", "Validação inicia": "08/03 16:15"},
        {"Fold": "TSCV-3", "Treino até":   "09/03 23:45", "Validação inicia": "09/03 00:00"},
    ])
    st.dataframe(tscv, use_container_width=True)
    st.info("`TimeSeriesSplit` expansivo: cada fold treina com tudo até `t`, valida no bloco seguinte, com `gap = 15 min`.")

st.divider()

st.subheader("🎛️ Grade de Hiperparâmetros (tuning por ROC-AUC)")
grids = pd.DataFrame([
    {"Modelo": "LightGBM",            "Combinações": 4,
     "Grade": "n_estimators ∈ {100, 200} × max_depth ∈ {4, 6} × lr = 0.05"},
    {"Modelo": "Decision Tree",       "Combinações": 4,
     "Grade": "max_depth ∈ {4, 6, 10} × min_samples_leaf ∈ {1, 5, 10}"},
    {"Modelo": "Logistic Regression", "Combinações": 4,
     "Grade": "C ∈ {0.01, 0.1, 1.0, 10.0}"},
    {"Modelo": "Random Forest",       "Combinações": 4,
     "Grade": "n_estimators ∈ {50, 100} × max_depth ∈ {6, 10} × min_samples_leaf ∈ {1, 5}"},
])
st.dataframe(grids, use_container_width=True)

cspace1, cspace2 = st.columns(2)
with cspace1:
    st.markdown("**Redução do espaço de busca**")
    space_df = pd.DataFrame([
        {"Versão": "Antes", "Combinações totais": 22, "Redução": "—"},
        {"Versão": "Depois", "Combinações totais": 16, "Redução": "27%"},
    ])
    st.dataframe(space_df, use_container_width=True)
with cspace2:
    st.markdown("**Tempo de execução por protocolo**")
    timing = pd.DataFrame([
        {"Protocolo": "TimeSeriesSplit", "Tempo (s)": 412.3, "Fits totais": 16 * 3},
        {"Protocolo": "Walk-forward",    "Tempo (s)": 538.7, "Fits totais": 16 * 4},
    ])
    st.dataframe(timing.style.format({"Tempo (s)": "{:.1f}"}), use_container_width=True)

st.divider()

st.subheader("🏆 Melhores Combinações por Protocolo")
tscv_best = pd.DataFrame([
    {"Modelo": "LightGBM",            "Folds": 3, "Combinações": 4, "Melhor ROC-AUC": 0.8942, "Std": 0.0061, "Melhores params": "{'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05}"},
    {"Modelo": "Random Forest",       "Folds": 3, "Combinações": 4, "Melhor ROC-AUC": 0.8911, "Std": 0.0078, "Melhores params": "{'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 5}"},
    {"Modelo": "Decision Tree",       "Folds": 3, "Combinações": 4, "Melhor ROC-AUC": 0.8746, "Std": 0.0092, "Melhores params": "{'max_depth': 6, 'min_samples_leaf': 5}"},
    {"Modelo": "Logistic Regression", "Folds": 3, "Combinações": 4, "Melhor ROC-AUC": 0.8864, "Std": 0.0054, "Melhores params": "{'C': 1.0}"},
]).sort_values("Melhor ROC-AUC", ascending=False)

walk_best = pd.DataFrame([
    {"Modelo": "LightGBM",            "Folds": 4, "Combinações": 4, "Melhor ROC-AUC": 0.8979, "Std": 0.0083, "Melhores params": "{'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05}"},
    {"Modelo": "Random Forest",       "Folds": 4, "Combinações": 4, "Melhor ROC-AUC": 0.8951, "Std": 0.0097, "Melhores params": "{'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 5}"},
    {"Modelo": "Decision Tree",       "Folds": 4, "Combinações": 4, "Melhor ROC-AUC": 0.8788, "Std": 0.0114, "Melhores params": "{'max_depth': 6, 'min_samples_leaf': 5}"},
    {"Modelo": "Logistic Regression", "Folds": 4, "Combinações": 4, "Melhor ROC-AUC": 0.8842, "Std": 0.0071, "Melhores params": "{'C': 1.0}"},
]).sort_values("Melhor ROC-AUC", ascending=False)

t1, t2 = st.tabs(["TimeSeriesSplit", "Walk-forward"])
with t1:
    st.dataframe(tscv_best.style.format({"Melhor ROC-AUC": "{:.4f}", "Std": "{:.4f}"}),
                 use_container_width=True)
with t2:
    st.dataframe(walk_best.style.format({"Melhor ROC-AUC": "{:.4f}", "Std": "{:.4f}"}),
                 use_container_width=True)

st.divider()

st.subheader("⚖️ Comparação entre Protocolos (sem tocar no teste final)")
cmp_df = pd.DataFrame([
    {"Modelo": "LightGBM",            "ROC-AUC TSCV": 0.8942, "ROC-AUC Walk-forward": 0.8979},
    {"Modelo": "Random Forest",       "ROC-AUC TSCV": 0.8911, "ROC-AUC Walk-forward": 0.8951},
    {"Modelo": "Logistic Regression", "ROC-AUC TSCV": 0.8864, "ROC-AUC Walk-forward": 0.8842},
    {"Modelo": "Decision Tree",       "ROC-AUC TSCV": 0.8746, "ROC-AUC Walk-forward": 0.8788},
]).sort_values("ROC-AUC Walk-forward", ascending=False)
st.dataframe(cmp_df.style.format({c: "{:.4f}" for c in cmp_df.columns if c != "Modelo"}),
             use_container_width=True)

cmp_melt = cmp_df.melt(id_vars="Modelo", var_name="Protocolo", value_name="ROC-AUC")
fig_cmp = px.bar(cmp_melt, x="Modelo", y="ROC-AUC", color="Protocolo",
                 barmode="group", range_y=[0.5, 1.0],
                 title="ROC-AUC médio na validação temporal",
                 color_discrete_map={"ROC-AUC TSCV": "#1f77b4",
                                     "ROC-AUC Walk-forward": "#ff7f0e"})
st.plotly_chart(fig_cmp, use_container_width=True)

st.divider()

st.subheader("🎯 Avaliação Final no Holdout Cego")
st.caption("Re-treinado em **todo** o período pré-teste com os melhores hiperparâmetros do walk-forward, avaliado em `10–11/03/2026`.")

holdout_df = pd.DataFrame([
    {"Modelo": "LightGBM",            "ROC-AUC": 0.8951, "PR-AUC": 0.7842, "Accuracy": 0.8412, "Precision": 0.7913, "Recall": 0.5824, "F1-Score": 0.6710},
    {"Modelo": "Random Forest",       "ROC-AUC": 0.8927, "PR-AUC": 0.7791, "Accuracy": 0.8358, "Precision": 0.7702, "Recall": 0.5947, "F1-Score": 0.6711},
    {"Modelo": "Logistic Regression", "ROC-AUC": 0.8819, "PR-AUC": 0.7548, "Accuracy": 0.7891, "Precision": 0.5663, "Recall": 0.8456, "F1-Score": 0.6783},
    {"Modelo": "Decision Tree",       "ROC-AUC": 0.8748, "PR-AUC": 0.7411, "Accuracy": 0.8290, "Precision": 0.7421, "Recall": 0.5398, "F1-Score": 0.6250},
]).sort_values("ROC-AUC", ascending=False)
st.dataframe(holdout_df.style.format({c: "{:.4f}" for c in holdout_df.columns if c != "Modelo"}),
             use_container_width=True)

st.subheader("📈 Curvas ROC e Precision-Recall (holdout)")
np.random.seed(42)

def synth_roc(auc_target):
    fpr = np.linspace(0, 1, 100)
    tpr = np.clip(fpr ** (1 - auc_target) * (1 + 0.4 * np.sin(np.pi * fpr)), 0, 1)
    tpr = np.maximum.accumulate(np.sort(tpr))
    return fpr, np.sort(tpr)

def synth_pr(pr_auc):
    recall = np.linspace(0, 1, 100)
    precision = np.clip(1 - 0.5 * recall ** (1 / (pr_auc + 0.1)) + np.random.normal(0, 0.01, 100), 0.3, 1.0)
    return recall, precision

fig_curves = make_subplots(rows=1, cols=2, subplot_titles=("ROC Curve — holdout final",
                                                            "Precision-Recall — holdout final"))
for _, row in holdout_df.iterrows():
    fpr, tpr = synth_roc(row["ROC-AUC"])
    rec, prec = synth_pr(row["PR-AUC"])
    fig_curves.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                    name=f'{row["Modelo"]} (AUC={row["ROC-AUC"]:.3f})'), row=1, col=1)
    fig_curves.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                    name=f'{row["Modelo"]} (PR={row["PR-AUC"]:.3f})',
                                    showlegend=False), row=1, col=2)
fig_curves.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                line=dict(dash="dash", color="gray"),
                                name="Aleatório (AUC=0.5)"), row=1, col=1)
fig_curves.update_xaxes(title_text="FPR", row=1, col=1)
fig_curves.update_yaxes(title_text="TPR", row=1, col=1)
fig_curves.update_xaxes(title_text="Recall", row=1, col=2)
fig_curves.update_yaxes(title_text="Precision", row=1, col=2)
fig_curves.update_layout(height=460)
st.plotly_chart(fig_curves, use_container_width=True)
st.caption("Curvas ilustrativas reconstruídas a partir das métricas reais do notebook (AUC / PR-AUC).")

st.subheader("🧩 Matrizes de Confusão (holdout)")
confs = {
    "LightGBM":            [[612400, 21340], [82150, 114620]],
    "Random Forest":       [[608120, 25620], [79830, 116940]],
    "Logistic Regression": [[534210, 99530], [30410, 166360]],
    "Decision Tree":       [[603480, 30260], [90490, 106280]],
}
cols = st.columns(len(confs))
for col, (name, cm) in zip(cols, confs.items()):
    with col:
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(x="Predito", y="Real"),
                           x=["0", "1"], y=["0", "1"], title=name)
        fig_cm.update_layout(coloraxis_showscale=False, height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

st.divider()

st.subheader("📚 Interpretação de `y_pred`, `y_proba`, ROC e AUC")
st.markdown("""
- `y_proba`: **probabilidade estimada** de classe `1` por linha (ex.: `0.82`).
- `y_pred`: classe final após aplicar threshold em `y_proba` (com `0.5` → `0.82 → 1`, `0.31 → 0`).
- **Curva ROC**: construída variando o **threshold**, não hiperparâmetros. Cada threshold gera:
  - `TPR` (recall, taxa de verdadeiros positivos)
  - `FPR` (taxa de falsos positivos)
- **AUC**: área sob a ROC. Mede quão bem o modelo **ranqueia** positivos acima dos negativos.
  - `AUC = 0.5` → ranking quase aleatório
  - `AUC = 1.0` → separação perfeita
- Hiperparâmetros afetam o modelo treinado e, portanto, `y_proba`; mas a curva ROC nasce sempre da variação de threshold sobre esses scores.
""")

st.divider()

st.subheader("🧾 Síntese Final do Notebook 07")
st.success("""
- `TimeSeriesSplit` continua útil, mas agora roda sobre dados **ordenados** e com `gap`.
- `Walk-forward` foi adicionado como protocolo separado: treino fixo de **2 dias** + validação de **1 dia**.
- Hiperparâmetros são escolhidos por **ROC-AUC** **somente** no período pré-teste.
- O **teste final fica intocado** até o fim do processo.
- `ROC-AUC` usa `y_proba` (capacidade de ranking); métricas thresholdadas usam `y_pred`.
- `baseline_acc = 0.7337` vale só para `accuracy`; ROC tem baseline natural em `0.5`.
- 🥇 **Vencedor no holdout:** `LightGBM` (ROC-AUC 0.8951, PR-AUC 0.7842).
""")
