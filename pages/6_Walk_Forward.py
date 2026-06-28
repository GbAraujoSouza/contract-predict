import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Walk-Forward", page_icon="🚀", layout="wide")
st.title("🚀 Protocolo rigoroso — Walk-forward com purge gap")

st.markdown(
    """
    Versão mais rigorosa do treinamento, focada em **reproduzir condições reais de operação**.

    ### Fluxo
    1. Ordenação temporal global do dataset
    2. **Holdout final cego** (10–11/03) reservado só para avaliação final
    3. **Tuning de hiperparâmetros por ROC-AUC** no período pré-teste, com dois protocolos:
       - `TimeSeriesSplit` expansivo
       - `Walk-forward` com janela fixa (2 dias treino + 1 dia validação)
    4. **Purge gap de 15 min** entre treino e validação para evitar leakage pelo horizonte do alvo
    5. Avaliação final no holdout
    """
)

st.header("Esquema visual dos folds")

fig = go.Figure()
days = ["06/03", "07/03", "08/03", "09/03", "10/03", "11/03"]

def add_block(y, x0, x1, color, label):
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=y - 0.4, y1=y + 0.4,
                  fillcolor=color, line=dict(color="white", width=2))
    fig.add_annotation(x=(x0 + x1) / 2, y=y, text=label,
                       showarrow=False, font=dict(color="white", size=11))

add_block(4, 0, 2, "#1f77b4", "Treino")
add_block(4, 2.05, 3, "#ff7f0e", "Val")
add_block(3, 1, 3, "#1f77b4", "Treino")
add_block(3, 3.05, 4, "#ff7f0e", "Val")
add_block(2, 2, 4, "#1f77b4", "Treino")
add_block(2, 4.05, 5, "#ff7f0e", "Val")

add_block(0, 0, 4, "#1f77b4", "Treino completo (pré-teste)")
add_block(0, 4.05, 6, "#d62728", "Holdout final (cego)")

fig.update_layout(
    xaxis=dict(tickmode="array", tickvals=list(range(7)), ticktext=days + [""],
               title="Dia"),
    yaxis=dict(tickmode="array", tickvals=[0, 2, 3, 4],
               ticktext=["Treino final", "Fold 3", "Fold 2", "Fold 1"],
               title=""),
    height=400, showlegend=False,
    title="Walk-forward: janela de treino fixa (2 dias) + validação (1 dia), avançando no tempo",
)
st.plotly_chart(fig, use_container_width=True)

st.header("Por que `purge gap`?")
st.markdown(
    """
    O alvo (`target`) olha **15 minutos à frente**. Sem gap, as últimas linhas do treino têm
    rótulos calculados com preços que já caem dentro do bloco de validação — vazamento direto pelo target.

    O `purge gap = 15 min` remove essas linhas-fronteira em todas as divisões.
    """
)

st.header("Grade de hiperparâmetros tunados por ROC-AUC")
st.code(
    """
LightGBM:
    num_leaves        ∈ {31, 63, 127}
    learning_rate     ∈ {0.05, 0.1}
    n_estimators      ∈ {200, 500}
    min_child_samples ∈ {20, 50}

Decision Tree:
    max_depth         ∈ {6, 10, 15, None}
    min_samples_leaf  ∈ {50, 200, 1000}

Random Forest:
    n_estimators      ∈ {200, 500}
    max_depth         ∈ {10, 20, None}
    min_samples_leaf  ∈ {50, 200}
    """,
    language="text",
)

st.header("Interpretação de ROC-AUC")
st.markdown(
    """
    - `y_proba`: probabilidade estimada de classe 1 para cada linha
    - `y_pred`: classe final após aplicar threshold sobre `y_proba`
    - **Curva ROC**: construída variando o threshold, não o hiperparâmetro
       - `TPR` = taxa de verdadeiros positivos (recall)
       - `FPR` = taxa de falsos positivos
    - **AUC**: área sob a curva ROC — mede quão bem o modelo ranqueia positivos acima de negativos
       - `AUC = 0,5`: aleatório
       - `AUC = 1,0`: separação perfeita
    - Hiperparâmetros mudam o modelo treinado e, portanto, `y_proba`. A curva ROC nasce
      da variação de threshold sobre esses scores.
    """
)

st.success(
    "Walk-forward é o método temporal mais próximo de operação real. Os melhores hiperparâmetros "
    "selecionados por ele são usados para treinar no período pré-teste completo e avaliar no holdout final."
)
