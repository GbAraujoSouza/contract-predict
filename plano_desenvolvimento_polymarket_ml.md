# Plano de Desenvolvimento — Projeto de Machine Learning com Polymarket

## 1. Visão Geral

### Nome do projeto

**Previsão de Direção de Preços em Mercados de Previsão usando Desequilíbrio do Livro de Ofertas**

### Objetivo geral

Desenvolver uma aplicação completa de Machine Learning capaz de prever movimentos de curto prazo nos preços de contratos binários negociados no Polymarket, utilizando dados históricos do livro de ofertas disponíveis no dataset público do Kaggle.

O sistema deve contemplar:

- Pipeline de tratamento de dados;
- Engenharia de features;
- Treinamento de modelos supervisionados;
- Avaliação dos modelos;
- API para exposição dos resultados;
- Frontend com dashboards interativos.

### Stack definida

| Camada                     | Tecnologia                                             |
| -------------------------- | ------------------------------------------------------ |
| Dataset                    | Kaggle — Polymarket Tick-Level Orderbook Dataset       |
| Processamento de dados     | Python + Pandas                                        |
| Modelagem ML               | Scikit-learn, opcionalmente XGBoost/LightGBM           |
| API                        | FastAPI                                                |
| Frontend                   | Angular                                                |
| Visualizações              | Angular + Chart.js, ECharts ou ngx-charts              |
| Persistência de resultados | `.csv`, `.parquet`, `.json`, `.joblib`                 |
| Organização                | Monorepo com `backend/`, `frontend/`, `data/`, `docs/` |

---

# 2. Descrição do Projeto

## 2.1 Resumo

Este projeto tem como objetivo desenvolver um sistema de aprendizado de máquina para prever movimentos de curto prazo nos preços de contratos binários negociados no Polymarket, uma plataforma de mercados de previsão. Em mercados desse tipo, os usuários negociam ações associadas a possíveis resultados de eventos futuros, como “SIM” ou “NÃO”. O preço dessas ações varia entre 0 e 1 dólar e pode ser interpretado como uma estimativa coletiva da probabilidade de ocorrência de determinado evento.

A proposta do projeto é investigar se sinais presentes no livro de ofertas, como spread, volume, liquidez, profundidade de compra e venda, volatilidade recente e desequilíbrio entre ordens de compra e venda, podem ser utilizados para prever se o preço de um contrato tende a subir nos próximos 15 minutos. Para isso, será utilizado um dataset público do Kaggle contendo dados tick-level e informações do livro de ofertas do Polymarket.

O sistema será composto por uma pipeline de Machine Learning desenvolvida em Python com Pandas, responsável por limpeza dos dados, análise exploratória, engenharia de features, criação dos rótulos, treinamento, validação e avaliação dos modelos. Serão utilizados pelo menos dois métodos supervisionados, como Regressão Logística, Árvore de Decisão e Random Forest. Os resultados serão disponibilizados por uma API construída com FastAPI e visualizados em um dashboard desenvolvido em Angular.

O projeto busca não apenas comparar o desempenho de diferentes modelos, mas também interpretar quais variáveis mais influenciam as previsões, discutir limitações do uso de aprendizado de máquina em mercados financeiros e refletir sobre questões éticas relacionadas ao uso de sistemas preditivos em ambientes de negociação.

---

## 2.2 Introdução detalhada para a página `/home` ou `/info`

Mercados de previsão são ambientes nos quais participantes negociam contratos associados à ocorrência de eventos futuros. Diferentemente de uma bolsa tradicional, onde são negociadas ações de empresas, commodities ou moedas, em um mercado de previsão os ativos representam possíveis respostas para uma pergunta. Por exemplo, um contrato pode perguntar se determinado candidato vencerá uma eleição, se uma equipe vencerá uma partida ou se determinado indicador econômico atingirá um valor específico até uma data definida.

No Polymarket, muitos desses contratos possuem resolução binária, geralmente representada pelas opções “SIM” e “NÃO”. Cada uma dessas opções pode ser negociada por preços entre 0 e 1 dólar. De maneira simplificada, o preço de uma ação pode ser interpretado como a probabilidade atribuída pelo mercado àquele resultado. Por exemplo, se uma ação “SIM” está sendo negociada a 0,65 dólar, isso indica que o mercado está precificando aquele evento como tendo aproximadamente 65% de chance de ocorrer. Caso o evento seja resolvido como verdadeiro, a ação correspondente paga 1 dólar; caso contrário, ela perde seu valor.

Essa dinâmica cria um ambiente interessante para análise com aprendizado de máquina, pois os preços dos contratos mudam ao longo do tempo conforme novas informações surgem e conforme compradores e vendedores ajustam suas expectativas. Além do preço em si, o mercado também possui uma estrutura conhecida como livro de ofertas, ou order book. O livro de ofertas registra ordens de compra e venda abertas em diferentes níveis de preço. A partir dele, é possível observar não apenas o preço atual do contrato, mas também a pressão compradora, a pressão vendedora, a liquidez disponível, o spread entre compra e venda e o desequilíbrio entre os dois lados do mercado.

O problema investigado neste projeto é se esses sinais de microestrutura do mercado podem ajudar a prever movimentos de preço em curto prazo. Mais especificamente, o objetivo é construir modelos supervisionados capazes de estimar se o preço de um contrato tende a subir nos próximos 15 minutos, utilizando informações históricas extraídas do livro de ofertas e do comportamento recente do mercado. Essa abordagem não busca prever diretamente o resultado final de um evento, mas sim analisar a dinâmica de curto prazo do preço negociado no mercado.

Para desenvolver essa solução, será utilizado um dataset público do Kaggle contendo dados tick-level e registros do livro de ofertas do Polymarket. A partir desses dados, será construída uma pipeline de aprendizado de máquina responsável por limpar os dados, selecionar mercados com qualidade suficiente, criar variáveis explicativas, gerar os rótulos de classificação, treinar diferentes modelos e avaliar seus resultados. Entre as variáveis analisadas estarão preço médio, spread, volume negociado, volatilidade recente, retornos passados, profundidade do livro de ofertas e indicadores de desequilíbrio entre ordens de compra e venda.

Do ponto de vista técnico, o projeto será estruturado como uma aplicação completa. A etapa de processamento e modelagem será desenvolvida em Python, utilizando Pandas para manipulação dos dados e bibliotecas de aprendizado de máquina para treinamento e avaliação dos modelos. A API será construída com FastAPI, permitindo disponibilizar métricas, previsões, dados agregados e resultados dos modelos. A interface será desenvolvida em Angular, oferecendo dashboards interativos para visualização dos contratos, comparação dos modelos, análise das métricas e exploração das previsões realizadas.

Além da implementação técnica, o projeto também considera aspectos de interpretação e responsabilidade. Modelos aplicados a mercados financeiros ou mercados de previsão podem produzir resultados úteis para análise, mas também possuem limitações importantes. Dados de mercado são ruidosos, podem sofrer influência de baixa liquidez, mudanças repentinas de informação, comportamento especulativo e padrões que não se repetem no futuro. Por isso, o sistema será apresentado como uma ferramenta acadêmica de análise e visualização, não como uma recomendação financeira ou mecanismo de tomada de decisão para negociação real.

Dessa forma, o projeto integra conceitos centrais de aprendizado de máquina, como pré-processamento de dados, engenharia de atributos, treinamento de modelos supervisionados, validação temporal, avaliação por métricas apropriadas e interpretação dos resultados. Ao final, espera-se compreender em que medida informações do livro de ofertas contribuem para prever movimentos de curto prazo em contratos de mercados de previsão e quais são as limitações práticas e éticas desse tipo de abordagem.

---

# 3. Especificação Principal

## 3.1 Problema

Prever se o preço de um contrato binário do Polymarket irá subir nos próximos 15 minutos com base em sinais extraídos do livro de ofertas e do histórico recente do contrato.

## 3.2 Hipótese

O desequilíbrio entre ordens de compra e venda, combinado com informações de spread, volume, volatilidade e retornos recentes, contém sinais úteis para antecipar movimentos de curto prazo no preço dos contratos.

## 3.3 Tipo de aprendizado

- Aprendizado supervisionado;
- Classificação binária;
- Possível extensão para classificação multiclasse.

## 3.4 Entrada do modelo

As entradas devem ser features extraídas dos dados históricos, como:

- preço atual;
- melhor bid;
- melhor ask;
- spread;
- volume;
- liquidez;
- profundidade do livro de ofertas;
- desequilíbrio entre ordens de compra e venda;
- volatilidade recente;
- retornos passados.

## 3.5 Saída do modelo

Primeira versão:

```txt
1 = preço sobe nos próximos 15 minutos
0 = preço não sobe nos próximos 15 minutos
```

Versão posterior opcional:

```txt
1 = sobe
0 = estável
-1 = cai
```

## 3.6 Modelos mínimos

O projeto deve implementar pelo menos dois métodos supervisionados. Sugestão:

1. Regressão Logística;
2. Árvore de Decisão;
3. Random Forest.

Opcionalmente:

4. XGBoost ou LightGBM.

## 3.7 Métricas

As métricas principais devem incluir:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- ROC-AUC;
- Matriz de confusão;
- Baseline da classe majoritária.

## 3.8 Validação

A validação deve respeitar a natureza temporal dos dados.

Estratégia recomendada:

```txt
Treino: primeiros 70% dos dados no tempo
Validação: próximos 15%
Teste: últimos 15%
```

Evitar split aleatório simples para reduzir risco de vazamento temporal.

---

# 4. Estrutura Inicial do Repositório

```txt
polymarket-ml-project/
  README.md

  docs/
    specification.md
    dataset.md
    api-contract.md
    ml-pipeline.md
    frontend-spec.md
    ethics.md

  data/
    raw/
    interim/
    processed/
    outputs/

  backend/
    app/
      main.py
      routes/
        health.py
        info.py
        markets.py
        metrics.py
        predictions.py
      schemas/
        info_schema.py
        market_schema.py
        metrics_schema.py
        prediction_schema.py
      services/
        info_service.py
        market_service.py
        metrics_service.py
        prediction_service.py
      core/
        config.py

    pipelines/
      01_clean_data.py
      02_exploratory_analysis.py
      03_feature_engineering.py
      04_create_targets.py
      05_train_models.py
      06_evaluate_models.py
      07_export_results.py

    models/
      logistic_regression.joblib
      decision_tree.joblib
      random_forest.joblib

    outputs/
      metrics.json
      confusion_matrices.json
      feature_importance.json
      market_summary.json
      predictions_sample.json

    requirements.txt

  frontend/
    src/
      app/
        pages/
          home/
          dashboard/
          markets/
          model-comparison/
          prediction-explorer/
          ethics-limitations/
        services/
          api.service.ts
          market.service.ts
          metrics.service.ts
          prediction.service.ts
        models/
          market.model.ts
          metrics.model.ts
          prediction.model.ts
```

---

# 5. Plano 1 — Estrutura Inicial do Projeto

## 5.1 Objetivo

Criar a base do projeto com:

- backend FastAPI funcionando;
- frontend Angular funcionando;
- comunicação básica entre frontend e backend;
- rotas de teste;
- página `/home` ou `/info` com descrição, objetivo e escopo do projeto.

Essa fase não deve implementar Machine Learning ainda. O foco é validar a arquitetura.

---

## 5.2 Especificação funcional

O sistema deve permitir que um usuário acesse uma página inicial contendo:

- nome do projeto;
- descrição geral;
- problema estudado;
- explicação breve do que é o Polymarket;
- explicação do que são contratos binários;
- objetivo do modelo;
- tecnologias utilizadas;
- etapas previstas do projeto.

O frontend deve consumir pelo menos uma rota da API para buscar essas informações.

---

## 5.3 Especificação da API inicial

### `GET /health`

Verifica se a API está ativa.

Resposta esperada:

```json
{
  "status": "ok",
  "message": "API is running"
}
```

---

---

## 5.4 Especificação do frontend inicial

### Rota `/home`

Deve exibir:

- título do projeto;
- resumo;
- cards com tecnologias;
- seção “Problema”;
- seção “Objetivo”;
- seção “Escopo”;
- seção “Próximas etapas”.

### Rota `/info`

Pode ser igual à `/home` ou uma página mais técnica.

### Rota `/api-test`

Opcionalmente, uma página simples para testar a conexão com o backend.

---

## 5.5 Entregáveis da fase 1

| Entregável           | Descrição                         |
| -------------------- | --------------------------------- |
| Backend inicial      | FastAPI rodando localmente        |
| Frontend inicial     | Angular rodando localmente        |
| Rota `/health`       | Teste básico da API               |
| Rota `/info`         | Dados descritivos do projeto      |
| Página `/home`       | Página institucional do projeto   |
| Documentação inicial | README com instruções de execução |

---

## 5.6 Critérios de aceite

A fase 1 estará pronta quando:

- [ ] O backend iniciar sem erro;
- [ ] O frontend iniciar sem erro;
- [ ] O frontend conseguir consumir a rota `/info`;
- [ ] A página `/home` exibir a descrição do projeto;
- [ ] O README explicar como rodar backend e frontend;
- [ ] A estrutura de pastas estiver definida.

---

## 5.7 Tarefas sugeridas

### Backend

- [ ] Criar ambiente virtual Python;
- [ ] Instalar FastAPI, Uvicorn e Pydantic;
- [ ] Gerenciar dependencias com `uv`;
- [ ] Criar `backend/app/main.py`;
- [ ] Criar rota `GET /health`;
- [ ] Criar rota `GET /info`;
- [ ] Configurar CORS para permitir acesso do Angular;
- [ ] Criar README do backend.

### Frontend

- [ ] Criar projeto Angular;
- [ ] Criar componente `Home`;
- [ ] Criar serviço `ApiService`;
- [ ] Exibir dados do projeto na página `Home`;
- [ ] Criar layout inicial com navbar;
- [ ] Analisar e utilizar os estilos dos compoenentes [desse projeto](https://github.com/mscaro23/ProjFinal-AlgGraf/tree/main/front-end)
- [ ] Criar README do frontend.

### Documentação

- [ ] Criar `docs/specification.md`;
- [ ] Criar `docs/api-contract.md`;
- [ ] Criar `docs/frontend-spec.md`;
- [ ] Documentar escopo inicial.

---

# 6. Plano 2 — Pipelines de Dados e Machine Learning

## 6.1 Objetivo

Construir a parte central do trabalho: a pipeline de Machine Learning.

Essa fase deve cobrir:

- carregamento do dataset do Kaggle;
- entendimento da estrutura dos dados;
- limpeza;
- análise exploratória;
- engenharia de features;
- criação do target;
- treinamento de pelo menos dois modelos supervisionados;
- validação;
- avaliação;
- exportação dos resultados para API e frontend.

---

## 6.2 Especificação funcional

O sistema deve ser capaz de processar dados históricos do Polymarket e gerar modelos supervisionados que tentam prever a direção futura do preço de um contrato.

Pergunta central:

> Dado o estado recente de um contrato no Polymarket, é possível prever se seu preço irá subir nos próximos 15 minutos?

---

## 6.3 Pipeline proposta

## Etapa 1 — Carregamento e limpeza dos dados

Script:

```txt
backend/pipelines/01_clean_data.py
```

Responsabilidades:

- carregar arquivos do dataset;
- identificar colunas disponíveis;
- padronizar nomes de colunas;
- converter timestamps;
- remover linhas inválidas;
- remover duplicatas;
- tratar valores ausentes;
- salvar versão intermediária.

Saída:

```txt
data/interim/cleaned_orderbook.parquet
```

---

## Etapa 2 — Análise exploratória

Script:

```txt
backend/pipelines/02_exploratory_analysis.py
```

Perguntas a responder:

- Quantos mercados existem?
- Quantos contratos possuem liquidez suficiente?
- Como os preços se distribuem?
- Como o spread se comporta?
- Existem muitos valores ausentes?
- Existem contratos com poucos registros?
- Quais mercados são bons candidatos para treino?

Saídas:

```txt
backend/outputs/market_summary.json
backend/outputs/eda_summary.json
data/outputs/eda_tables/
```

Visualizações possíveis:

- distribuição de preços;
- distribuição de spreads;
- volume por mercado;
- quantidade de registros por contrato;
- série temporal de preço de alguns contratos.

---

## Etapa 3 — Engenharia de features

Script:

```txt
backend/pipelines/03_feature_engineering.py
```

### Features de preço

- `mid_price`;
- `last_price`;
- `return_1m`;
- `return_5m`;
- `return_10m`;
- `return_15m`;
- `rolling_mean_5m`;
- `rolling_mean_15m`;
- `rolling_std_5m`;
- `rolling_std_15m`.

### Features de spread

- `spread`;
- `relative_spread`;
- `spread_change_1m`;
- `rolling_spread_mean_5m`.

### Features de volume/liquidez

- `volume_1m`;
- `volume_5m`;
- `volume_15m`;
- `rolling_volume_mean_5m`;
- `liquidity`;
- `liquidity_change`.

### Features de order book

- `best_bid`;
- `best_ask`;
- `bid_depth`;
- `ask_depth`;
- `book_imbalance`;
- `depth_imbalance`.

Fórmula simples para imbalance:

```txt
book_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
```

Interpretação:

- valor próximo de `1`: pressão compradora;
- valor próximo de `-1`: pressão vendedora;
- valor próximo de `0`: equilíbrio entre compra e venda.

Saída:

```txt
data/processed/features.parquet
```

---

## Etapa 4 — Criação do target

Script:

```txt
backend/pipelines/04_create_targets.py
```

Target principal:

```txt
future_return_15m = mid_price_t+15m - mid_price_t
```

Classe:

```txt
target_up_15m = 1 se future_return_15m > threshold
target_up_15m = 0 caso contrário
```

O `threshold` evita classificar microvariações irrelevantes como subida real.

Exemplo:

```txt
threshold = 0.01
```

Ou seja, o preço precisa subir pelo menos 1 centavo para ser considerado uma subida relevante.

Saída:

```txt
data/processed/model_dataset.parquet
```

---

## Etapa 5 — Treinamento dos modelos

Script:

```txt
backend/pipelines/05_train_models.py
```

Modelos recomendados:

### Modelo 1 — Regressão Logística

Função no projeto:

- baseline interpretável;
- modelo simples;
- bom para comparar com modelos mais complexos.

### Modelo 2 — Árvore de Decisão

Função no projeto:

- modelo supervisionado não linear;
- fácil de explicar;
- permite visualizar regras.

### Modelo 3 — Random Forest

Função no projeto:

- modelo mais robusto;
- melhor capacidade preditiva;
- permite ranking de importância das features.

Opcional:

- XGBoost;
- LightGBM.

---

## Etapa 6 — Validação

Como os dados são temporais, evitar split aleatório simples.

Estratégia recomendada:

```txt
Treino: primeiros 70% dos dados no tempo
Validação: próximos 15%
Teste: últimos 15%
```

Ou por timestamp:

```txt
train = dados até determinada data
validation = período seguinte
test = período final
```

---

## Etapa 7 — Avaliação

Script:

```txt
backend/pipelines/06_evaluate_models.py
```

Métricas principais:

- `accuracy`;
- `precision`;
- `recall`;
- `f1_score`;
- `roc_auc`;
- `confusion_matrix`.

Também calcular:

- `baseline_accuracy`;
- `class_distribution`;
- `feature_importance`.

Baseline importante:

```txt
Sempre prever a classe majoritária
```

Exemplo:

Se 63% dos exemplos são “não sobe”, um modelo com 62% de acurácia é ruim, porque perde para o baseline.

---

## Etapa 8 — Exportação dos resultados

Script:

```txt
backend/pipelines/07_export_results.py
```

Saídas para a API:

```txt
backend/models/logistic_regression.joblib
backend/models/decision_tree.joblib
backend/models/random_forest.joblib

backend/outputs/metrics.json
backend/outputs/confusion_matrices.json
backend/outputs/feature_importance.json
backend/outputs/predictions_sample.json
backend/outputs/market_summary.json
```

---

## 6.4 Entregáveis da fase 2

| Entregável               | Descrição                                          |
| ------------------------ | -------------------------------------------------- |
| Dataset limpo            | Arquivo tratado em `.parquet` ou `.csv`            |
| Dataset modelável        | Arquivo com features + target                      |
| Scripts de pipeline      | Limpeza, EDA, features, target, treino e avaliação |
| Modelos treinados        | Arquivos `.joblib`                                 |
| Métricas                 | JSON com resultados dos modelos                    |
| Análise exploratória     | Tabelas e gráficos exportados                      |
| Documentação da pipeline | Explicação de cada etapa                           |

---

## 6.5 Critérios de aceite

A fase 2 estará pronta quando:

- [ ] A pipeline rodar do início ao fim;
- [ ] Os dados brutos forem transformados em dataset de treino;
- [ ] Houver pelo menos dois modelos supervisionados treinados;
- [ ] Os modelos forem avaliados com métricas apropriadas;
- [ ] Houver divisão temporal entre treino, validação e teste;
- [ ] Os resultados estiverem exportados em formato consumível pela API;
- [ ] O grupo souber explicar as principais features e limitações.

---

# 7. Plano 3 — Integração com Frontend e Dashboards

## 7.1 Objetivo

Transformar os resultados da pipeline em uma aplicação interativa.

O usuário deve conseguir acessar o frontend Angular e visualizar:

- descrição do projeto;
- resumo do dataset;
- contratos analisados;
- gráficos de preço, spread, volume e imbalance;
- comparação entre modelos;
- métricas de avaliação;
- importância das features;
- exemplos de previsão;
- limitações e discussão ética.

---

## 7.2 Especificação da API final

### `GET /markets`

Retorna lista de mercados/contratos disponíveis.

Resposta exemplo:

```json
[
  {
    "market_id": "123",
    "title": "Will event X happen?",
    "total_records": 15230,
    "avg_volume": 243.5,
    "avg_spread": 0.021
  }
]
```

---

### `GET /markets/{market_id}/timeseries`

Retorna dados temporais de um contrato.

Resposta exemplo:

```json
{
  "market_id": "123",
  "series": [
    {
      "timestamp": "2025-01-01T10:00:00",
      "mid_price": 0.52,
      "spread": 0.02,
      "volume": 1200,
      "book_imbalance": 0.31
    }
  ]
}
```

---

### `GET /models/metrics`

Retorna comparação dos modelos.

Resposta exemplo:

```json
{
  "models": [
    {
      "name": "Logistic Regression",
      "accuracy": 0.58,
      "precision": 0.56,
      "recall": 0.61,
      "f1_score": 0.58,
      "roc_auc": 0.62
    },
    {
      "name": "Decision Tree",
      "accuracy": 0.55,
      "precision": 0.54,
      "recall": 0.57,
      "f1_score": 0.55,
      "roc_auc": 0.58
    },
    {
      "name": "Random Forest",
      "accuracy": 0.63,
      "precision": 0.62,
      "recall": 0.6,
      "f1_score": 0.61,
      "roc_auc": 0.67
    }
  ]
}
```

---

### `GET /models/feature-importance`

Retorna importância das features.

Resposta exemplo:

```json
{
  "model": "Random Forest",
  "features": [
    {
      "name": "book_imbalance",
      "importance": 0.21
    },
    {
      "name": "spread",
      "importance": 0.17
    },
    {
      "name": "return_5m",
      "importance": 0.14
    }
  ]
}
```

---

### `GET /predictions/sample`

Retorna exemplos de previsões.

Resposta exemplo:

```json
[
  {
    "market_id": "123",
    "timestamp": "2025-01-01T10:00:00",
    "actual": 1,
    "predicted": 1,
    "probability_up": 0.72,
    "mid_price": 0.52,
    "future_price_15m": 0.55
  }
]
```

---

### `POST /predict`

Opcional para demonstração interativa.

Recebe features manuais e retorna uma previsão.

Request:

```json
{
  "mid_price": 0.52,
  "spread": 0.02,
  "volume_5m": 1200,
  "book_imbalance": 0.31,
  "return_5m": 0.015,
  "rolling_std_15m": 0.04
}
```

Resposta:

```json
{
  "model": "Random Forest",
  "prediction": 1,
  "label": "Preço tende a subir",
  "probability_up": 0.72
}
```

---

# 8. Telas do Frontend Angular

## 8.1 Página `/home`

Conteúdo:

- título do projeto;
- resumo;
- introdução;
- problema;
- objetivo;
- escopo;
- tecnologias;
- explicação sobre Polymarket;
- explicação sobre contratos;
- explicação sobre livro de ofertas;
- seção ética.

---

## 8.2 Página `/dashboard`

Conteúdo:

- quantidade de mercados analisados;
- quantidade de registros;
- período dos dados;
- distribuição das classes;
- melhor modelo;
- métricas principais;
- cards de resumo.

Cards sugeridos:

- total de contratos analisados;
- total de registros processados;
- modelo com maior F1-score;
- ROC-AUC do melhor modelo;
- feature mais importante;
- proporção da classe “sobe”.

---

## 8.3 Página `/markets`

Conteúdo:

- tabela de contratos;
- filtros por volume;
- filtros por spread médio;
- filtros por quantidade de registros;
- seleção de contrato.

Ao clicar em um contrato, abrir detalhes com:

- gráfico de preço;
- gráfico de spread;
- gráfico de volume;
- gráfico de book imbalance.

---

## 8.4 Página `/model-comparison`

Conteúdo:

- tabela de métricas por modelo;
- matriz de confusão;
- gráfico comparando accuracy, precision, recall, F1 e ROC-AUC;
- explicação sobre o papel de cada modelo.

---

## 8.5 Página `/prediction-explorer`

Conteúdo:

- exemplos de previsões corretas;
- exemplos de previsões erradas;
- probabilidade prevista;
- preço atual;
- preço depois de 15 minutos;
- classe real;
- classe prevista.

---

## 8.6 Página `/ethics-limitations`

Conteúdo:

- limitações técnicas;
- limitações de mercado;
- risco de overfitting;
- baixa liquidez;
- ruído;
- uso responsável;
- aviso de que o sistema é acadêmico e não recomendação financeira.

---

# 9. Critérios de Aceite da Fase 3

A fase 3 estará pronta quando:

- [ ] O frontend consumir os resultados via API;
- [ ] O dashboard exibir métricas reais dos modelos;
- [ ] Houver gráficos de pelo menos um contrato;
- [ ] Houver comparação entre modelos;
- [ ] Houver visualização de importância das features;
- [ ] Houver exemplos de previsão;
- [ ] A página inicial explicar o projeto claramente;
- [ ] O grupo conseguir apresentar o fluxo completo: dados → pipeline → modelos → API → dashboard.

---

# 10. Roadmap Resumido

## Fase 1 — Base da aplicação

- [ ] Criar repositório;
- [ ] Criar backend FastAPI;
- [ ] Criar frontend Angular;
- [ ] Criar rota `/health`;
- [ ] Criar rota `/info`;
- [ ] Criar página `/home`;
- [ ] Documentar escopo.

## Fase 2 — Machine Learning

- [ ] Carregar dataset do Kaggle;
- [ ] Limpar dados;
- [ ] Fazer análise exploratória;
- [ ] Criar features;
- [ ] Criar target de 15 minutos;
- [ ] Treinar modelos supervisionados;
- [ ] Avaliar modelos;
- [ ] Exportar métricas e modelos.

## Fase 3 — Dashboard

- [ ] Criar endpoints finais;
- [ ] Exibir resumo do dataset;
- [ ] Exibir gráficos de contratos;
- [ ] Comparar modelos;
- [ ] Exibir métricas;
- [ ] Exibir importância das features;
- [ ] Exibir exemplos de previsão;
- [ ] Adicionar página de limitações e ética.

---

# 11. Ordem Recomendada de Implementação

Seguir esta ordem para reduzir risco técnico:

1. Criar monorepo;
2. Criar FastAPI com `/health` e `/info`;
3. Criar Angular com `/home`;
4. Conectar Angular com FastAPI;
5. Baixar e entender o dataset;
6. Fazer notebook inicial de EDA;
7. Criar script de limpeza;
8. Criar script de features;
9. Criar target de 15 minutos;
10. Treinar Regressão Logística;
11. Treinar Árvore de Decisão;
12. Treinar Random Forest;
13. Avaliar modelos;
14. Exportar métricas para JSON;
15. Criar endpoints `/models/metrics` e `/models/feature-importance`;
16. Criar dashboard de comparação;
17. Criar página de contratos;
18. Criar gráficos temporais;
19. Criar página de exemplos de previsão;
20. Finalizar documentação e apresentação.

---

# 12. Instruções para o Codex

## 12.1 Prioridade de desenvolvimento

O desenvolvimento deve seguir os três planos na ordem:

1. Estrutura inicial;
2. Pipelines de dados e Machine Learning;
3. Integração com frontend e dashboards.

Não implementar dashboards antes de existir pelo menos uma API básica.  
Não implementar treinamento de modelos antes de existir estrutura de dados processados.  
Não conectar frontend aos resultados finais antes de existirem arquivos de saída exportados pela pipeline.

---

## 12.2 Regras de implementação

- Usar Python no backend;
- Usar FastAPI para API;
- Usar Pandas nas pipelines;
- Usar Scikit-learn para os modelos iniciais;
- Usar Angular no frontend;
- Manter contratos de API documentados;
- Separar rotas, schemas e services no backend;
- Separar pages, services e models no Angular;
- Salvar modelos treinados em `backend/models/`;
- Salvar métricas e arquivos de saída em `backend/outputs/`;
- Não expor dados brutos gigantes via API;
- Para dashboards, expor dados agregados ou amostras controladas.

---

## 12.3 Definition of Done geral

O projeto será considerado completo quando:

- O frontend Angular estiver funcional;
- A API FastAPI estiver funcional;
- A página `/home` explicar o projeto;
- A pipeline processar dados do Kaggle;
- Pelo menos dois modelos supervisionados forem treinados;
- Os modelos forem avaliados com métricas adequadas;
- Os resultados forem exportados;
- O frontend exibir dashboards com esses resultados;
- Houver uma seção de limitações e ética;
- O README explicar como executar o projeto.

---

# 13. Observações Finais

Este projeto deve ser apresentado como uma ferramenta acadêmica de análise de microestrutura de mercado, não como recomendação financeira ou sistema de trading real.

O foco principal é demonstrar domínio sobre:

- definição de problema supervisionado;
- tratamento de dados;
- engenharia de features;
- treinamento e comparação de modelos;
- avaliação com métricas apropriadas;
- interpretação dos resultados;
- construção de uma aplicação com backend, frontend e visualização.
