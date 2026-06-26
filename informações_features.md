# DESCRIÇÃO DAS VARIÁVEIS:

| Coluna | Explicação simples |
|----------|----------|
| **market_id** | Identifica qual mercado está sendo analisado. É como o CPF do contrato. |
| **minute_bar** | O minuto ao qual aquela linha se refere. Ex.: 10:35 da manhã. |
| **close_mid** | Preço médio do contrato naquele minuto. Se estiver em 0,80, o mercado acredita em aproximadamente 80% de chance do evento acontecer. |
| **mean_spread** | Distância média entre o melhor comprador e o melhor vendedor durante o minuto. Quanto menor, mais fácil negociar. |
| **close_spread** | Essa mesma distância no final do minuto. |
| **bar_volatility** | Quanto o preço ficou oscilando durante aquele minuto. Alta volatilidade = preço se mexeu bastante. |
| **total_volume** | Quantidade total negociada naquele minuto. Mede a movimentação do mercado. |
| **buy_volume** | Quanto foi comprado naquele minuto. |
| **sell_volume** | Quanto foi vendido naquele minuto. |
| **trade_count** | Número de negociações realizadas. Não importa o tamanho delas, apenas quantas aconteceram. |
| **order_flow_imbalance** | Mostra se houve mais compras ou mais vendas. Valor positivo = mais compras. Valor negativo = mais vendas. |
| **return_1m** | Quanto o preço mudou em relação ao minuto anterior. Se for positivo, o preço subiu; se for negativo, caiu. |
| **bid_depth** | Quantidade de ordens de compra esperando no livro de ofertas. Mede o interesse dos compradores. |
| **ask_depth** | Quantidade de ordens de venda esperando no livro de ofertas. Mede o interesse dos vendedores. |
| **depth_imbalance** | Compara a força dos compradores com a dos vendedores no livro de ofertas. Positivo = mais compradores; negativo = mais vendedores. |
| **target** | Variável que queremos prever. Vale 1 se o preço subir nos próximos 15 minutos e 0 se não subir. |                                                                                      |

---

# Interpretação Financeira das Features

As variáveis podem ser agrupadas em quatro categorias principais:

1. Variáveis de Preço
- close_mid
- return_1m
- bar_volatility

Capturam o comportamento recente dos preços e a intensidade das movimentações do mercado.

2. Variáveis de Liquidez
- mean_spread
- close_spread

Medem o custo de negociação e a facilidade de execução de ordens.

3. Variáveis de Volume e Fluxo
- total_volume
- buy_volume
- sell_volume
- trade_count
- order_flow_imbalance

Representam a atividade dos participantes do mercado e a direção predominante do fluxo de negociações.

4. Variáveis de Microestrutura do Livro de Ofertas
- bid_depth
- ask_depth
- depth_imbalance

Capturam informações sobre a oferta e demanda presentes no livro de ordens, frequentemente utilizadas em estratégias quantitativas e modelos de previsão de curto prazo.
