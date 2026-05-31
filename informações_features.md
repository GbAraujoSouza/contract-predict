# DESCRIÇÃO DAS VARIÁVEIS:

| Variável                 | Tipo           | Descrição                                                                                                                                                                                                            |
| ------------------------ | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **market_id**            | String         | Identificador único do mercado da Polymarket. Diferencia cada contrato de previsão (ex.: eleições, esportes, criptomoedas etc.).                                                                                     |
| **minute_bar**           | Datetime (UTC) | Timestamp correspondente ao início da barra temporal de 1 minuto utilizada na agregação das features.                                                                                                                |
| **close_mid**            | Float32        | Preço médio de fechamento (*mid-price*) da barra. Calculado como a média entre o melhor preço de compra (*best bid*) e o melhor preço de venda (*best ask*).                                                         |
| **mean_spread**          | Float32        | Spread médio da barra de 1 minuto. Mede o custo médio implícito de negociação durante o período.                                                                                                                     |
| **close_spread**         | Float32        | Spread observado no fechamento da barra temporal.                                                                                                                                                                    |
| **bar_volatility**       | Float32        | Volatilidade do preço dentro da barra de 1 minuto. Representa a intensidade das oscilações de preço durante o período.                                                                                               |
| **total_volume**         | Float32        | Volume total negociado na barra de 1 minuto.                                                                                                                                                                         |
| **buy_volume**           | Float32        | Volume total de negociações iniciadas por compradores durante a barra.                                                                                                                                               |
| **sell_volume**          | Float32        | Volume total de negociações iniciadas por vendedores durante a barra.                                                                                                                                                |
| **trade_count**          | UInt32         | Quantidade de negociações executadas durante a barra temporal.                                                                                                                                                       |
| **order_flow_imbalance** | Float32        | Desequilíbrio do fluxo de ordens. Calculado como: (buy_volume - sell_volume) / total_volume. Valores positivos indicam predominância compradora; negativos indicam predominância vendedora.                          |
| **return_1m**            | Float32        | Retorno percentual do preço médio (*mid-price*) ao longo de 1 minuto.                                                                                                                                                |
| **bid_depth**            | Float64        | Profundidade total do lado comprador do livro de ofertas (*bid side*), obtida a partir dos snapshots L2.                                                                                                             |
| **ask_depth**            | Float64        | Profundidade total do lado vendedor do livro de ofertas (*ask side*), obtida a partir dos snapshots L2.                                                                                                              |
| **depth_imbalance**      | Float64        | Desequilíbrio de profundidade do livro de ofertas. Calculado como: (bid_depth - ask_depth) / (bid_depth + ask_depth). Valores positivos indicam maior pressão compradora; negativos indicam maior pressão vendedora. |
| **target**               | Int8           | Variável alvo do problema de classificação. Assume valor 1 se o retorno futuro de 15 minutos for positivo e 0 caso contrário.                                                                                        |

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
