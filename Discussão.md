## Discussão sobre qualidade do modelo

Com base nos dois notebooks, ambos tratam do mesmo problema: prever, a partir de features de microestrutura de mercado (preço médio, profundidade do book, spread, volatilidade, order flow imbalance), se o preço de um contrato da Polymarket vai subir num horizonte futuro (1 min no primeiro notebook, 15 min no segundo). O segundo notebook é uma versão muito mais rigorosa metodologicamente do primeiro.

### Pontos fortes

- O ROC-AUC ficando consistentemente entre 0,89 e 0,92 em ambos os notebooks é um resultado forte para dados financeiros de microestrutura, sugerindo que as features de book (especialmente `close_mid`, `depth_imbalance` e `bid_depth`/`ask_depth`) carregam sinal real.
- O segundo notebook corrige falhas metodológicas importantes do primeiro: introduz purge gap de 15 minutos entre treino e teste (evitando que o rótulo, que olha 15 min para frente, "veja" dados do período de validação), usa holdout final cego nunca tocado durante o tuning, e compara dois protocolos de validação temporal (TimeSeriesSplit expansivo e walk forward com janela fixa). Essa é a prática recomendada para séries temporais financeiras.
- A consistência dos resultados entre TSCV e walk forward (ROC-AUC final entre 0,90 e 0,91 para os quatro modelos no holdout) é tranquilizadora. Indica que o sinal não é um artefato de overfitting no protocolo de validação.

### Limitações relevantes

1. **Vazamento temporal no primeiro notebook.** O uso de KFold tradicional (mesmo com shuffle=False) e do split 80/20 sem purge gap é problemático, porque o target provavelmente é calculado com base em preços futuros. Sem gap, há contaminação entre treino e teste. Isso explica por que as métricas do K Fold caem em relação ao split cronológico simples no primeiro notebook: o protocolo "ingênuo" provavelmente estava inflando os resultados.

2. **Desbalanceamento de classes.** O baseline de sempre prever classe 0 já acerta cerca de 73% das vezes. Isso torna a accuracy uma métrica pouco informativa. O ROC-AUC e a curva precision recall (PR-AUC entre 0,80 e 0,82) são mais confiáveis, mas mesmo assim a precisão (em torno de 0,52) indica que, das vezes que o modelo prevê "vai subir", ele erra quase metade.

3. **Poucas features e ausência de variáveis exógenas.** O conjunto de features é limitado a sinais de book e preço de um único mercado, sem nenhuma feature de contexto (notícias, eventos externos, mercados correlacionados). Em mercados de previsão como a Polymarket, o preço frequentemente reage a eventos externos ao próprio book, que o modelo não capta.

4. **Generalização entre mercados.** Os dados parecem agregar vários `market_id` diferentes. Não há clareza se o split temporal preserva isolamento adequado entre mercados ou se há vazamento de padrões específicos de um mercado para outro via o agrupamento temporal.

5. **Curto período de dados.** O dataset cobre apenas cerca de 6 dias (06 a 11/03), uma amostra pequena para validar robustez em diferentes regimes de mercado (alta volatilidade, baixa liquidez, eventos de cauda).

6. **Custos de transação não considerados.** A métrica de qualidade é estatística (ROC-AUC, F1), mas não há simulação de PnL considerando spread, slippage e taxas. Um modelo com AUC 0,90 ainda pode ser não lucrativo na prática se o spread consumir o edge.

7. **Threshold fixo de 0,5.** A avaliação final usa `y_proba >= 0,5` sem otimizar o threshold para o caso de uso (por exemplo, maximizar o valor esperado de uma estratégia de trading), o que é uma escolha arbitrária do ponto de vista de aplicação real.

## Reflexão ética

Esse tipo de modelo levanta algumas questões importantes:

- **Assimetria de informação em mercados de apostas e previsão.** A Polymarket é, na prática, uma plataforma de apostas sobre eventos do mundo real (eleições, esportes, eventos geopolíticos). Um modelo que prevê micro movimentos de preço com vantagem estatística pode ser usado para extrair valor de participantes menos sofisticados, como apostadores casuais. Isso reabre debates antigos sobre fairness em mercados financeiros e de apostas quando há desigualdade de acesso a ferramentas analíticas.

- **Mercados de previsão como sinal social.** A Polymarket é frequentemente citada como termômetro de eventos reais, como a probabilidade de resultados eleitorais. Estratégias automatizadas de alta frequência operando sobre micro movimentos podem distorcer esse sinal, fazendo com que o preço reflita mais a dinâmica de bots do que a sabedoria coletiva agregada que dá legitimidade a esses mercados.

- **Risco de uso problemático e vício em apostas.** Ferramentas de previsão de movimento de preço podem incentivar trading excessivo de curtíssimo prazo (scalping). Isso se aproxima de dinâmicas de apostas compulsivas, especialmente se incorporado em produtos voltados a usuários não profissionais.

- **Questões regulatórias e jurisdicionais.** Mercados de previsão como a Polymarket operam em uma zona regulatória ambígua em várias jurisdições, incluindo restrições nos EUA. Construir e operar modelos preditivos para extrair lucro desses mercados levanta a questão de até que ponto isso se sujeita, ou deveria se sujeitar, à mesma supervisão que mercados financeiros tradicionais.

- **Transparência e responsabilidade do modelo.** Como o modelo provavelmente será usado para decisões financeiras automatizadas, erros sistemáticos (por exemplo em eventos de cauda, ou em mercados manipulados ou com baixa liquidez) podem gerar perdas concentradas rapidamente. É importante que qualquer aplicação real tenha mecanismos de controle e supervisão humana, não apenas confiança nas métricas de validação offline.
