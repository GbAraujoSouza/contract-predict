# Especificação do frontend

## Direção visual

Interface escura responsiva com fundo radial, cards translúcidos e destaques em azul e verde. A
linguagem visual adapta o projeto de referência para um contexto de dados de mercado.

## Rotas

### `/home`

Página institucional com:

- título e resumo do projeto;
- problema estudado;
- representação visual simplificada de um livro de ofertas;
- indicador de conexão com `GET /health`;
- escopo da análise;
- próximas etapas.

### `/info`

Página técnica estática com:

- explicação sobre o Polymarket;
- explicação sobre contratos binários;
- objetivo do modelo;
- limitações de uso;
- stack planejada.

## Estados da integração

O indicador da API na home deve apresentar três estados:

- verificando API;
- API conectada;
- API indisponível.

