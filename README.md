# Contract Predict

Aplicação acadêmica para investigar se sinais do livro de ofertas ajudam a prever movimentos de
curto prazo em contratos binários do Polymarket.

Este repositório contém a base da aplicação:

- API FastAPI com endpoint de saúde;
- frontend Angular com páginas institucionais;
- comunicação básica entre frontend e backend;
- estrutura reservada para dados, modelos e resultados futuros.

O projeto não é uma recomendação financeira nem um sistema de trading.

## Estrutura

```text
backend/   API FastAPI e diretórios reservados para artefatos de ML
data/      Dados brutos, intermediários, processados e saídas futuras
docs/      Especificações da aplicação
frontend/  Aplicação Angular
```

## Requisitos

- Python 3.12 ou superior;
- [uv](https://docs.astral.sh/uv/);
- Node.js 22 ou superior;
- npm 11 ou superior.

## Executar localmente

Em um terminal, inicie a API:

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload
```

Em outro terminal, inicie o frontend:

```bash
cd frontend
npm install
npm start
```

A aplicação estará disponível em `http://localhost:4200`. A documentação interativa da API estará
em `http://localhost:8000/docs`.

## Roadmap

1. Base da aplicação com FastAPI e Angular.
2. Pipeline de dados e modelos supervisionados.
3. Integração dos resultados com dashboards.

Consulte [plano_desenvolvimento_polymarket_ml.md](./plano_desenvolvimento_polymarket_ml.md) para o
planejamento completo.

