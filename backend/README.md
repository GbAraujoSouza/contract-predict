# Backend

API FastAPI do Contract Predict. Nesta fase, ela expõe somente o endpoint de saúde usado pelo
frontend para validar a integração.

## Executar

```bash
uv sync
uv run uvicorn app.main:app --reload
```

A API estará disponível em `http://localhost:8000`.

## Endpoint

```http
GET /health
```

Resposta esperada:

```json
{
  "status": "ok",
  "message": "API is running"
}
```

Os diretórios `models/` e `outputs/` estão reservados para fases posteriores.

