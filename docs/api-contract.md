# Contrato da API

URL local: `http://localhost:8000`

## `GET /health`

Verifica se a API está disponível.

### Resposta de sucesso

Status HTTP: `200 OK`

```json
{
  "status": "ok",
  "message": "API is running"
}
```

## Evolução planejada

Endpoints para mercados, métricas e previsões serão definidos somente após a criação da pipeline de
dados e a exportação de resultados reais. Não existe endpoint `/info`: o conteúdo institucional é
mantido diretamente no frontend.

