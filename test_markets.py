import requests
import pandas as pd
import json

url = "https://gamma-api.polymarket.com/markets"

params = {
    "limit": 10
}

response = requests.get(url, params=params)

print("Status:", response.status_code)

data = response.json()

print(f"Mercados encontrados: {len(data)}")

df = pd.json_normalize(data)

print(df.columns.tolist())

print(df[[
    "question",
    "volume",
    "liquidity",
    "endDate"
]].head())

with open("markets.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=3)

print("Arquivo JSON salvo com sucesso!")