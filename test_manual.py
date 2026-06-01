import json

with open("markets.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(type(data))

print(f"Mercados encontrados: {len(data)}")

print(data[0])
print(data[0].keys())
print(data[0]["question"])