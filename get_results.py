import json
with open("results_og90_500.json") as f:
    data = json.load(f)

print("Top 10 Matches:")
for m in data["results"][0]["matches"][:10]:
    print(f"ID: {m['id']}, Score: {m['score']}")

print("\nValidation Summary:")
print(json.dumps(data["validation"], indent=2))
