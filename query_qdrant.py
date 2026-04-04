from qdrant_client import QdrantClient
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from huggingface_hub import login
print("\n" + "="*60)
print("SEARCH TEST")
print("="*60)

print("Connecting to Qdrant on localhost:6333...")
client = QdrantClient(url="http://localhost:6333")
with open("token.txt", "r") as f:
    token = f.read().strip()  # Load Hugging Face token from file
login(token=token)  # Replace with your Hugging Face token if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
print("Loading gLM2_650M_embed model...")
model = AutoModel.from_pretrained('tattabio/gLM2_650M_embed', dtype=MODEL_DTYPE, trust_remote_code=True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M_embed', trust_remote_code=True)

print("Loading test plasmids from CSV...")
plasmids_df = pd.read_csv("ref_seq_plasmids.csv").head(50)  # Load first 50 plasmids for demo
all_query_sequences = plasmids_df["contig"].tolist()

BATCH_SIZE = 4
collection_name = "ref_seq_plasmids"
# Embed query
q_embeddings = []

for i in range(0, len(all_query_sequences), BATCH_SIZE):
    batch = all_query_sequences[i:i + BATCH_SIZE]
    enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        query_emb = model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE)
        ).pooler_output.float().cpu().numpy()
    q_embeddings.extend(query_emb)

for i, seq in enumerate(all_query_sequences):
    # Search in Qdrant
    res =client.query_points(
        collection_name=collection_name,
        query=q_embeddings[i].tolist(),
        limit=3,
    )
    results = getattr(res, "points", [])

    print(f"\nTop 3 most similar plasmids:")
    for i, result in enumerate(results, 1):
        plasmid_id = result.payload["plasmid_id"]
        similarity = result.score
        contig = result.payload["contig"]
        print(f"{i}. {plasmid_id} (similarity: {similarity:.4f})")
        print(f"   Contig: {contig}\n")

print("="*60)
print("SUCCESS! Qdrant + gLM2_embed working locally.")
print("="*60)