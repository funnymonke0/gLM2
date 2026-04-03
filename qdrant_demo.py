import torch
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import numpy as np
import pandas as pd

from huggingface_hub import login
# Setup
with open("token.txt", "r") as f:
    token = f.read().strip()  # Load Hugging Face token from file
login(token=token)  # Replace with your Hugging Face token if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

# Load gLM2_embed model (for retrieval, not base gLM2)
print("Loading gLM2_650M_embed model...")
model = AutoModel.from_pretrained('tattabio/gLM2_650M_embed', dtype=MODEL_DTYPE, trust_remote_code=True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M_embed', trust_remote_code=True)

# Connect to Qdrant (running in Docker)
print("Connecting to Qdrant on localhost:6333...")
client = QdrantClient(url="http://localhost:6333")


print("Loading test proteins from CSV...")
proteins_df = pd.read_csv("ref_seq_plasmids.csv").head(50)  # Load first 150 plasmids for demo
print(f"Loaded {len(proteins_df)} proteins:")
all_sequences = proteins_df['contig'].tolist()
BATCH_SIZE = 64 # number of sequences to embed at once (adjust based on your GPU memory)
embeddings = []

print(f"Generating embeddings in batches of {BATCH_SIZE}...")
for i in range(0, len(all_sequences), BATCH_SIZE):
    batch = all_sequences[i:i + BATCH_SIZE]
    tokenized = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        batch_emb = model(
            input_ids=tokenized['input_ids'].to(DEVICE),
            attention_mask=tokenized['attention_mask'].to(DEVICE)
        ).pooler_output.float().cpu().tolist()
    embeddings.extend(batch_emb)
    print(f"  Batch {i // BATCH_SIZE + 1}/{(len(all_sequences) + BATCH_SIZE - 1) // BATCH_SIZE} done")

print(f"Generated {len(embeddings)} embeddings, each of size {len(embeddings[0])}")



# Create Qdrant collection
collection_name = "ref_seq_plasmids"
print(f"\nCreating Qdrant collection '{collection_name}'...")

# remove old test
try:
    client.delete_collection(collection_name)
except Exception:
    pass

# Create with vector config
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)
print("Collection created!")

# # Upload embeddings to Qdrant
print("\nUploading embeddings to Qdrant...")
points = [
    PointStruct(
        id=idx,
        vector=emb,
        payload={"protein_id": protein_id, "sequence": seq}
    )
    for idx, (protein_id, seq, emb) in enumerate(zip(proteins_df['protein_id'], proteins_df['sequence'], embeddings))
]


client.upsert(
    collection_name=collection_name,
    points=points,
)

print("Embeddings uploaded to Qdrant!")