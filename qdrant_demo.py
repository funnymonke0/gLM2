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


print("Loading test plasmids from CSV...")
plasmids_df = pd.read_csv("ref_seq_plasmids.csv").head(1000)  # Load all plasmids
print(f"Loaded {len(plasmids_df)} plasmids:")
all_sequences = plasmids_df['contig'].tolist()
BATCH_SIZE = 6 # number of sequences to embed at once (adjust based on your GPU memory)
MAX_SEQ_LEN = 4096  # expected context length of plasmids (adjust if needed)
embeddings = []

print(f"Generating embeddings in batches of {BATCH_SIZE}...")
for i in range(0, len(all_sequences), BATCH_SIZE):
    batch = all_sequences[i:i + BATCH_SIZE]
    tokenized = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LEN)
    with torch.no_grad():
        batch_emb = model(
            input_ids=tokenized['input_ids'].to(DEVICE),
            attention_mask=tokenized['attention_mask'].to(DEVICE)
        ).pooler_output.float().cpu().tolist()
    embeddings.extend(batch_emb)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

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
        payload={"plasmid_id": plasmid_id, "contig": seq}
    )
    for idx, (plasmid_id, seq, emb) in enumerate(zip(plasmids_df['plasmid_id'], plasmids_df['contig'], embeddings))
]


client.upsert(
    collection_name=collection_name,
    points=points,
)

print("Embeddings uploaded to Qdrant!")