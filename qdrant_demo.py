import torch
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import numpy as np

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

# Load gLM2_embed model (for retrieval, not base gLM2)
print("Loading gLM2_650M_embed model...")
model = AutoModel.from_pretrained('tattabio/gLM2_650M_embed', torch_dtype=MODEL_DTYPE, trust_remote_code=True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M_embed', trust_remote_code=True)

# Connect to Qdrant (running in Docker)
print("Connecting to Qdrant on localhost:6333...")
client = QdrantClient(url="http://localhost:6333")

# Test data: some example proteins
test_proteins = {
    "protein_1": "<+>MALTKVEKRNRIKRRVRGKISGTQASPRLSVYKSNK",
    "protein_2": "<+>MKVLEENLRQQQARKEGLKPVLEWDQTVKK",
    "protein_3": "<+>MKKLAVTMLLTASACDEFVQAKEEGLKPVLEWDQ",
    "protein_4": "<+>MLGIDNIERVKPGGLELVDRLVAVNRVTKVTKGGRAFGFSAIVVVGNED",
    "protein_5": "<+>MKVLEENLRQQQARKEGLKPVLEWDQTVKKEEEEEE",
}

# Generate embeddings
print("Generating embeddings...")
embeddings = {}
for protein_id, seq in test_proteins.items():
    enc = tokenizer([seq], return_tensors='pt')
    with torch.no_grad():
        emb = model(enc.input_ids.to(DEVICE)).pooler_output.cpu().numpy()
    embeddings[protein_id] = emb[0]  # Shape: (512,)
    print(f"  {protein_id}: shape {embeddings[protein_id].shape}")

# Create Qdrant collection
collection_name = "proteins_test"
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

# Upload embeddings to Qdrant
print("\nUploading embeddings to Qdrant...")
points = []
for idx, (protein_id, emb) in enumerate(embeddings.items()):
    points.append(
        PointStruct(
            id=idx,
            vector=emb.tolist(),
            payload={"protein_id": protein_id, "sequence": test_proteins[protein_id]}
        )
    )

client.upsert(
    collection_name=collection_name,
    points=points,
)
print(f"Uploaded {len(points)} embeddings!")

# Query: find similar proteins
print("\n" + "="*60)
print("SEARCH TEST")
print("="*60)

query_id = "protein_1"
query_seq = test_proteins[query_id]
print(f"\nQuery: {query_id} = {query_seq}")

# Embed query
enc = tokenizer([query_seq], return_tensors='pt') 
with torch.no_grad():
    query_emb = model(enc.input_ids.to(DEVICE)).pooler_output.cpu().numpy()[0]

# Search in Qdrant
res =client.query_points(
    collection_name=collection_name,
    query=query_emb.tolist(),
    limit=3,
)
results = getattr(res, "points", [])

print(f"\nTop 3 most similar proteins:")
for i, result in enumerate(results, 1):
    protein_id = result.payload["protein_id"]
    similarity = result.score
    sequence = result.payload["sequence"]
    print(f"{i}. {protein_id} (similarity: {similarity:.4f})")
    print(f"   Sequence: {sequence}\n")

print("="*60)
print("SUCCESS! Qdrant + gLM2_embed working locally.")
print("="*60)
