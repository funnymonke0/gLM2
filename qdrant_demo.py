import torch
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import numpy as np
import pandas as pd
from huggingface_hub import login
# Setup
with open("hf_token.txt", "r") as f:
    token = f.read().strip()  # Load Hugging Face token from file
login(token=token)  # Replace with your Hugging Face token if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

# Load gLM2_embed model (for retrieval, not base gLM2)
print("Loading gLM2_650M_embed model...")
model = AutoModel.from_pretrained('tattabio/gLM2_650M_embed', dtype=MODEL_DTYPE, trust_remote_code=True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M_embed', trust_remote_code=True)

# Connect to Qdrant (running in Docker)
print("Connecting to Qdrant on localhost:6333...")
client = QdrantClient(url="http://localhost:6333")

# Test data: some example proteins
# test_proteins = {
#     "protein_1": "<+>MALTKVEKRNRIKRRVRGKISGTQASPRLSVYKSNK",
#     "protein_2": "<+>MKVLEENLRQQQARKEGLKPVLEWDQTVKK",
#     "protein_3": "<+>MKKLAVTMLLTASACDEFVQAKEEGLKPVLEWDQ",
#     "protein_4": "<+>MLGIDNIERVKPGGLELVDRLVAVNRVTKVTKGGRAFGFSAIVVVGNED",
#     "protein_5": "<+>MKVLEENLRQQQARKEGLKPVLEWDQTVKKEEEEEE",
# }
print("Loading test proteins from CSV...")
proteins_df = pd.read_csv("proteins_test.csv")
print(f"Loaded {len(proteins_df)} proteins:")
# Batch tokenize all sequences at once
print("Tokenizing all sequences...")
all_sequences = proteins_df['sequence'].tolist()
tokenized = tokenizer(all_sequences, return_tensors='pt', padding=True, truncation=True)
print(f"Tokenized {len(all_sequences)} sequences. Input IDs shape: {tokenized['input_ids'].shape}")
# Batch embed all proteins in one forward pass
print("Generating embeddings for all proteins in a single batch...")
with torch.no_grad():
    embeddings = model(
        input_ids=tokenized['input_ids'].to(DEVICE),
        attention_mask=tokenized['attention_mask'].to(DEVICE)
    ).pooler_output.cpu().tolist()  # Shape: (num_proteins, 512)
print(f"Generated embeddings for {len(embeddings)} proteins. Each embedding shape: {len(embeddings[0])}")


# Generate embeddings
# print("Generating embeddings...")
# embeddings = {}
# for protein_id, seq in test_proteins.items():
#     enc = tokenizer([seq], return_tensors='pt')
#     with torch.no_grad():
#         emb = model(enc.input_ids.to(DEVICE)).pooler_output.cpu().numpy()
#     embeddings[protein_id] = emb[0]  # Shape: (512,)
#     print(f"  {protein_id}: shape {embeddings[protein_id].shape}")

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

# # Upload embeddings to Qdrant
# print("\nUploading embeddings to Qdrant...")
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

# Query: find similar proteins
print("\n" + "="*60)
print("SEARCH TEST")
print("="*60)

query_id = "protein_00001"
query_seq = proteins_df.loc[proteins_df['protein_id'] == query_id, 'sequence'].values[0]
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
