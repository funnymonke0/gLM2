import torch
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import numpy as np
import datasets
from itertools import chain
from huggingface_hub import login

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32

## loads OMG plasmid data, generates gLM2_650M_embed embeddings, and uploads to local Qdrant instance for retrieval testing in query_qdrant.py
class QdrantLoader:
    def __init__(self):
        # Setup
        with open("token.txt", "r") as f:
            token = f.read().strip()  # Load Hugging Face token from file
        login(token=token)

        # Load gLM2_embed model (for retrieval, not base gLM2)
        print("Loading gLM2_650M_embed model...")
        self.model = AutoModel.from_pretrained('tattabio/gLM2_650M_embed', dtype=MODEL_DTYPE, trust_remote_code=True).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M_embed', trust_remote_code=True)

        # Connect to Qdrant (running in Docker)
        print("Connecting to Qdrant on localhost:6333...")
        self.client = QdrantClient(url="http://localhost:6333")
        # Load plasmid data
        ds = datasets.load_dataset('tattabio/OMG', streaming=True)["train"]
        self.dataset = next(iter(ds))



    def embed(self, tokenized, model = None, device=DEVICE):
        if model is None:
            model = self.model
        
        embeddings = []
        with torch.no_grad():
            embeddings = model(
                input_ids=torch.from_numpy(tokenized['input_ids']).to(device=device, dtype=torch.long),
                attention_mask=torch.from_numpy(tokenized['attention_mask'],).to(device)
            ).pooler_output.float().cpu().tolist()
            
        return embeddings


    def load_to_qdrant(self, records, embeddings):
        # Create Qdrant collection
        collection_name = "OMG_dataset"
        print(f"\nCreating Qdrant collection '{collection_name}'...")

        # remove old test
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

        # Create with vector config
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
        print("Collection created!")

        # # Upload embeddings to Qdrant
        print("\nUploading embeddings to Qdrant...")
        if isinstance(records, dict):
            records = [records]

        def _scaffold_id(record, fallback_idx):
            # OMG rows expose CDS_ids / IGS_ids
            ids = record.get("CDS_ids") or record.get("IGS_ids") or []
            if ids:
                id_parts = ids[0].split("|")
                if len(id_parts) >= 2:
                    return f"{id_parts[0]}|{id_parts[1]}"
                return ids[0]
            return f"scaffold_{fallback_idx}"

        points = [
            PointStruct(
                id=idx,
                vector=emb,
                payload={
                    "scaffold_id": _scaffold_id(record, idx),
                    "cds_count": len(record.get("CDS_seqs", [])),
                    "igs_count": len(record.get("IGS_seqs", [])),
                    "cds_ids": record.get("CDS_ids", []),
                    "igs_ids": record.get("IGS_ids", []),
                }
            )
            for idx, (record, emb) in enumerate(zip(records, embeddings))
        ]

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )
        print("Embeddings uploaded to Qdrant!")



    # this is basically directly from https://github.com/TattaBio/OMG/blob/main/scripts/tokenize.py 
    def tokenize_function(self, dataset = None, tokenizer=None, max_seq_length: int = 4096):
        """Tokenization for mixed modality data."""
        if dataset is None:
            dataset = self.dataset
        if tokenizer is None:
            tokenizer = self.tokenizer

        # Get CDS features.
        cds_seqs = dataset["CDS_seqs"]
        cds_positions = dataset["CDS_position_ids"]
        cds_orientations = dataset["CDS_orientations"]

        # Get IGS features.
        igs_seqs = dataset["IGS_seqs"]
        igs_positions = dataset["IGS_position_ids"]

        # NOTE: This assumes the tokenizer handles all nucleotide sequences as lower-case.
        # This is to avoid collisions with amino-acids A, T, C, G (for example Glycine).
        igs_seqs = [seq.lower() for seq in igs_seqs]

        # Add orientation tokens.
        orientation_map = {True: "<+>", False: "<->"}
        cds_seqs = [
            f"{orientation_map[orient]}{seq}"
            for orient, seq in zip(cds_orientations, cds_seqs)
        ]

        # Tokenize the CDS and IGS sequences.
        tokenized_cds = tokenizer(cds_seqs)
        tokenized_igs = (
            tokenizer(igs_seqs) if igs_seqs else {k: [] for k in tokenized_cds.keys()}
        )

        def _interleave(cds_elems, igs_elems):
            """Interleave the cds and igs elements based on the position ids."""
            num_elems = len(cds_elems) + len(igs_elems)
            elems = [None] * num_elems
            for i, elem in zip(cds_positions, cds_elems):
                elems[i] = elem
            for i, elem in zip(igs_positions, igs_elems):
                elems[i] = elem
            return [elem for elem in elems if elem is not None]

        input_ids = _interleave(tokenized_cds["input_ids"], tokenized_igs["input_ids"])
        attention_mask = _interleave(
            tokenized_cds["attention_mask"], tokenized_igs["attention_mask"]
        )

        # Flatten the sequence elements
        input_ids = np.array(list(chain(*input_ids)), dtype=np.int64)
        attention_mask = np.array(list(chain(*attention_mask)), dtype=bool)

        # Pad to multiple of max_seq_length
        pad_length = max_seq_length - len(input_ids) % max_seq_length
        input_ids = np.pad(
            input_ids, (0, pad_length), constant_values=tokenizer.pad_token_id
        )
        attention_mask = np.pad(attention_mask, (0, pad_length), constant_values=0)

        # Reshape to (num_examples, max_seq_length)
        input_ids = input_ids.reshape(-1, max_seq_length)
        attention_mask = attention_mask.reshape(-1, max_seq_length)

        tokenized_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return tokenized_data
    

def main():
    loader = QdrantLoader()
    tokenized = loader.tokenize_function()
    embeddings = loader.embed(tokenized)
    loader.load_to_qdrant(loader.dataset, embeddings)


if __name__ == "__main__":
    main()