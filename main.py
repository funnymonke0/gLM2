import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32 # Fix assertion error due to deprecation. Use bfloat16 on GPU for faster inference, and float32 on CPU for compatibility.

model = AutoModelForMaskedLM.from_pretrained('tattabio/gLM2_650M', dtype=MODEL_DTYPE, trust_remote_code=True).to(DEVICE).glm2
tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M', trust_remote_code=True)

# A contig with two proteins and an inter-genic sequence.
# NOTE: Nucleotides should always be lowercase, and prepended with `<+>`.
sequence = "<+>MALTKVEKRNRIKRRVRGKISGTQASPRLSVYKSNK<+>aatttaaggaa<->MLGIDNIERVKPGGLELVDRLVAVNRVTKVTKGGRAFGFSAIVVVGNED"

# Tokenize the sequence.
encodings = tokenizer([sequence], return_tensors='pt')
# Extract embeddings.
with torch.no_grad():
    embeddings = model(encodings.input_ids.to(DEVICE), output_hidden_states=True).last_hidden_state

# Check the output.
assert embeddings.ndim == 3, "Expected 3D tensor (batch, seq_len, hidden_dim)"
assert embeddings.shape[0] == 1
assert not torch.isnan(embeddings).any(), "NaNs in output!"
assert not torch.isinf(embeddings).any(), "Infs in output!"
print(f"OK — embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")