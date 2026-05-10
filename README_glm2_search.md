# GLM2 Sequence Search

A streamlined sequence similarity search tool using GLM2 embeddings. Embed protein sequences, compute cosine similarity, and rank matches.

## Setup

```bash
# Install dependencies in venv
pip install numpy torch transformers biopython

# Or use existing .venv if available
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\Activate.ps1  # Windows
```

## Usage

### Basic search: Query sequences against a corpus

```bash
python glm2_search.py <corpus.fasta> <queries.fasta> --top-k 10 --output results.json
```

**Arguments:**
- `corpus_fasta` – Reference sequences to search against
- `query_fasta` – Query sequences to find matches for
- `--top-k` – Number of top matches to return per query (default: 10)
- `--output` – Save results to JSON file (omit to print to stdout)
- `--model` – GLM2 model name (default: `tattabio/gLM2_650M_embed`)
- `--batch-size` – Embedding batch size; auto-reduces on OOM (default: 32)

### Example

```bash
# Find matches for P02981 in seqhub corpus
python glm2_search.py seqhub_matches.fa P02981.fasta --top-k 10 --output search_results.json

# View results
cat search_results.json | python -m json.tool
```

## Output Format

Results are written as JSON with the following structure:

```json
{
  "model": "tattabio/gLM2_650M_embed",
  "corpus_file": "seqhub_matches.fa",
  "corpus_count": 101,
  "query_file": "P02981.fasta",
  "query_count": 1,
  "top_k": 10,
  "results": [
    {
      "query_id": "sp|P02981|TCR3_ECOLX",
      "query_description": "...",
      "query_length": 396,
      "matches": [
        {
          "rank": 1,
          "corpus_id": "...",
          "corpus_description": "...",
          "similarity_score": 0.9999,
          "corpus_sequence": "..."
        }
      ]
    }
  ]
}
```

## Features

- **Automatic batch-size reduction** on CUDA out-of-memory errors
- **CPU and GPU support** – Uses CUDA if available, falls back to CPU
- **Remote code trust** – gLM2 model requires `trust_remote_code=True`
- **Minimal dependencies** – Only torch, transformers, biopython, numpy

## How It Works

1. **Load sequences** – Parse FASTA files into memory
2. **Embed corpus** – Generate GLM2 embeddings for all corpus sequences (batched with OOM handling)
3. **Embed queries** – Generate GLM2 embeddings for all query sequences
4. **Compute similarity** – Row-wise cosine similarity between query and corpus embeddings
5. **Rank and output** – Return top-k matches per query sorted by similarity score

## Performance Notes

- First run downloads the GLM2 model (~2-3 GB)
- Embedding is the bottleneck; GPU acceleration is strongly recommended
- Batch size auto-reduces on OOM; disable with `--no-auto-batch-size` if using older PyTorch
