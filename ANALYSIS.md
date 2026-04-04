# Analysis: gLM2 Local Retrieval Pipeline

---

## 1) Summary



1. Parse plasmid genomic files into mixed-modality contig text.
2. Embed contigs with tattabio/gLM2_650M_embed.
3. Store vectors in local Qdrant.
4. Query top-k nearest neighbors.


---

## 2) Timeline

### Clone and baseline verification

2. Baseline inference validated through model loading and hidden-state extraction in main.py.
3. Updated deprecated lines

### Commits

Local commit progression on main branch:

1. 44ac77b: inference working
2. 30d2e58: small demo batch works
3. f606096: larger database-like csv working
4. 50d214e: works on gpu
5. 88b1994: qdrant embedding storage verified (tested with a short 1000 sequence csv)
6. 6c6f29e: moved to RefSeq plasmid flow (BBTools acquisition for fna/gbff)


### Testing

1. qdrant_demo.py changed to batched embedding with token truncation.
2. Memory failures observed (torch.OutOfMemoryError) during embedding on GPU (6 GB).
3. Current script values increased to:
  - BATCH_SIZE = 4
  - MAX_SEQ_LEN = 4096
4. These settings are within safe limits for this hardware profile in many runs.


---

## 3) 

1. Source generation
  - fasta_reader.py parses GBFF + FNA and generates csv with plasmid id and contig string

2. Embedding and indexing
  - qdrant_demo.py loads ref_seq_plasmids.csv.
  - Tokenize with truncation and max_length.
  - Embed with gLM2_650M_embed pooler output.
  - Upsert vectors to Qdrant collection ref_seq_plasmids.

3. Querying
  - use docker to run an instance of qdrant
  - query_qdrant.py embeds query contigs.
  - Sends vector search requests to Qdrant.
  - Prints top-3 results.

---

## 4) Hardware and Software Limitations

### Hardware limits

1. GPU memory: 6 GB total VRAM 
2. Transformer memory cost grows quickly with sequence length.
3. Long context >4096 plus non-trivial batch size causes large intermediate allocations.

Practical implication:

1. MAX_SEQ_LEN contributes heavily to memory usage (can quadruple memory allocation when increased).
2. BATCH_SIZE scales memory approximately linearly.
3. Combined settings BATCH_SIZE=4 and MAX_SEQ_LEN=4096 are stable on 6 GB.
4. GPU VRAM is an issue especially with larger context and larger datasets

### Software and runtime limits

1. Qdrant must be running on localhost:6333 before ingest/query scripts. (can use local version of qdrant but probably slower)
2. CSV parsing over large contig files can be slow on local workstation, and can be interrupted.


---

## 5) Current Known Issues (Updated)

1. Critical: GPU OOM risk remains high at current qdrant_demo.py settings.
2. Medium: Output is console-only; no structured retrieval results file.

---

## 6) Verified Progress to Date

1. Environment and dependency stack assembled, including torch, transformers, qdrant-client, pandas, biopython.
2. Basic model inference path validated in main.py.
3. Genome-to-contig conversion implemented in fasta_reader.py.
4. Qdrant collection creation and vector upsert implemented.
5. Query workflow implemented.


## 7) Current State Summary

The project has successfully moved from base model inference to a near-complete local semantic retrieval pipeline over plasmid contigs. The largest barrier to reliable execution is resource limitations: loading large datasets like refseq for analysis remains slow on current hardware. Not yet implemented other factors into contig (Quadrupia, invertia)

