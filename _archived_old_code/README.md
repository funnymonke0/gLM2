# Archive: Old Code (_archived_old_code)

This folder contains superseded files from the old Qdrant-based gaia_local system.

## Contents:

- **gaia_local/** - Old module system (replaced by glm2_search.py)
  - CLI, pipeline, embedder, Qdrant integration (all replaced)
  
- **seqhub_local.py** - Old main script (replaced by glm2_search.py)

- **search_results*.json** - Old test results (superseded by newer runs)

- **test_results_100.json** - Earlier validation test

## Recovery:

If needed, any file can be recovered with:
\\\powershell
Move-Item _archived_old_code/<filename> .
\\\

Or restore the entire directory:
\\\powershell
Move-Item _archived_old_code/gaia_local .
\\\

---
Cleanup performed: 2026-05-09
Reason: Transitioned from Qdrant+gaia_local to simplified glm2_search.py system
