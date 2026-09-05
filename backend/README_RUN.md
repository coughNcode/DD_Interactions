# DDI-Guard — Backend Quick Start

## Prerequisites
Make sure you are in the `conda` environment `E` and all dependencies are installed.

```bash
conda activate E
```

## Start the server

From the **project root** (`DD_Interactions/`):

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Or equivalently from inside the `backend/` directory:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag watches for source changes and restarts automatically.  
Omit it in production.

## Interactive API docs

Once running, open:

- **Swagger UI** (try endpoints in-browser): http://localhost:8000/docs  
- **ReDoc**:                                  http://localhost:8000/redoc

## Endpoints

| Method | Path       | Description |
|--------|------------|-------------|
| `GET`  | `/health`  | Liveness probe — confirms all models are loaded |
| `GET`  | `/drugs`   | Sorted list of ~1,700 target drug names for the dropdown |
| `POST` | `/analyze` | Full 3-panel DDI analysis |

## Example curl calls

### Health check
```bash
curl http://localhost:8000/health
```

### List drugs (first 5)
```bash
curl -s http://localhost:8000/drugs | python3 -m json.tool | head -20
```

### Analyze: Aspirin + Warfarin (expects clinical evidence = MAJOR)
```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_name":   "Aspirin",
    "candidate_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "target_drug":      "Warfarin"
  }' | python3 -m json.tool
```

Expected `clinical_evidence.severity`: **`"Major"`**  
(Bug 2 fix: "Aspirin" is aliased to "Acetylsalicylic acid" before DDInter lookup.)

### Analyze: Aspirin + Carbamazepine
```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_name":   "Aspirin",
    "candidate_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "target_drug":      "Carbamazepine"
  }' | python3 -m json.tool
```

Expected `clinical_evidence.reason`: **`"found"`** with severity **`"Unknown"`**.

## Startup logs to expect

```
Loading DDI-Guard resources (GraphSAGE, ADMET-AI, DDInter, catalogue)...
GraphSAGE model loaded from .../models/graphsage_mvp.pt
ADMET-AI model loaded.
DDInter loaded: 156858 rows, 1932 unique drug names, ~1970 alias entries.
Drug catalogue loaded: 1705 entries.
All resources loaded. Server ready.
```
