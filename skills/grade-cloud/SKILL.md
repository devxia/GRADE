---
name: grade-cloud
description: >
  Rank peptide candidates for predicted GPCR agonist activity via the GRADE cloud
  API. Use when the user wants to score peptides against a GPCR target or
  prioritize candidates for experimental validation. Requires a valid GRADE API key.
---

# GRADE API — GPCR Agonist Candidate Ranking

## Overview

GRADE (GPCR-targeted Ranking-guided Agonist Design Engine) scores candidate peptides against a GPCR target for agonist prioritization. Given a target sequence and peptide sequences, it returns scores from 0 to 1, sorted from highest to lowest. Higher scores indicate greater predicted agonist potential, not confirmed binding or activation.

The model uses two protein sequence encoders and a bidirectional cross-attention module (IFM). The API runs on serverless GPU infrastructure — no local installation or model download required.

## First-use Guidance

1. Ask the user to supply a valid API key through the script's hidden terminal prompt (recommended) or the `GRADE_API_KEY` environment variable. Do not ask them to paste secrets into chat or write them to a file. Do not read credential files, save keys, echo keys, or include them in logs or generated examples. An already available key does not need to be requested again.
2. If the user has no key, show this English guidance and stop before calling the API:

   > If you do not have an API key, request one from the authors at xiabb@im.ac.cn.

3. Ask which model they want to use, explaining that **GRADE-paper** is the paper's frozen model and the current default and only available choice. Use an explicit supported choice. If they do not specify a model or skip the question, proceed with `GRADE-paper` without blocking on another confirmation. `GRADE-base` and `GRADE-medium` are future versions, currently unavailable; they are not registered models or aliases. If explicitly requested, explain their unavailability and ask for an available choice; do not silently substitute paper.

## Prerequisites

The script is at `skills/grade-cloud/scripts/grade_predict.py` in this repository. Submitting requests requires the `requests` package; `--help` and local argument/file validation do not import it. If needed, install it in the user's chosen environment:

```bash
pip install requests
```

Key resolution uses the first non-empty value in this order: `--api-key`, `GRADE_API_KEY`, hidden interactive terminal input. Surrounding whitespace is stripped. The script never saves the key. Prefer hidden input or a key already supplied through the environment over putting a secret in command-line history. In a non-interactive session, a missing key fails immediately with the request email and input options. Empty input, EOF, Ctrl-C, or unavailable hidden input stop submission; there is no plain-text fallback.

## Quick Start

Run this in an interactive terminal. It uses the default `GRADE-paper` model and prompts for a hidden key unless `GRADE_API_KEY` is set:

```bash
python skills/grade-cloud/scripts/grade_predict.py \
  --target-seq MRYRLAWLLHPALPSTFRSVLGARLPPPERLCGFQKKTYSKMNNPAI \
  --peptides candidates.fasta \
  --output results.csv
```

## Using the Script

The CLI handles file loading, request submission, and output formatting. These examples use hidden input or the environment variable and contain no key value.

### Explicit model with CSV input and top-k

```bash
python skills/grade-cloud/scripts/grade_predict.py \
  --model GRADE-paper \
  --target-seq MRYRLAWLLHPALPSTFRSVLGARLPPPERLCGFQKKTYSKMNNPAI \
  --peptides peptides.csv \
  --top-k 20 \
  --output top20.csv
```

### Print to stdout (no output file)

```bash
python skills/grade-cloud/scripts/grade_predict.py \
  --target-seq MRYRLAWLLHPALPSTFRSVLGARLPPPERLCGFQKKTYSKMNNPAI \
  --peptides candidates.fasta
```

### JSON or TSV output

```bash
python skills/grade-cloud/scripts/grade_predict.py \
  --target-seq MRYRLAWLLHPALPSTFRSVLGARLPPPERLCGFQKKTYSKMNNPAI \
  --peptides candidates.fasta \
  --output results.json
```

Use `--output results.tsv` for TSV, or `--output-format` to select a format explicitly.

### Script Parameters

| Parameter | Required | Description |
|---|---|---|
| `--api-key` | No | Explicit key; otherwise use `GRADE_API_KEY`, then hidden terminal input. A valid key is required for submission. |
| `--model` | No | Default and only supported value: `GRADE-paper` |
| `--target-seq` | Yes | Target GPCR amino acid sequence |
| `--peptides` | Yes | Peptide file (`.fasta`/`.faa`/`.fa`/`.csv`/`.txt`) |
| `--output` | No | Output path (default: print to stdout). Extension selects format. |
| `--output-format` | No | `csv` / `tsv` / `json` (default: inferred from `--output`, otherwise csv) |
| `--top-k` | No | Keep only top K results; must be a positive integer |

### Input File Formats

| Format | Extensions | Notes |
|---|---|---|
| FASTA | `.fasta` `.faa` `.fa` | Header → name, body → sequence |
| CSV | `.csv` | Requires `sequence`/`peptide`/`seq` column; optional `name`/`peptide_name`/`id` |
| TXT | `.txt` | One per line; single field = sequence, multi-field = first=name, last=sequence |

### Output Format

The script writes CSV by default, or TSV/JSON when `--output` ends in `.tsv`/`.json` (or `--output-format` is set). All formats contain:

| Column | Description |
|---|---|
| `rank` | 1-based rank (sorted by score descending) |
| `peptide_name` | Peptide identifier |
| `sequence` | Peptide amino acid sequence |
| `score` | Positive-class softmax score (0–1); higher scores indicate greater predicted agonist potential |

## API Reference

### Endpoint

```
POST https://devxia--grade-web.modal.run/predict
```

### Request Body

The script always submits `model: "GRADE-paper"`. The key placeholder below describes the in-memory request shape; do not save a populated request body to a file.

```json
{
  "api_key": "<API key supplied at runtime>",
  "model": "GRADE-paper",
  "target_seq": "MRYRLAWLLH...",
  "peptides": [
    {"name": "p1", "sequence": "ACDEF..."},
    {"name": "p2", "sequence": "KLMNP..."}
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `api_key` | string | Yes | Your API key, supplied at runtime |
| `model` | string | Always sent by this client | `GRADE-paper`; default and only supported model |
| `target_seq` | string | Yes | Target GPCR amino acid sequence |
| `peptides` | array | Yes | Peptide objects accepted by the hosted API |
| `peptides[].name` | string | No | Peptide name (auto-generated if omitted) |
| `peptides[].sequence` | string | Yes | Peptide amino acid sequence |
| `format` | string | No | `json` (default), `csv`, or `tsv`; the script requests JSON and formats output locally |

### Response (200 OK)

Example JSON response with model identity (illustrative scores):

```json
{
  "target": "custom_target",
  "model": "GRADE-paper",
  "results": [
    {"peptide_name": "p1", "sequence": "ACDEF...", "score": 0.8732}
  ],
  "count": 1
}
```

The client checks the JSON `model` and, if present, the `X-GRADE-Model` response header. Any reported identity must be a non-empty string exactly matching the requested model, and JSON/header identities must agree. A malformed identity, explicit mismatch, or conflicting identities stops the client before writing results or overwriting an output file. When identity is reported consistently, stderr reports `Server reported model: GRADE-paper`.

For legacy paper services only, if both identities are absent, the client continues and prints exactly:

```text
Warning: server did not report model identity; cannot verify requested model.
```

Do not describe that legacy response as verified. This compatibility behavior does not permit an explicitly invalid, null, empty, or different identity. Only `GRADE-paper` can be requested, so it cannot be used to claim that a future model was served.

`call_api(api_key, target_seq, peptides, model='GRADE-paper')` continues to return the `results` list; existing three-argument Python calls remain supported. Model diagnostics go to stderr, and local CSV/TSV/JSON result columns remain unchanged.

When `format` is `csv` or `tsv` in a direct API request, the response body is the ranked table in that format with `Content-Type: text/csv` or `text/tab-separated-values`. The client uses the JSON response for identity checks before producing its own output.

### Error Codes

| Status | Meaning | Action |
|---|---|---|
| 401 | Invalid API key | Check that the key is valid. If you do not have an API key, request one from the authors at xiabb@im.ac.cn. |
| 422 | Request rejected by validation or current request policy | Check the model, field names, duplicate names, non-empty sequences, and the API-provided message |
| 429 | Cloud-side usage/rate limit reached | Read the API-provided message; retry later, reduce the request, or contact the authors for higher throughput |
| 500 | Server error | Retry after a moment; if persistent, contact the authors |

## Interpreting Results

### Score Meaning

The `score` field is the positive-class softmax score from 0 to 1. It is primarily for ranking peptides for predicted agonist potential at the specified GPCR: higher scores indicate stronger model support for agonist activity. In the accompanying work, scores above 0.9 are treated as high-confidence candidates for experimental prioritization. This is not a calibrated probability of experimental success, nor proof of binding or agonist activity. Do not assign interaction or non-interaction labels using unsupported score cutoffs.

### Sorting

Results are sorted by `score` descending, so the first peptide has the highest predicted agonist potential for the target GPCR.

### Recommended Workflow

1. Start with a peptide library (e.g., natural or generated peptides).
2. Run GRADE to score the candidates within the hosted service limits.
3. Prioritize top-ranked peptides; scores above 0.9 can be treated as high-confidence candidates, not a mandatory selection threshold.
4. Verify agonist activity experimentally with appropriate functional assays.

## Cloud-side Limits and Errors

The public hosted endpoint currently limits usage to control serverless GPU compute cost: up to 50 peptides per request and 50 peptides per API key per UTC day. These are cloud-side policy values and may change over time, so the client script does not hard-code or pre-enforce them; it submits the request and reports the API's response.

If a request exceeds the current policy, the API returns an explanatory error, usually:

| Status | How to explain it to the user |
|---|---|
| 422 | The request shape or current request policy rejected the input. Quote the API message and suggest checking file format, duplicate names, empty sequences, or request size. Explain that request-size limits exist to control GPU compute cost. |
| 429 | A cloud-side usage/rate limit was reached. Quote the API message and explain that the hosted reader service is rate-limited to control GPU compute cost. Suggest retrying later, reducing the request, or asking the GRADE authors for higher throughput. |

Cold starts may add noticeable latency on the first request after the service has been idle. Warm-request latency depends on request size and current cloud capacity.
