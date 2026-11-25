---
name: grade-cloud
metadata:
  version: "1.3.0"
description: >
  Rank peptide candidates for predicted GPCR agonist activity via the GRADE cloud
  API. Use when the user wants to score peptides against a GPCR target or
  prioritize candidates for experimental validation. Uses a verified GRADE login.
---

# GRADE API — GPCR Agonist Candidate Ranking

## Overview

GRADE (GPCR-targeted Ranking-guided Agonist Design Engine) scores candidate peptides against a GPCR target for agonist prioritization. Given a target sequence and peptide sequences, it returns scores from 0 to 1, sorted from highest to lowest. Higher scores indicate greater predicted agonist potential, not confirmed binding or activation.

The model uses two protein sequence encoders and a bidirectional cross-attention module (IFM). The API runs on serverless GPU infrastructure — no local inference installation or model download required.

## First-use Guidance

1. Use `python skills/grade-cloud/scripts/grade_predict.py whoami` to check for a verified session. When login is needed, run the same script with `login --no-browser` and show the user the verification URL and device code. Ask the user to open the site, verify their email there, and approve the device.
2. Keep the login command running until it completes. The script manages the private session and shares it with the GRADE package. Use the script to manage the session and keep credentials out of chat, logs, and output.
3. Use `GRADE-paper` by default. It is the frozen paper model and the only supported choice. `GRADE-base` and `GRADE-medium` are future versions, not available selections or aliases. If the user explicitly requests one, explain its unavailability and ask for an available choice; do not silently substitute paper.

## Prerequisites

The script is `skills/grade-cloud/scripts/grade_predict.py` in this repository. This Skill directory runs independently: do not require or install `grade-client` to use it, and do not replace the script with the package CLI. Submitting requests requires `requests`; `--help` and local argument/file validation do not import it. If needed, install it in the user's chosen environment:

```bash
pip install requests
```

The script and package share a verified session for the same service. Sessions last up to 14 days. Use `logout` to revoke the CLI session. Non-interactive predictions require an existing session.

## Quick Start

```bash
python skills/grade-cloud/scripts/grade_predict.py login

python skills/grade-cloud/scripts/grade_predict.py \
  --target-seq MRYRLAWLLHPALPSTFRSVLGARLPPPERLCGFQKKTYSKMNNPAI \
  --peptides candidates.fasta \
  --output results.csv
```

The script checks prediction inputs and the output destination before calling Cloud.

For top-k JSON output:

```bash
python skills/grade-cloud/scripts/grade_predict.py \
  --model GRADE-paper \
  --target-seq MRYRLAWLLHPALPSTFRSVLGARLPPPERLCGFQKKTYSKMNNPAI \
  --peptides candidates.fasta \
  --top-k 20 \
  --output top20.json
```

Query quota without prediction inputs:

```bash
python skills/grade-cloud/scripts/grade_predict.py --usage
```

Quota queries require a verified session and do not consume prediction quota.

## Script Parameters

| Parameter | Required | Description |
|---|---|---|
| `--email` | No | Expected verified account email |
| `--usage` | No | Show remaining quota; cannot combine with prediction/output arguments |
| `--model` | No | Default and only supported value: `GRADE-paper` |
| `--target-seq` | For prediction | Target GPCR amino acid sequence |
| `--peptides` | For prediction | Peptide input file |
| `--output`, `-o` | No | Output path; default prints results to stdout |
| `--output-format` | No | `csv`, `tsv` or `json`; otherwise inferred from output extension |
| `--top-k` | No | Retain the highest K results; positive integer |

Use `login` for access if you previously used an API key or automatic client registration.

## Inputs and Outputs

| Input format | Extensions | Notes |
|---|---|---|
| FASTA | `.fasta`, `.faa`, `.fa` | Full header is the peptide name |
| CSV | `.csv` | `sequence`/`peptide`/`seq`; optional `name`/`peptide_name`/`id` |
| TXT | `.txt` | One sequence per line, or name followed by sequence |

Sequences are stripped of whitespace and uppercased. Empty sequences and duplicate names are rejected. Results contain `rank`, `peptide_name`, `sequence` and `score`, sorted descending. `--top-k` filters returned results, so it does not reduce submitted peptide count or quota consumption. Diagnostics and remaining quota go to stderr, leaving stdout available for result data. `--usage` prints its quota summary to stdout.

## API Contract

Base URL: `https://devxia--grade-web.modal.run`

| Route | Request | Result |
|---|---|---|
| `POST /usage` | Bearer access token and JSON `{}` | `limit`, `remaining`, `reset_at` |
| `POST /predict` | Bearer access token and prediction JSON | Descending predictions |

The script sends `Authorization: Bearer <access_token>`. Cloud gets the verified email from the session.

Prediction request:

```json
{
  "model": "GRADE-paper",
  "target_seq": "MRYRLAWLLH...",
  "peptides": [
    {"name": "p1", "sequence": "ACDEF..."},
    {"name": "p2", "sequence": "KLMNP..."}
  ]
}
```

Optional `format` is `json` (default), `csv` or `tsv`. The script requests JSON and formats output locally. The JSON response contains `model`, `target`, `results` and `count`. Each result has `peptide_name`, `sequence` and `score`.

Successful responses include `X-GRADE-Model`, `X-GRADE-Quota-Limit`, `X-GRADE-Quota-Remaining` and `X-GRADE-Quota-Reset`. CSV/TSV direct responses retain the columns `rank,target,peptide_name,src,sequence,score`.

The client checks JSON and header model identities. Any present identity must exactly match the requested model, and JSON/header identities must agree. A malformed or conflicting identity stops output without overwriting an existing result. For a legacy paper response missing both identities, the client prints:

```text
Warning: server did not report model identity; cannot verify requested model.
```

Do not describe that response as verified. This model-response compatibility does not restore retired Key authentication.

## Quota and Errors

The service accepts up to 50 peptides per request. Each email and each client independently has 50 predictions per UTC day, shared across models. Effective remaining quota is the smaller of those two balances. UTC midnight is 08:00 in Beijing. One peptide against two targets counts twice. Different users behind a shared public IP do not share a daily quota; IP limits only short request bursts.

| Status | Action |
|---|---|
| `401` | The session is missing, expired or revoked. Run `login` again. |
| `422` | Report the API message and check the model, sequences, duplicate names and request size. |
| `429` | For daily quota use the reset time; for short-term throttling use `Retry-After` |
| `500`/`503` | Report the error; an unknown outcome may have consumed quota, so check `--usage` before retrying |

Invalid prediction inputs do not consume daily quota. A definite inference failure refunds the request. Timeouts and transport failures can leave reserved quota because execution may still be running. Do not automatically resubmit. Cold starts can add latency.

## Interpreting Results

The `score` field is the positive-class softmax score from 0 to 1. It is primarily for ranking peptides for predicted agonist potential at the specified GPCR: higher scores indicate stronger model support for agonist activity. In the accompanying work, scores above 0.9 are treated as high-confidence candidates for experimental prioritization. This is not a calibrated probability of experimental success, nor proof of binding or agonist activity. Do not assign interaction or non-interaction labels using unsupported score cutoffs.

Start with a peptide library, score it within service limits, prioritize high-ranked candidates, and verify agonist activity experimentally with appropriate functional assays.
