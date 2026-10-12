---
name: grade-cloud
description: >
  Rank peptide candidates for predicted GPCR agonist activity via the GRADE cloud
  API. Use when the user wants to score peptides against a GPCR target or
  prioritize candidates for experimental validation. Uses an email address and
  an automatically managed client credential.
---

# GRADE API — GPCR Agonist Candidate Ranking

## Overview

GRADE (GPCR-targeted Ranking-guided Agonist Design Engine) scores candidate peptides against a GPCR target for agonist prioritization. Given a target sequence and peptide sequences, it returns scores from 0 to 1, sorted from highest to lowest. Higher scores indicate greater predicted agonist potential, not confirmed binding or activation.

The model uses two protein sequence encoders and a bidirectional cross-attention module (IFM). The API runs on serverless GPU infrastructure — no local inference installation or model download required.

## First-use Guidance

1. Use an email address already supplied by the user, `GRADE_EMAIL`, or the client's saved email. Let the script handle its private configuration; do not open credential files. If an email is needed, ask exactly:

   > Please enter your email address:

   Add no explanation of purpose or verification status to this question. Pass the supplied address with `--email`. Do not ask for an API key, password, or client token.
2. The script automatically registers a client on first use and saves its credential in a private user configuration file. Do not print, copy, read, or reset the credential yourself. Do not remove configuration or change email to bypass quotas. A saved invalid credential stops the client; it does not create a replacement automatically.
3. Use `GRADE-paper` by default. It is the frozen paper model and the only supported choice. `GRADE-base` and `GRADE-medium` are future versions, not available selections or aliases. If the user explicitly requests one, explain its unavailability and ask for an available choice; do not silently substitute paper.

## Prerequisites

The script is `skills/grade-cloud/scripts/grade_predict.py` in this repository. This Skill directory runs independently: do not require or install `grade-client` to use it, and do not replace the script with the package CLI. Submitting requests requires `requests`; `--help` and local argument/file validation do not import it. If needed, install it in the user's chosen environment:

```bash
pip install requests
```

The script and the package Cloud client reuse the same user configuration for the same service. Email precedence is `--email`, `GRADE_EMAIL`, saved email, then terminal input. Surrounding whitespace is stripped and the address is lowercased; dots and `+tags` are preserved. In a non-interactive session, missing email fails immediately without waiting for input. Empty input, EOF or Ctrl-C stops submission.

## Quick Start

```bash
python skills/grade-cloud/scripts/grade_predict.py \
  --email user@example.org \
  --target-seq MRYRLAWLLHPALPSTFRSVLGARLPPPERLCGFQKKTYSKMNNPAI \
  --peptides candidates.fasta \
  --output results.csv
```

Omit `--email` to use the environment, saved email or interactive prompt. The script checks prediction inputs and output destination before prompting or making network requests.

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

On a fresh installation, `--usage` needs an email and automatically registers a client. Registration and quota queries do not consume prediction quota.

## Script Parameters

| Parameter | Required | Description |
|---|---|---|
| `--email` | No | Email address; otherwise use environment, saved email or terminal input |
| `--usage` | No | Show remaining quota; cannot combine with prediction/output arguments |
| `--model` | No | Default and only supported value: `GRADE-paper` |
| `--target-seq` | For prediction | Target GPCR amino acid sequence |
| `--peptides` | For prediction | Peptide input file |
| `--output`, `-o` | No | Output path; default prints results to stdout |
| `--output-format` | No | `csv`, `tsv` or `json`; otherwise inferred from output extension |
| `--top-k` | No | Retain the highest K results; positive integer |

`--api-key` is retired and reports an upgrade error. `GRADE_API_KEY` is not used. Existing positional `call_api(api_key, ...)` Python calls are replaced by `call_api(target_seq, peptides, *, email, client_token, model='GRADE-paper')`; prefer the CLI so it manages credentials automatically.

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
| `POST /client/register` | JSON `email` | Automatically issued `client_token`, `client_id`, normalized `email` |
| `POST /usage` | Bearer credential; JSON `email` | `limit`, `remaining`, `reset_at` |
| `POST /predict` | Bearer credential; JSON shown below | Descending predictions |

Only the client handles the returned credential. It sends it in `Authorization: Bearer <client_token>`; never include a real credential in examples, chat, logs or result files. Registration and other requests are not automatically retried or redirected.

Prediction request:

```json
{
  "email": "user@example.org",
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
| `401` | Report invalid client credential; do not reset configuration or register a new identity automatically |
| `422` | Report the API message; check email, model, sequence, duplicate names and request size |
| `429` | For daily quota use the reset time; for short-term throttling use `Retry-After` |
| `500`/`503` | Report the error; an unknown outcome may have consumed quota, so check `--usage` before retrying |

Invalid prediction inputs do not consume daily quota. A definite inference failure refunds the request. Timeouts and transport failures can leave reserved quota because execution may still be running. Do not automatically resubmit. Cold starts can add latency.

## Interpreting Results

The `score` field is the positive-class softmax score from 0 to 1. It is primarily for ranking peptides for predicted agonist potential at the specified GPCR: higher scores indicate stronger model support for agonist activity. In the accompanying work, scores above 0.9 are treated as high-confidence candidates for experimental prioritization. This is not a calibrated probability of experimental success, nor proof of binding or agonist activity. Do not assign interaction or non-interaction labels using unsupported score cutoffs.

Start with a peptide library, score it within service limits, prioritize high-ranked candidates, and verify agonist activity experimentally with appropriate functional assays.
