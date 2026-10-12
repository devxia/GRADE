# GRADE

**GPCR-targeted Ranking-guided Agonist Design Engine**

GRADE ranks candidate peptides for predicted agonist activity at a target GPCR. Supply a GPCR amino acid sequence and a peptide list to receive scores sorted from highest to lowest for experimental prioritization.

The **grade-client** Python package provides the `grade` command. GRADE is also available through an independent [Agent Skill](#agent-skill) and the [Cloud API](#cloud-api).

**GRADE-paper** is the model used in the accompanying publication.

## Install

Use Python 3.11 or newer. Clone the [GRADE repository](https://github.com/devxia/GRADE) and install the package in a virtual environment:

```bash
git clone https://github.com/devxia/GRADE.git
cd GRADE

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Quick Start

An email address is required to use the prediction service.

Save your candidates as `candidates.fasta`:

```fasta
>peptide_1
ACDEFGHIKLMNPQRSTVWY
>peptide_2
YTVSRQPNMLKIHGFEDCA
```

Run a prediction against the bundled human CCR5 target:

```bash
grade \
  --target-name CCR5_HUMAN \
  --peptides candidates.fasta \
  --output scores.csv
```

Replace the example peptides with your candidates. For non-interactive use, add `--email user@example.org`. Run `grade --help` for all command options.

### Use your own target

Save the complete amino acid sequence of your target in `target.fasta`, then run:

```bash
grade \
  --target-fasta target.fasta \
  --peptides candidates.fasta \
  --output scores.csv
```

If the target FASTA has several entries, add `--target-name` with the complete header of the desired entry. Each invocation scores one target. You may instead supply its amino acid sequence directly with `--target-seq`.

### Inputs and result files

| Target input | Arguments | Requirement |
|---|---|---|
| Bundled GPCR | `--target-name CCR5_HUMAN` | Exact header in the [bundled FASTA](grade/targets/targets_gpcr_v2.fa) |
| One sequence in a file | `--target-fasta target.fasta` | Exactly one FASTA entry |
| Selected entry in a file | `--target-fasta targets.fa --target-name CCR5_HUMAN` | Exact complete header in the supplied FASTA |
| Direct sequence | `--target-seq` followed by the amino acid sequence | Cannot be combined with other target arguments |

| Peptide format | Extensions | Contents |
|---|---|---|
| FASTA | `.fasta`, `.faa`, `.fa` | Full header becomes the candidate identifier |
| CSV | `.csv` | `sequence`, `peptide` or `seq` column; optional `name`, `peptide_name` or `id` |
| Text | `.txt` | One sequence per line, or whitespace-separated name and sequence |

Sequence whitespace is removed and letters are uppercased. Empty sequences and duplicate identifiers are rejected. Supply amino acid letters without residue numbers or chemical modification notation. Cloud encodes each target and peptide with a 1024-token limit, including special tokens; longer sequences are truncated.

Use the output extension to choose CSV, TSV or JSON. To keep only the highest-scoring candidates:

```bash
grade \
  --target-name CCR5_HUMAN \
  --peptides candidates.fasta \
  --top-k 20 \
  --output top20.json
```

`--output-format` overrides extension-based selection. An unrecognized extension defaults to CSV. The package writes the following fields; JSON output is an array with the same keys:

| Field | Meaning |
|---|---|
| `rank` | Descending-score rank, starting at 1 |
| `target` | Selected target header, or `custom_target` for a direct sequence |
| `peptide_name` | Candidate identifier |
| `src` | Source label; override with `--source-label` |
| `sequence` | Normalized candidate sequence |
| `score` | Positive-class softmax score from 0 to 1 |

Higher scores indicate greater predicted agonist potential for the target GPCR. The accompanying work treats scores above 0.9 as high-confidence candidates for experimental prioritization, not as a mandatory cutoff or calibrated probability of success. Functional assays are needed to establish agonist activity.

Results are checked before the output file is replaced. `grade-inference`, `python -m grade` and `python -m grade.inference` accept the same arguments as `grade`; `--output-csv` remains an alias for `--output`.

### Cloud quota

View the remaining daily prediction quota and reset time:

```bash
grade --usage
```

| Limit | Value |
|---|---|
| Peptides per request | Up to 50 |
| Predictions per email per UTC day | 50 |
| Predictions per client per UTC day | 50, counted independently |
| Effective available quota | Smaller of the two remaining balances |
| Reset time | UTC midnight; 08:00 Beijing time |

The package and Skill share the same quota. Repeated predictions count again, and `--top-k` does not reduce the number submitted. Users sharing a public IP do not share a daily quota.

If the service returns 429, follow the daily reset time or `Retry-After` value. A timed-out request may have consumed quota; check `grade --usage` before retrying.

## Agent Skill

The [GRADE Cloud Skill](skills/grade-cloud/SKILL.md) runs independently of the `grade-client` package. It needs Python and `requests`.

Install with Node.js and access to the GitHub repository:

```bash
npx skills add devxia/GRADE --skill grade-cloud
```

Alternatively, ask your agent:

> Install the grade-cloud skill from https://github.com/devxia/GRADE.

Example request:

> Use GRADE to rank the peptides in candidates.fasta against my GPCR target and save the results to scores.csv.

Provide an email address, the target sequence and candidate peptides. The Skill uses GRADE-paper and the same prediction service as the package.

The bundled script can also run directly from the cloned repository without installing `grade-client`:

```bash
python -m pip install requests
python skills/grade-cloud/scripts/grade_predict.py --help
```

Supply the target sequence with `--target-seq`, the candidate file with `--peptides`, and optionally an output path with `--output`. The script supports CSV, TSV, JSON and Top-K; it prints results to stdout when no output path is given. Its fields are `rank,peptide_name,sequence,score`.

## Use a local model

GRADE uses Cloud resources unless you specify a local model directory with `--model-dir`. Local inference requires no email and has no Cloud quota.

From the cloned `GRADE` directory, install the local inference dependencies:

```bash
python -m pip install ".[local]"
```

Use Python 3.11 with PyTorch 2.4.0 and Transformers 4.44.0. A CUDA-enabled GPU is recommended for larger screens; CPU inference is also supported.

Specify your configured model directory:

```bash
grade \
  --model-dir /path/to/model \
  --target-name CCR5_HUMAN \
  --peptides candidates.fasta \
  --output local_scores.csv
```

Replace `/path/to/model` with your model directory. Local inference accepts the same [inputs and result formats](#inputs-and-result-files). The `src` field defaults to the candidate filename stem locally and `grade-api` for Cloud predictions.

Local requests remain on your machine. Errors do not trigger a fallback to Cloud. `--model-dir` cannot be combined with `--model`, which selects the Cloud model.

### GPU and multi-GPU inference

For CUDA mixed precision:

```bash
grade \
  --model-dir /path/to/model \
  --target-name CCR5_HUMAN \
  --peptides candidates.fasta \
  --output local_scores.csv \
  --batch-size 64 \
  --fp16
```

For four GPUs:

```bash
torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  -m grade.inference \
  --model-dir /path/to/model \
  --target-name CCR5_HUMAN \
  --peptides candidates.fasta \
  --output local_scores.csv \
  --batch-size 64 \
  --fp16
```

Batch size applies per process. Distributed execution requires an explicit local model directory. CPU users should omit mixed-precision flags.

### Local options and resume

| Local-only option | Default | Purpose |
|---|---|---|
| `--batch-size` | 8 | Candidates per batch per process |
| `--num-workers` | 0 | Data-loading workers per process |
| `--max-length` | 1024 | Tokenizer maximum length; longer sequences are truncated |
| `--fp16` / `--bf16` | Off | CUDA mixed precision; choose at most one |
| `--attn-implementation` | `sdpa` | `eager`, `sdpa`, or `flash_attention_2`, subject to runtime support |
| `--checkpoint-interval` | 100 | Save progress every N batches |
| `--no-resume` | Off | Start fresh, replacing prior progress and output |
| `--no-target-cache` | Off | Disable target encoding reuse |

These options require `--model-dir`.

Local runs save progress automatically. Resume with the same output path, model identity and resolved directory, target identifier/sequence, ordered candidate identifiers/sequences, effective source label, maximum length, process count, precision and attention settings. Changing the candidate filename can change its default source label. `--top-k` and output format may change when resuming. Incompatible progress files are rejected without replacing existing results; choose a new output path or explicitly use `--no-resume`.

## Cloud API

Base URL: `https://devxia--grade-web.modal.run`

The package and Skill handle these requests for most users. Direct integrations can use the same protocol:

| Endpoint | Request | Response |
|---|---|---|
| `POST /client/register` | JSON `email` | `client_token`, `client_id`, normalized `email` |
| `POST /usage` | Bearer credential and JSON `email` | `limit`, `remaining`, `reset_at` |
| `POST /predict` | Bearer credential and prediction JSON | Ranked predictions |

Register once with an email, retain the issued credential privately, and send it in subsequent requests as `Authorization: Bearer <client_token>`. Do not put real credentials in published scripts or logs.

Example prediction request for the bundled CCR5_HUMAN sequence:

```json
{
  "email": "user@example.org",
  "model": "GRADE-paper",
  "target_seq": "MDYQVSSPIYDINYYTSEPCQKINVKQIAARLLPPLYSLVFIFGFVGNMLVILILINCKRLKSMTDIYLLNLAISDLFFLLTVPFWAHYAAAQWDFGNTMCQLLTGLYFIGFFSGIFFIILLTIDRYLAVVHAVFALKARTVTFGVVTSVITWVVAVFASLPGIIFTRSQKEGLHYTCSSHFPYSQYQFWKNFQTLKIVILGLVLPLLVMVICYSGILKTLLRCRNEKKRHRAVRLIFTIMIVYFLFWAPYNIVLLLNTFQEFFGLNNCSSSNRLDQAMQVTETLGMTHCCINPIIYAFVGEKFRNYLLVFFQKHIAKRFCKCCSIFQQEAPERASSVYTRSTGEQEISVGL",
  "peptides": [
    {"name": "peptide_1", "sequence": "ACDEFGHIKLMNPQRSTVWY"},
    {"name": "peptide_2", "sequence": "YTVSRQPNMLKIHGFEDCA"}
  ]
}
```

Omitting `model` selects GRADE-paper. Explicit null, empty or unsupported selections are rejected. Optional `format` is `json` (default), `csv` or `tsv`.

JSON responses contain `model`, `target`, `count` and descending-score `results`. Each result has `peptide_name`, `sequence` and `score`. The direct API labels its target `custom_target`; the package retains a locally resolved FASTA target name in its output. Direct CSV/TSV responses contain `rank,target,peptide_name,src,sequence,score`, with `src` set to `grade-api`.

Successful responses include `X-GRADE-Model`, `X-GRADE-Quota-Limit`, `X-GRADE-Quota-Remaining` and `X-GRADE-Quota-Reset`. The [Cloud quota](#cloud-quota) applies to every access method.

| Status | Meaning |
|---|---|
| 401 | Missing or invalid client credential |
| 422 | Invalid email, target, candidate list, request size, model or format |
| 429 | Daily quota or short-term frequency limit; follow reset/`Retry-After` |
| 500/503 | Server failure or uncertain outcome; check usage before retrying |

## Citation

If you use GRADE in your research, please cite:

> [Authors]. [Article title]. *[Journal]* **[Volume]**, [Pages] ([Year]). DOI: [DOI].

## License

The source code is licensed under the [MIT License](LICENSE).
