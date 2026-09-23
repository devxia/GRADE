# GRADE

**GPCR-targeted Ranking-guided Agonist Design Engine**

GRADE accompanies the academic work on ranking peptide candidates for GPCR targets. Given a target GPCR sequence and candidate peptide sequences, it uses a cross-attention model to score each pair and return a ranked list for experimental prioritization.

This guide explains how to use GRADE through the hosted Cloud API, an Agent Skill, or local inference. **GRADE-paper** is the model used in the publication and the currently supported model selection. GRADE-base and GRADE-medium are planned follow-up models; they are not available through these interfaces.

Scores support ranking peptides by predicted agonist potential for a GPCR target.

## Cloud API

The hosted API runs GRADE-paper without a local inference installation. **If you do not have an API key, request one from the authors at xiabb@im.ac.cn.**

Endpoint: `POST https://devxia--grade-web.modal.run/predict`

Send a JSON request with your API key, target sequence, and candidate list. The key below is a placeholder:

```bash
curl -i -X POST https://devxia--grade-web.modal.run/predict \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "YOUR_API_KEY",
    "model": "GRADE-paper",
    "target_seq": "MRYRLAWLLHPALPSTFRSVLGARLPPPERLCGFQKKTYSKMNNPAI",
    "peptides": [
      {"name": "peptide_1", "sequence": "ACDEFGHIKLMNPQRSTVWY"},
      {"name": "peptide_2", "sequence": "YTVSRQPNMLKIHGFEDCA"}
    ]
  }'
```

| Request field | Required | Description |
|---|---|---|
| `api_key` | Yes | Your API key |
| `target_seq` | Yes | GPCR amino acid sequence |
| `peptides` | Yes | Array of objects with `name` and `sequence` |
| `model` | No | Defaults to `GRADE-paper`, currently the only supported selection |
| `format` | No | `json` (default), `csv`, or `tsv` |

The default JSON response contains `model`, `target`, `count`, and a descending-score `results` array. Each result contains `peptide_name`, `sequence`, and `score`. This response envelope differs from the local JSON result array.

Add `"format": "csv"` or `"format": "tsv"` to the request to receive a table with `rank`, `target`, `peptide_name`, `src`, `sequence`, and `score` fields; its `src` is `grade-api`. Successful responses include `X-GRADE-Model: GRADE-paper`.

The hosted service currently allows **50 peptides per request and 50 peptides per API key per UTC day**. Contact the authors if your work requires a larger screen.

- **401:** the supplied key is not authorized; check it or request a key at the address above.
- **422:** invalid input, including a missing/empty key or an unsupported model. An omitted model uses GRADE-paper; explicit `null` or an empty model does not.
- **429:** the daily quota has been reached.

## Agent Skill and standalone client

Install the [GRADE Cloud Skill](skills/grade-cloud/SKILL.md) using either of the following methods once the `devxia/GRADE` GitHub repository is published.

**Using the command line:**

```bash
npx skills add devxia/GRADE
```

**Asking your agent directly:**

> Help me install the Agent Skill from the `skills/` directory of the GitHub repository `devxia/GRADE`.

Once installed, ask your agent to rank your candidate peptides against a target GPCR using GRADE. The skill guides model selection, prediction, and interpretation; the default model is GRADE-paper.

You can also use the included client directly. It requires only `requests`:

```bash
python -m pip install requests
python skills/grade-cloud/scripts/grade_predict.py \
  --model GRADE-paper \
  --target-seq MRYRLAWLLHPALPSTFRSVLGARLPPPERLCGFQKKTYSKMNNPAI \
  --peptides candidates.fasta \
  --output cloud_scores.csv
```

On first interactive use, the client asks for an API key using hidden input. It does not automatically save or print the key. Key selection follows `--api-key`, then the `GRADE_API_KEY` environment variable, then the terminal prompt. For noninteractive use, supply a key through one of the first two methods.

The client accepts peptide inputs in FASTA (`.fasta`, `.faa`, `.fa`), CSV (`.csv`), or text (`.txt`) formats (see [Candidate peptides](#candidate-peptides)), supports `--top-k`, and writes CSV, TSV, or JSON according to the output extension or `--output-format`. Omit `--output` to print results to standard output. `--top-k` filters returned results and does not reduce the number of peptides submitted or the API quota used. Omitting `--model` uses GRADE-paper. The client checks the model identity returned by the service.

```bash
python skills/grade-cloud/scripts/grade_predict.py --help
```

## Local inference

### Installation

For local inference on Linux, use Python 3.11. A CUDA-enabled GPU is recommended for larger screens; CPU inference is also supported. The installation pins PyTorch 2.4.0 and Transformers 4.44.0.

From the GRADE source directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

### Rank candidate peptides

Save your candidates as `candidates.fasta`, for example:

```fasta
>peptide_1
ACDEFGHIKLMNPQRSTVWY
>peptide_2
YTVSRQPNMLKIHGFEDCA
```

The commands below assume a configured local model directory at `/path/to/model`. Replace it with the directory used for your run, and replace input paths with your own files.

```bash
grade-inference \
  --model-dir /path/to/model \
  --target-name CCR5_HUMAN \
  --peptides candidates.fasta \
  --output-csv CCR5_HUMAN_scores.csv \
  --batch-size 64 \
  --fp16
```

The target name must exactly match a header in the [bundled GPCR FASTA](grade/targets/targets_gpcr_v2.fa). For CPU inference, omit `--fp16`; reduce the batch size if memory is limited.

`python -m grade.inference` accepts the same arguments as `grade-inference`. `python run_inference.py` also accepts those arguments, but is available only when run from the GRADE source directory.

### Supply your own target

A FASTA file with one sequence can be used directly:

```bash
grade-inference \
  --model-dir /path/to/model \
  --target-fasta target.fasta \
  --peptides candidates.fasta \
  --output-csv target_scores.csv
```

For a FASTA containing several targets, add `--target-name` with the complete header of the target to score. Each invocation scores one target. You can also pass its amino acid sequence directly:

```bash
grade-inference \
  --model-dir /path/to/model \
  --target-seq MRYRLAWLLHPALPSTFRSVLGARLPPPERLCGFQKKTYSKMNNPAI \
  --peptides candidates.fasta \
  --output-csv custom_target_scores.csv
```

The short sequence above illustrates the command syntax. Use the GPCR sequence relevant to your experiment.

### Multiple GPUs

For four GPUs, run:

```bash
torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  -m grade.inference \
  --model-dir /path/to/model \
  --target-name CCR5_HUMAN \
  --peptides candidates.fasta \
  --output-csv CCR5_HUMAN_scores.csv \
  --batch-size 64 \
  --fp16
```

The batch size applies to each process. GRADE reuses the target encoding within each process and distributes candidates without adding duplicate padding samples.

### Select results and resume a run

Add `--top-k 5000` to retain the 5,000 highest-scoring candidates. Select CSV, TSV, or JSON using the output extension, or set `--output-format` explicitly:

```bash
grade-inference \
  --model-dir /path/to/model \
  --target-name CCR5_HUMAN \
  --peptides candidates.fasta \
  --output-csv CCR5_HUMAN_top5000.json \
  --output-format json \
  --top-k 5000
```

GRADE saves progress periodically. To continue an interrupted run, use the same output path, model identity and resolved model directory, target identifier and normalized sequence, candidate identifiers, sequences and order, effective `src` source label, `--max-length`, multi-GPU process count (`WORLD_SIZE`), and scoring settings (`--fp16`, `--bf16`, `--attn-implementation`, including their defaults). The v4 checkpoint fingerprints parsed inputs, the effective source label, and these settings. Changing the target file path without changing its parsed input, or changing sequence whitespace alone, does not invalidate a run. Changing the candidate file path is allowed only if the parsed candidates and effective `src` label stay the same: a new filename stem changes the default label and prevents resuming, but an unchanged explicit `--source-label` keeps the label stable. An empty `--source-label` uses the filename stem. The resolved model directory must still match. `--top-k` and output format do not affect the fingerprint. Older checkpoints, including v3 fingerprints, cannot be resumed. For changed inputs, labels, or settings, or older checkpoints, choose a new output path or use `--no-resume` to start fresh and replace the associated progress files and final results.

### Inputs

#### Target GPCR

| Input | Arguments | Requirement |
|---|---|---|
| Bundled target | `--target-name CCR5_HUMAN` | Exact bundled FASTA header |
| Named target in a file | `--target-name my_target --target-fasta targets.fa` | Exact header in the supplied file |
| Single-target FASTA | `--target-fasta target.fasta` | File must contain exactly one sequence |
| Raw sequence | `--target-seq MRYR...` | Cannot be combined with target name or FASTA arguments |

#### Candidate peptides

| Format | Extensions | Contents |
|---|---|---|
| FASTA | `.fasta`, `.faa`, `.fa` | Complete header becomes the identifier; sequence lines form the peptide |
| CSV | `.csv` | A `sequence`, `peptide`, or `seq` column; optional `name`, `peptide_name`, or `id` column |
| Text | `.txt` | One peptide per line. A single column is a sequence; for whitespace-separated columns, the first is the identifier and the last is the sequence |

For example, a CSV input can contain:

```csv
name,sequence
peptide_1,ACDEFGHIKLMNPQRSTVWY
peptide_2,YTVSRQPNMLKIHGFEDCA
```

Sequence whitespace is removed and letters are converted to uppercase. Identifiers must be unique, and sequences must not be empty after normalization. Provide sequences as amino acid letters, without residue numbers or chemical modification notation.

### Outputs

Local results are sorted by descending score. CSV and TSV contain the following headers; local JSON output is an array of objects with the same keys.

| Field | Meaning |
|---|---|
| `rank` | Position in the ranked list, starting at 1 |
| `target` | Target identifier; `custom_target` for a raw sequence |
| `peptide_name` | Candidate identifier |
| `src` | Source label; defaults to the input filename without its extension |
| `sequence` | Normalized candidate sequence |
| `score` | Positive-class softmax score from 0 to 1; higher values rank first |

Higher scores indicate greater predicted agonist potential for the target GPCR. The accompanying work treats scores above 0.9 as high-confidence candidates for experimental prioritization, not as a mandatory cutoff or a calibrated probability of experimental success. Functional assays are needed to establish agonist activity.

### Local options

Run `grade-inference --help` for the complete command help.

| Option | Default | Purpose |
|---|---|---|
| `--model-dir` | — | Explicit local model directory, as used in the examples |
| `--model` | `GRADE-paper` | Registered selection; resolves to `models/grade-paper` relative to the working directory. Use only when that local path is configured; cannot be combined with `--model-dir` |
| `--peptides` | Required | Candidate input file |
| `--output-csv` | Required | Output path for CSV, TSV, or JSON, despite the option name |
| `--output-format` | Inferred | `csv`, `tsv`, or `json`; unrecognized extensions default to CSV |
| `--top-k` | All candidates | Retain the highest-scoring K candidates |
| `--batch-size` | `8` | Candidates per batch, per process |
| `--num-workers` | `0` | Data-loading workers per process |
| `--max-length` | `1024` | Tokenizer maximum length; longer inputs are truncated |
| `--fp16` / `--bf16` | Off | CUDA mixed precision; choose at most one |
| `--attn-implementation` | `sdpa` | `eager`, `sdpa`, or `flash_attention_2`; availability depends on the runtime |
| `--source-label` | Input filename stem | Label stored in `src` |
| `--checkpoint-interval` | `100` | Save progress every N batches |
| `--no-resume` | Off | Start fresh, replacing the saved progress for this output |
| `--no-target-cache` | Off | Disable target encoding reuse |

An explicit `--model-dir` is tracked as a custom path for resuming runs; its directory name does not automatically establish a registered model identity.

## Citation

If you use GRADE in your research, please cite the accompanying GRADE publication. Full bibliographic details and the DOI will be added when available.

## License

The source code is licensed under the [MIT License](LICENSE).
