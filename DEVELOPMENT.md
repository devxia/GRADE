# Maintaining grade-client

The distribution is `grade-client` version 1.2.0; its import name is `grade` and primary command is `grade`. The basic installation provides Cloud inference with `requests`. `.[local]` adds the pinned local inference dependencies; use Python 3.11 for that environment.

## Cloud implementation and independent Skill

Maintain Cloud registration, credentials, HTTP behavior and the standalone CLI in `grade/cloud.py`. It must remain independently executable using standard-library imports plus the lazily imported `requests` dependency. Do not introduce relative imports or dependencies on other `grade` modules into this file.

The package's six-column orchestration lives in `grade/runner.py`; the Skill script retains its four-column interface. Both use the same credential configuration and Cloud contract. Generate the standalone script after changing the source:

```bash
python scripts/sync_cloud_skill.py
python scripts/sync_cloud_skill.py --check
```

`skills/grade-cloud/scripts/grade_predict.py` is a checked-in generated artifact. Building a wheel or source distribution fails if it does not match the source; builds do not silently regenerate it. The distributed Skill contains the complete script and does not import `grade`.

## Tests and packaging

Run from this directory:

```bash
python -m grade --help
python -m grade.inference --help
python skills/grade-cloud/scripts/grade_predict.py --help
python scripts/sync_cloud_skill.py --check
python -m unittest discover -s tests -v
```

Full inference-cache tests require local inference dependencies. Import failures due to missing PyTorch are not passing tests. Cloud client, unified CLI, registry, writers and resume-control tests run without PyTorch and use synthetic credentials and temporary configurations. Do not use the operator's actual client configuration in tests.

For a clean build environment:

```bash
python -m pip install build
python -m build
```

This produces `grade_client-1.2.0.tar.gz` and `grade_client-1.2.0-py3-none-any.whl`. Check that the wheel contains bundled targets, both console entrypoints, and only `requests` as a mandatory runtime dependency. Torch 2.4.0 and Transformers 4.44.0 belong to the `local` extra. Install the wheel and source archive in separate fresh environments, outside the source tree, and verify default Cloud routing and explicit local errors without importing heavy dependencies. Separately test a copied Skill directory in an environment without the `grade` module.

Publication to PyPI or changes to GitHub visibility require explicit authorization. Keep README installation instructions limited to available installation methods, without release-status notes or future publication instructions. Usage examples should use macOS/Linux commands and paths without a separate platform-support declaration. Keep Citation as bibliographic placeholders until a confirmed citation is supplied. Do not publish under the unrelated PyPI name `grade`.

## Repository boundaries

Changes here do not update the independent GRADE-Cloud deployment or its fixed historical core wheel. Do not replace its lock, source snapshot, requirements or model assets as part of a client release. Cloud deployment and real authenticated inference checks are separate authorized operations.

Local model loading, target caching, DDP, scoring and resume fingerprints retain their training/inference contracts. A supplied `--model-dir` never falls back to Cloud. CLI calls without that argument default to Cloud, including the compatibility entrypoints. Programmatic `grade.inference.run_inference(args)` remains the local pipeline.
