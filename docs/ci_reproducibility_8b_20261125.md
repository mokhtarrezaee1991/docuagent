# CI and reproducibility audit — 25 November 2026

## Scope and execution

The workflow runs all 665 currently collected tests, with no exclusions, on Ubuntu 24.04 / Python 3.11. Run from `backend`: `python -m pip check`, then `python -m pytest -v`. There is no repository pytest configuration file; the module invocation supplies the backend import root. Third-party pytest plugin autoload is disabled: the suite does not require async plugins, and incidental locally installed plugins must not influence CI.

Push and pull-request triggers are restricted to `main`. Repository permission is `contents: read`, checkout credentials are not persisted, superseded jobs are cancelled, and the job timeout is 30 minutes. No secrets, deployment or live evaluation are configured.

The existing public [checkout v6](https://github.com/actions/checkout) and [setup-python v6](https://github.com/actions/setup-python/releases) actions support the selected hosted runner. Major tags are intentionally not immutable pins: resolve reviewed full commit SHAs in a future hardening change. See [GitHub's Python CI guide](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).

## Offline test boundary

Installation may download public Python packages, Python itself and action code. CPU PyTorch is installed first from the public PyTorch wheel index to avoid unnecessary CUDA dependencies on Linux. Requirements are then installed unchanged. No model download step exists.

During tests the API-key variable is empty, dotenv loading is disabled, Hugging Face offline flags are set, model-cache paths point at runner-temporary locations, and Python socket connections to external hosts are denied by `ci/offline_guard/sitecustomize.py`. The guard is explicitly checked before pytest and inherited by Python subprocesses. Loopback and local IPC remain available. This is defense in depth for Python tests, not an OS sandbox against malicious/native code. Dependency installation is deliberately outside the network guard.

OpenAI tests use local mocks/MockTransport, including live-runner error and crash simulations. Embedding tests use fake models/vectors. The tokenizer-warning regression creates a tiny vocabulary in memory using installed tokenizers/transformers, without pretrained weights. SQLite tests use temporary databases. No paid API calls, real pilot authorization, production database or private response reports are needed.

Two tracked synthetic PDFs in `data/sample_documents/` are required by dataset-reference tests. Evaluation JSON datasets are required, including the two new Phase-7 answer-quality files. The eleven Phase-7 source/test/data additions in the publication selection must be committed together; the current remote base alone does not contain them. No test depends on withheld live-response bundles or manual reviews.

Existing `.gitignore` rules cover `.env`, virtual environments, Python/pytest caches and `/data/storage/`. Many historical generated reports are still untracked and are not all ignored: use the explicit publication selection, never blanket staging. This phase does not change ignore rules.

## Reproducibility assessment and proposal

The local interpreter is Windows Python 3.11.8. CI selects the available 3.11 patch, not that exact patch. Binary wheels differ across Windows and Linux, particularly PyTorch, NumPy, tokenizers and PyMuPDF. No fresh Linux installation or hosted Actions run is claimed by local Windows tests. A Docker daemon was unavailable for local Linux verification.

Twelve direct requirements are unbounded; LangGraph alone is constrained to `>=1.2.22,<1.3`. There is no transitive lock. Locally observed examples are pytest 9.2.0, sentence-transformers 6.3.0, transformers 5.23.0, tokenizers 0.23.3, torch 2.15.0, openai 3.51.0 and langgraph 1.2.23. These describe one environment, not a cross-platform compatibility guarantee or recommended upgrade set. Transitive dependencies include pydantic/starlette, Hugging Face libraries, NumPy/SciPy/scikit-learn and LangChain components. `pip check` validates declared installed constraints, not behavioral compatibility.

Proposed next step, requiring separate approval:

1. Retain a reviewed direct-dependency input, optionally separating runtime from test tooling.
2. Resolve exact direct and transitive versions in fresh Python-3.11 Linux CPU and Windows environments. Generate hash-checked platform-specific locks with a selected locking tool; do not blindly publish Windows `pip freeze` as a Linux lock.
3. Record the CPU-wheel source, Python patch, resolver version and lock-generation command. Install locked versions with hashes and test both platforms before adoption.
4. Update dependencies intentionally through reviewed lock changes. Cache by lock hash once locks exist; the current pip cache is keyed by requirements and is not a lock or installed-environment cache.
5. Pin the embedding repository revision and record tokenizer/config fingerprints for real-model experiments. Current `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` loading is name-based and not immutable; offline unit tests do not validate its current remote contents.

No requirements, package versions or production code are changed by this phase. A fresh installation can resolve different dependencies tomorrow; the first hosted run and subsequent locks remain necessary gates.

## Local validation results

On 25 November 2026, using the existing Windows Python 3.11.8 virtual environment:

- Complete suite with empty API-key variable, disabled dotenv/plugin autoload, Hugging Face offline flags and the Python network guard: **665 passed in 37.29 seconds**, no exclusions.
- A separate temporary copy containing only tracked files plus the explicit publication selection, without private reports, runtime storage or a virtual environment, with empty model-cache paths: **665 passed in 43.56 seconds**. This reused installed Windows packages through the original interpreter; it was not a fresh dependency installation or Linux run.
- `python -m pip check`: **No broken requirements found**.
- YAML parsing and assertions for triggers, runner, permissions, timeout and complete-suite command passed. Local README/document links resolve. No actionlint executable was available; GitHub's own workflow validation is pending.
- Guard smoke checks rejected external DNS resolution through getaddrinfo, TCP connection and UDP send before communication, while allowing localhost resolution and preserving activation in a Python child process.
- `git diff --check` passed; whitespace was also checked for the new untracked files. All 18 publication-selection paths exist. A targeted credential-format/local-path scan passed; this is not an exhaustive security certification.

Only README is modified among previously tracked files. Its total Git diff includes the already completed Phase-8a rewrite; it should not be interpreted as entirely new CI work. The four newly introduced Phase-8b files are the workflow, offline guard, this audit and the updated publication selection. Historical untracked local artifacts remain outside the selection and were not modified.
