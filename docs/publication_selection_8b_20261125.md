# Proposed publication selection ? Phase 8b

This supersedes the Phase-8a change selection for a later approved commit. No staging has been performed. Existing tracked application files remain part of the checkout.

```text
README.md
backend/evaluation/answer_quality.py
backend/evaluation/answer_pilot.py
backend/evaluation/answer_pilot_live.py
backend/scripts/evaluate_answer_quality.py
backend/scripts/prepare_answer_pilot.py
backend/scripts/run_answer_pilot.py
backend/tests/test_answer_quality.py
backend/tests/test_answer_pilot.py
backend/tests/test_answer_pilot_live.py
data/evaluation/answer_quality_cases.json
data/evaluation/answer_quality_fixtures.json
docs/public_evaluation_summary_20261125.md
docs/publication_selection_8a_20261125.md
.github/workflows/ci.yml
ci/offline_guard/sitecustomize.py
docs/ci_reproducibility_8b_20261125.md
docs/publication_selection_8b_20261125.md
```

Exactly 18 changed/new files: the previous 14-file selection plus the workflow, offline guard, CI audit and this updated selection. The eleven Phase-7 source/test/data files are essential for all 665 tests to be available.

Exclude all other untracked historical reports, real-pilot artifacts, reviews, approval/ledger files, credentials, databases, environments and model caches. The Phase-8a selection remains a historical record; its statement that CI is future work is superseded here. No existing source, tests, requirements or original reports were modified in Phase 8b.
