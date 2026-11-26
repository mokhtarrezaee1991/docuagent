# Proposed Publication Selection — Phase 8a

This is an explicit recommendation for a later user-approved Git commit, not a staging action. No file has been added to the index by this task. Paths below are repository-relative; they are not private machine paths.

## Exact recommended change set

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
```

This comprises the updated tracked README, eleven new Phase-7 source/test/data files, and two new public documentation files. Earlier application code, requirements, PDFs and retrieval datasets already tracked in the repository are not new Phase-7 additions. Do not stage them again unless a later approved change modifies them.

The README points only to this public documentation and source/data files intended for publication. It does not require publishing the earlier local audit reports or original live response bundle. The original tracked retrieval reports remain unchanged; this task does not certify or rewrite all historical repository contents.

## Excluded from this recommendation

- Original live responses and original/derived real-answer snapshots, reviews and provenance sidecars. Retain the complete originals locally for v2 validation; do not redact them in place.
- Historical internal readiness reports, README proposals, review packages and pilot instructions containing local paths or internal run identifiers.
- Preliminary or duplicate generated reports and older manual-review summary copies.
- The local SQLite database, pilot claim markers, reservations, journals and approvals under `data/storage/`.
- Credentials, `.env`, virtual environments, model caches and private documents.

The public summary reports aggregates and their limitations. It deliberately does not promise full public reproducibility of the withheld real-pilot artifact chain. Existing local report links in historical internal documents are not new public README links.

## Review scope and remaining gates

The selected source/test/data files and public documents are reviewed specifically for credential-like literals, personal/local-machine paths, provider response identifiers and internal execution identifiers. Intentional test strings such as offline placeholders and mock secrets are synthetic test data, not actual access credentials. Environment-variable names and code that implements authorization are necessary source code, not a leaked approval.

This targeted review is not a credential scan of `.env`, runtime databases, approval files, all previously tracked files or Git history. No credential values are retrieved from the process environment for the publication review.

Before staging in a later step, review the exact diff and this allowlist again. Do not use a blanket add-all command. A license, CI workflow, dependency-lock policy and independently human-reviewed answer benchmark remain separate future tasks; none is introduced here.

The secret-safe public documentation must not be replaced with the full local audit package merely to make every historical link visible. A future release of original live evidence requires a separate publication decision and preservation of its complete integrity chain.

## Validation performed in Phase 8a

- All 14 recommended files exist and match this explicit selection. README links and local documentation anchors resolve; documentation links do not point to withheld internal reports.
- Targeted pattern/AST review found no credential-format literals, private machine paths, personal email addresses, real provider response identifiers or embedded pilot execution identifiers in the selection. A flagged API-key property in a mock response was verified as an intentional synthetic marker, not a credential.
- Documented CLI options were checked against the actual parsers; the required packages are installed and `pip check` reports no broken requirements. No dependencies were installed or changed.
- The Mermaid graph was reviewed against application behavior and checked structurally for declared nodes, references and both branches. A Mermaid renderer is not installed; no visual rendering or GitHub rendering test was performed.
- **197 relevant offline tests passed** in this phase. The README's **665 full-suite tests** is the separately verified result from 25 November 2026, not a new full-suite run in Phase 8a.
- No live model call, runtime authorization, production code edit, original-artifact overwrite, credential retrieval, staging, commit or push was performed.

The targeted publication review is not a proof that every conceivable sensitive string or all historical repository content has been exhaustively inspected. A future commit should retain this explicit selection and review the staged diff before publication.
