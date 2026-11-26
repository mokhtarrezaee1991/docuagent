"""Dateibasierte Prüfung: Quellenstruktur automatisch, Semantik nur annotiert."""

from collections import Counter
from hashlib import sha256
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Text = Annotated[str, Field(min_length=1, strict=True, pattern=r"\S")]
Positive = Annotated[int, Field(gt=0, strict=True)]
Score = Annotated[int, Field(ge=0, le=2, strict=True)]
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)]


class StrictModel(BaseModel):
    """Keine stillschweigende Typkonvertierung oder unbekannte Eingabefelder."""
    model_config = ConfigDict(extra="forbid", strict=True)

    @model_validator(mode="before")
    @classmethod
    def strict_version(cls, value):
        if isinstance(value, dict) and "version" in value and type(value["version"]) is not int:
            raise ValueError("version muss eine ganze Zahl sein.")
        return value


class Source(StrictModel):
    document_id: Text
    chunk_id: Positive
    page_number: Positive


def source_key(source):
    return (source.document_id, source.chunk_id, source.page_number)


class Evidence(Source):
    quote: Text


class Chunk(Source):
    text: Text


class Document(StrictModel):
    document_id: Text
    filename: Text
    split: Literal["development", "test"]
    chunks: list[Chunk] = Field(min_length=1)


class Facet(StrictModel):
    facet_id: Text
    question: Text
    expected_fact: Text | None
    necessary_conditions: list[Text]
    forbidden_claims: list[Text]
    in_allowed_corpus: bool
    in_context: bool
    expected_behavior: Literal["answer", "abstain", "correct_premise"]
    corpus_evidence: list[Evidence]
    context_evidence: list[Evidence]


class Case(StrictModel):
    case_id: Text
    split: Literal["development", "test"]
    family: Text
    category: Literal["answerable", "unanswerable", "multipart", "partial", "misleading", "excluded", "empty_selection"]
    query: Text
    allowed_document_ids: list[Text]
    context: list[Chunk]
    facets: list[Facet] = Field(min_length=1)


class Dataset(StrictModel):
    version: Literal[1]
    description: Text
    documents: list[Document] = Field(min_length=1)
    cases: list[Case] = Field(min_length=1)

    @model_validator(mode="after")
    def cross_validate(self):
        docs = {doc.document_id: doc for doc in self.documents}
        if len(docs) != len(self.documents) or len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("Dokument- und Fall-IDs müssen eindeutig sein.")
        families = [{case.family for case in self.cases if case.split == split} for split in ("development", "test")]
        if not all(families) or families[0] & families[1]:
            raise ValueError("Fragefamilien müssen zwischen den Splits getrennt sein.")
        chunks = {}
        for doc in self.documents:
            for chunk in doc.chunks:
                if chunk.document_id != doc.document_id or source_key(chunk) in chunks:
                    raise ValueError("Ungültige oder doppelte Dokumentabschnitte.")
                chunks[source_key(chunk)] = chunk
        for case in self.cases:
            if len(set(case.allowed_document_ids)) != len(case.allowed_document_ids) or any(identifier not in docs or docs[identifier].split != case.split for identifier in case.allowed_document_ids):
                raise ValueError("Ungültiger Dokumentbereich oder Split.")
            context = {source_key(chunk): chunk for chunk in case.context}
            if len(context) != len(case.context) or len({facet.facet_id for facet in case.facets}) != len(case.facets):
                raise ValueError("Doppelte Kontext- oder Teilfrage-IDs.")
            for key, chunk in context.items():
                if key not in chunks or chunks[key] != chunk or chunk.document_id not in case.allowed_document_ids:
                    raise ValueError("Kontext liegt außerhalb der erlaubten Originalabschnitte.")
            for facet in case.facets:
                for evidence in facet.corpus_evidence:
                    if source_key(evidence) not in chunks or docs[evidence.document_id].split != case.split or evidence.quote not in chunks[source_key(evidence)].text:
                        raise ValueError("Korpusbeleg ist nicht nachweisbar.")
                for evidence in facet.context_evidence:
                    if evidence not in facet.corpus_evidence or source_key(evidence) not in context or evidence.quote not in context[source_key(evidence)].text:
                        raise ValueError("Kontextbeleg ist nicht nachweisbar.")
                if facet.in_allowed_corpus != any(ref.document_id in case.allowed_document_ids for ref in facet.corpus_evidence) or facet.in_context != bool(facet.context_evidence):
                    raise ValueError("Verfügbarkeit und Belegstellen widersprechen sich.")
                if facet.expected_behavior != "abstain" and not facet.in_context:
                    raise ValueError("Eine erwartete Antwort benötigt einen Kontextbeleg.")
                if facet.in_context and facet.expected_fact is None:
                    raise ValueError("Ein belegter Fakt benötigt eine Sollangabe.")
        return self


class Claim(StrictModel):
    claim_id: Text
    text: Text
    cited_source_indices: list[Annotated[int, Field(ge=0, strict=True)]]


class Snapshot(StrictModel):
    snapshot_id: Text
    case_id: Text
    provenance: Literal["synthetic_fixture"]
    fixture_kind: Literal["intended_good", "intended_faulty"]
    mutation: Text
    answer: Text
    sources: list[Source]
    claims: list[Claim]

    @model_validator(mode="after")
    def validate_claims(self):
        if len({claim.claim_id for claim in self.claims}) != len(self.claims):
            raise ValueError("Doppelte Aussage-IDs.")
        for claim in self.claims:
            if claim.text not in self.answer or len(set(claim.cited_source_indices)) != len(claim.cited_source_indices) or any(index >= len(self.sources) for index in claim.cited_source_indices):
                raise ValueError("Aussage oder Quellenzuordnung passt nicht zum Antwort-Snapshot.")
        return self


class Snapshots(StrictModel):
    version: Literal[1]
    description: Text
    snapshots: list[Snapshot] = Field(min_length=1)


class ModelSnapshot(StrictModel):
    """Echte Antwort; Originaldaten bleiben zusätzlich im gebundenen Laufartefakt."""

    snapshot_id: Text
    case_id: Text
    provenance: Literal["openai_model_response"]
    run_id: Text
    run_artifact_sha256: Hash
    answer: Text
    sources: list[Source]
    claims: list[Claim]

    @model_validator(mode="after")
    def validate_claims(self):
        # Dieselben Aussage-/Quellenregeln wie bei bisherigen Fixtures.
        return Snapshot.validate_claims(self)


class ModelSnapshots(StrictModel):
    """Separates v2-Format; v1-Fixtures und ihre Hashes bleiben unverändert."""

    version: Literal[2]
    description: Text
    dataset_sha256: Hash
    run_id: Text
    run_artifact_sha256: Hash
    snapshots: list[ModelSnapshot] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_binding(self):
        if any(item.run_id != self.run_id or item.run_artifact_sha256 != self.run_artifact_sha256 for item in self.snapshots):
            raise ValueError("Snapshot und Laufbindung stimmen nicht überein.")
        return self


def validate_model_artifact(dataset, snapshots, dataset_hash, artifact_bytes):
    """Prüft die Bindung an Originalantwort und eingefrorenen Kontext.

    Ein Dry-Run ist kein Nachweis einer Modellantwort. Hashes beweisen nur
    Integrität und Zuordnung, niemals echte Herkunft oder semantische Wahrheit.
    """
    if sha256(artifact_bytes).hexdigest() != snapshots.run_artifact_sha256 or snapshots.dataset_sha256 != dataset_hash:
        raise ValueError("Laufartefakt oder Datensatz passt nicht zu den Modellantworten.")
    artifact = json.loads(artifact_bytes)
    if not isinstance(artifact, dict) or artifact.get("mode") != "live" or artifact.get("run_id") != snapshots.run_id or artifact.get("dataset_sha256") != dataset_hash:
        raise ValueError("Modellantworten benötigen ein zugehöriges Live-Laufartefakt.")
    cases = {case.case_id: case for case in dataset.cases}
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) > 6 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Ungültige Laufdatensätze.")
    if any(not isinstance(row.get("case_id"), str) or row["case_id"] not in cases for row in rows):
        raise ValueError("Unbekannte oder ungültige Fall-IDs im Laufartefakt.")
    by_id = {row.get("case_id"): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Doppelte Laufdatensätze.")
    filenames = {doc.document_id: doc.filename for doc in dataset.documents}
    for snapshot in snapshots.snapshots:
        row, case = by_id.get(snapshot.case_id), cases.get(snapshot.case_id)
        if row is None or case is None:
            raise ValueError("Modellantwort hat keinen passenden Fall im Laufartefakt.")
        context = [chunk.model_dump() for chunk in case.context]
        if row.get("query") != case.query or row.get("context") != context or row.get("allowed_document_ids") != case.allowed_document_ids:
            raise ValueError("Der Lauf verwendet einen abweichenden Kontext oder Dokumentbereich.")
        try:
            wire = row["planned_request"]["wire_body"]
            sent = json.loads(wire["input"])
            service_context = [{**chunk, "filename": filenames[chunk["document_id"]]} for chunk in context]
            if wire["model"] != "gpt-4.1-mini" or sent != {"query": case.query, "context": service_context}:
                raise ValueError("Abweichende Modellanfrage.")
            parsed = row["raw_response"]["structured"]
            sources = [Source.model_validate(source).model_dump() for source in parsed["sources"]]
            if parsed["answer"] != snapshot.answer or sources != [source.model_dump() for source in snapshot.sources]:
                raise ValueError("Abweichende Modellantwort.")
        except (KeyError, TypeError, ValidationError, ValueError):
            raise ValueError("Modellantwort oder Anfrage stimmt nicht mit dem Laufartefakt überein.") from None


class Assessment(StrictModel):
    status: Literal["pending", "scored", "not_applicable"] = "pending"
    score: Score | None = None
    reason: str = ""

    @model_validator(mode="after")
    def check_status(self):
        if self.status == "scored" and (self.score is None or not self.reason.strip()):
            raise ValueError("Bewertungen benötigen Note und Begründung.")
        if self.status != "scored" and self.score is not None:
            raise ValueError("Pending und N/A dürfen keine Note enthalten.")
        if self.status == "not_applicable" and not self.reason.strip():
            raise ValueError("N/A benötigt eine Begründung.")
        return self


class ClaimReview(StrictModel):
    claim_id: Text
    correctness: Assessment = Field(default_factory=Assessment)
    faithfulness: Assessment = Field(default_factory=Assessment)
    citation_coverage: Assessment = Field(default_factory=Assessment)


class SourceReview(StrictModel):
    source_index: Annotated[int, Field(ge=0, strict=True)]
    claim_ids: list[Text]
    source_precision: Assessment = Field(default_factory=Assessment)


class FacetReview(StrictModel):
    facet_id: Text
    completeness: Assessment = Field(default_factory=Assessment)
    outcome: Literal["pending", "correct_answer", "correct_partial_answer", "justified_abstention", "false_abstention", "incorrect_answer", "unsupported_answer", "not_applicable"] = "pending"
    reason: str = ""


class SnapshotReview(StrictModel):
    snapshot_id: Text
    snapshot_sha256: Hash
    claim_inventory_complete: bool | None = None
    claims: list[ClaimReview]
    sources: list[SourceReview]
    facets: list[FacetReview]


class Reviews(StrictModel):
    version: Literal[1]
    dataset_sha256: Hash
    fixtures_sha256: Hash
    reviewer: str
    snapshots: list[SnapshotReview]


def snapshot_hash(snapshot):
    return sha256(json.dumps(snapshot.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_inputs(dataset_path, fixtures_path, run_artifact_path=None):
    """Liest nur ausdrücklich angegebene lokale Evaluationsdateien."""
    raw_data, raw_fixtures = dataset_path.read_bytes(), fixtures_path.read_bytes()
    dataset = Dataset.model_validate_json(raw_data)
    raw_collection = json.loads(raw_fixtures)
    version = raw_collection.get("version") if isinstance(raw_collection, dict) else None
    fixtures = ModelSnapshots.model_validate_json(raw_fixtures) if type(version) is int and version == 2 else Snapshots.model_validate_json(raw_fixtures)
    cases = {case.case_id: case for case in dataset.cases}
    if len({item.snapshot_id for item in fixtures.snapshots}) != len(fixtures.snapshots) or any(item.case_id not in cases for item in fixtures.snapshots):
        raise ValueError("Antwort-Snapshots enthalten unbekannte oder doppelte IDs.")
    if isinstance(fixtures, ModelSnapshots):
        if run_artifact_path is None:
            raise ValueError("Echte Modellantworten benötigen --run-artifact zur Bindungsprüfung.")
        validate_model_artifact(dataset, fixtures, sha256(raw_data).hexdigest(), run_artifact_path.read_bytes())
    return dataset, fixtures, sha256(raw_data).hexdigest(), sha256(raw_fixtures).hexdigest()


def review_template(dataset, fixtures, dataset_hash, fixtures_hash):
    """Erstellt ausschließlich pending-Felder, niemals menschliche Bewertungen."""
    cases = {case.case_id: case for case in dataset.cases}
    return Reviews(version=1, dataset_sha256=dataset_hash, fixtures_sha256=fixtures_hash, reviewer="", snapshots=[
        SnapshotReview(snapshot_id=item.snapshot_id, snapshot_sha256=snapshot_hash(item),
            claims=[ClaimReview(claim_id=claim.claim_id) for claim in item.claims],
            sources=[SourceReview(source_index=index, claim_ids=[]) for index in range(len(item.sources))],
            facets=[FacetReview(facet_id=facet.facet_id) for facet in cases[item.case_id].facets])
        for item in fixtures.snapshots])


def evaluate(dataset, fixtures, dataset_hash, fixtures_hash, reviews=None):
    """Semantische Kriterien ausschließlich aus separat zugeordneten Reviews."""
    template = review_template(dataset, fixtures, dataset_hash, fixtures_hash)
    merged = {item.snapshot_id: item for item in template.snapshots}
    snapshots = {item.snapshot_id: item for item in fixtures.snapshots}
    if reviews is not None:
        if reviews.dataset_sha256 != dataset_hash or reviews.fixtures_sha256 != fixtures_hash:
            raise ValueError("Review gehört nicht zu Datensatz und Antwortdatei.")
        if len({item.snapshot_id for item in reviews.snapshots}) != len(reviews.snapshots):
            raise ValueError("Doppelte Review-Zuordnung.")
        for item in reviews.snapshots:
            if item.snapshot_id not in snapshots or item.snapshot_sha256 != snapshot_hash(snapshots[item.snapshot_id]):
                raise ValueError("Review gehört nicht zum Antwort-Snapshot.")
            target = merged[item.snapshot_id]
            for field, key in (("claims", "claim_id"), ("sources", "source_index"), ("facets", "facet_id")):
                known = {getattr(entry, key): entry for entry in getattr(target, field)}
                supplied = getattr(item, field)
                if len({getattr(entry, key) for entry in supplied}) != len(supplied) or any(getattr(entry, key) not in known for entry in supplied):
                    raise ValueError("Review referenziert unbekannte oder doppelte Aussagen, Quellen oder Teilfragen.")
                known.update({getattr(entry, key): entry for entry in supplied})
                setattr(target, field, list(known.values()))
            target.claim_inventory_complete = item.claim_inventory_complete
            claim_ids = {claim.claim_id for claim in snapshots[item.snapshot_id].claims}
            for source in target.sources:
                if len(set(source.claim_ids)) != len(source.claim_ids) or not set(source.claim_ids) <= claim_ids:
                    raise ValueError("Quellenreview enthält ungültige Aussageverweise.")
                if source.source_precision.status == "scored" and source.source_precision.score > 0 and not source.claim_ids:
                    raise ValueError("Positive Quellenbewertung benötigt konkrete Aussageverweise.")
            for facet in target.facets:
                if facet.outcome != "pending" and not facet.reason.strip():
                    raise ValueError("Teilfragenurteile benötigen eine Begründung.")
        annotated = any(item.claim_inventory_complete is not None or any(claim.model_dump()[name]["status"] != "pending" for claim in item.claims for name in ("correctness", "faithfulness", "citation_coverage")) or any(source.source_precision.status != "pending" for source in item.sources) or any(facet.completeness.status != "pending" or facet.outcome != "pending" for facet in item.facets) for item in merged.values())
        if annotated and not reviews.reviewer.strip():
            raise ValueError("Annotationen benötigen eine Reviewer-Angabe.")
    cases = {case.case_id: case for case in dataset.cases}
    rows, assessments, outcomes = [], {name: [] for name in ("correctness", "faithfulness", "completeness", "source_precision", "citation_coverage")}, []
    for snapshot in fixtures.snapshots:
        case, review = cases[snapshot.case_id], merged[snapshot.snapshot_id]
        known = {source_key(chunk) for chunk in case.context}
        valid = [source_key(source) in known and source.document_id in case.allowed_document_ids for source in snapshot.sources]
        for claim in review.claims:
            for name in ("correctness", "faithfulness", "citation_coverage"):
                assessments[name].append(getattr(claim, name))
        assessments["source_precision"].extend(source.source_precision for source in review.sources)
        assessments["completeness"].extend(facet.completeness for facet in review.facets)
        outcomes.extend(facet.outcome for facet in review.facets)
        rows.append({"snapshot_id": snapshot.snapshot_id, "case_id": case.case_id, "provenance": snapshot.provenance,
            "technical": {"valid_sources": sum(valid), "source_count": len(valid), "reference_validity": sum(valid) / len(valid) if valid else None,
                          "sources": [{"source_index": index, "valid": value} for index, value in enumerate(valid)],
                          "has_factual_claims_without_sources": bool(snapshot.claims) and not snapshot.sources},
            "semantic_review": review.model_dump()})
    semantic = {}
    for name, values in assessments.items():
        scores = [value.score for value in values if value.status == "scored"]
        semantic[name] = {"status": "annotated" if scores else "pending" if any(value.status == "pending" for value in values) else "not_applicable", "mean_score_0_to_2": sum(scores) / len(scores) if scores else None,
                          "scored_count": len(scores), "pending_count": sum(value.status == "pending" for value in values),
                          "not_applicable_count": sum(value.status == "not_applicable" for value in values), "unit_count": len(values)}
    total = sum(row["technical"]["source_count"] for row in rows)
    valid = sum(row["technical"]["valid_sources"] for row in rows)
    model_responses = isinstance(fixtures, ModelSnapshots)
    return {"dataset_sha256": dataset_hash, "fixtures_sha256": fixtures_hash, "input_kind": "recorded_model_responses" if model_responses else "synthetic_answer_fixtures_not_model_performance",
        **({"generation_run_id": fixtures.run_id, "run_artifact_sha256": fixtures.run_artifact_sha256,
            "call_count_scope": "llm_calls/openai_api_calls zählen nur diese Offline-Auswertung, nicht den ursprünglichen Generierungslauf."} if model_responses else {}),
        "llm_calls": 0, "openai_api_calls": 0, "snapshot_count": len(rows),
        "coverage": {"dataset_cases": len(cases), "cases_with_snapshots": len({item.case_id for item in fixtures.snapshots}),
                     "cases_by_split": dict(Counter(case.split for case in dataset.cases)), "cases_by_category": dict(Counter(case.category for case in dataset.cases))},
        "technical": {"valid_source_count": valid, "source_count": total, "reference_validity": valid / total if total else None,
                      "snapshots_without_sources": sum(not item.sources for item in fixtures.snapshots)},
        "semantic": semantic, "facet_outcomes": dict(Counter(outcomes)),
        "pending_review_units": sum(value["pending_count"] for value in semantic.values()),
        "claim_inventory_reviewed_snapshots": sum(item.claim_inventory_complete is not None for item in merged.values()),
        "claim_inventory_incomplete_snapshots": sum(item.claim_inventory_complete is False for item in merged.values()),
        "snapshots_pending_review": sum(item.claim_inventory_complete is None or any(getattr(claim, name).status == "pending" for claim in item.claims for name in ("correctness", "faithfulness", "citation_coverage")) or any(source.source_precision.status == "pending" for source in item.sources) or any(facet.completeness.status == "pending" or facet.outcome == "pending" for facet in item.facets) for item in merged.values()),
        "reviewer": reviews.reviewer if reviews else None, "snapshots": rows,
        "limitations": ("Aufgezeichnete Modellantworten werden nur offline ausgewertet; ohne Annotation keine semantische Qualitätsmessung. " if model_responses else "Synthetische Antworten testen nur die Auswertung, nicht GPT-4.1-mini. ") + "Technisch gültige Quellen beweisen keine Belegbarkeit. Keine automatische semantische Note aus Textmustern, Fixture-Art oder Quellenexistenz. Semantische Mittelwerte gelten nur für explizit annotierte Einheiten, nicht ungeprüfte ganze Antworten. Der Reviewer muss die Aussageinventare auf Vollständigkeit prüfen. Outcome-Urteile pro Teilfrage stammen ausschließlich aus Reviews. Annotationen sind zugeordnete Urteile, keine automatisch bewiesene Wahrheit. Keine produktive Integration und kein LLM-Judge."}
