# OrgPulse AI — EU AI Act Self-Classification

**System:** OrgPulse AI — PharmaOps
**Risk tier:** high-risk (Annex III)
**Medical device:** No — decision-support software only.

## Rationale

OrgPulse produces AI-generated recommendations that inform decisions in a GxP-regulated pharmaceutical operations context (clinical trials, QC/lab, supply chain). It self-classifies as a high-risk AI system under EU AI Act Annex III as a conservative, defensible posture. It is decision-support only and is NOT a medical device — it never takes automated action.

## Obligations & how OrgPulse meets them

| Article | Requirement | How OrgPulse meets it |
|---------|-------------|------------------------|
| Article 9 | Risk management | Anomaly detection is calibrated to a documented in-control false-alarm target across window lengths; data-quality failures surface as an explicit non-analyzable state rather than silently passing as 'all clear'. |
| Article 11 | Technical documentation | This self-classification, the versioned implementation plans, the roadmap change log, and the learning guide document the system's design, data sources, decision logic, and every deviation. |
| Article 12 | Record-keeping / logging | Every run and every AI-generated recommendation is written to a hash-chained, tamper-evident, append-only audit log. The run entry records the input lineage (data-generation seed and parameters, a SHA-256 hash of the actual input data frames, and — for a remembered-baseline tick — a hash of the baseline that drove detection), output-integrity hashes of the findings and report, the model+version, and a timestamp. For an OFFLINE run (the deterministic default) a past decision can be exactly regenerated from the recorded inputs and its outputs checked against the logged hashes; for an ONLINE run the recorded lineage and output-integrity hashes provide full traceability, but the LLM's output is stochastic and not bit-reproducible (no model snapshot or decoding parameters are captured). Tampering with a past entry, the findings, or the report is detectable via chain and hash verification. (Threat model: the chain is unkeyed, so it detects corruption and naïve edits but not a full-file rewrite that re-signs the chain — keyed signing is noted as future work.) |
| Article 13 | Transparency to deployers | Recommendations are plain-language and each carries an explicit disclaimer that it is decision-support, not an automated decision. |
| Article 14 | Human oversight | Every recommendation defaults to review_status=PENDING_HUMAN_REVIEW and carries a human-oversight disclaimer; a qualified human must review and approve before any action. The system is architecturally prohibited from taking action itself. |

> This is decision-support software for a regulated environment, **not a medical device**. Recommendations require human sign-off (Article 14 human oversight); the system never takes automated/kinetic action.

_Generated from `orgpulse.governance.classification` — the machine-readable source of truth. Regenerate with `python -c "from orgpulse.governance.classification import as_markdown; print(as_markdown())"`._