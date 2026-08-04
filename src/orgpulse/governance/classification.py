from __future__ import annotations

SELF_CLASSIFICATION: dict = {
    "system": "OrgPulse AI — PharmaOps",
    "risk_tier": "high-risk",
    "annex": "Annex III",
    "is_medical_device": False,
    "rationale": (
        "OrgPulse produces AI-generated recommendations that inform decisions in a "
        "GxP-regulated pharmaceutical operations context (clinical trials, QC/lab, "
        "supply chain). It self-classifies as a high-risk AI system under EU AI Act "
        "Annex III as a conservative, defensible posture. It is decision-support only "
        "and is NOT a medical device — it never takes automated action."
    ),
    "obligations": [
        {
            "article": "Article 9",
            "title": "Risk management",
            "how_orgpulse_meets_it": (
                "Anomaly detection is calibrated to a documented in-control false-alarm "
                "target across window lengths; data-quality failures surface as an explicit "
                "non-analyzable state rather than silently passing as 'all clear'."
            ),
        },
        {
            "article": "Article 11",
            "title": "Technical documentation",
            "how_orgpulse_meets_it": (
                "This self-classification, the versioned implementation plans, the roadmap "
                "change log, and the learning guide document the system's design, data "
                "sources, decision logic, and every deviation."
            ),
        },
        {
            "article": "Article 12",
            "title": "Record-keeping / logging",
            "how_orgpulse_meets_it": (
                "Every run and every AI-generated recommendation is written to a "
                "hash-chained, tamper-evident, append-only audit log with a trace_id, "
                "timestamp, input-data hash, and model+version, so any past decision is "
                "fully reconstructable and tampering is detectable via chain verification."
            ),
        },
        {
            "article": "Article 13",
            "title": "Transparency to deployers",
            "how_orgpulse_meets_it": (
                "Recommendations are plain-language and each carries an explicit disclaimer "
                "that it is decision-support, not an automated decision."
            ),
        },
        {
            "article": "Article 14",
            "title": "Human oversight",
            "how_orgpulse_meets_it": (
                "Every recommendation defaults to review_status=PENDING_HUMAN_REVIEW and "
                "carries a human-oversight disclaimer; a qualified human must review and "
                "approve before any action. The system is architecturally prohibited from "
                "taking action itself."
            ),
        },
    ],
}


def as_markdown() -> str:
    c = SELF_CLASSIFICATION
    lines = [
        "# OrgPulse AI — EU AI Act Self-Classification",
        "",
        f"**System:** {c['system']}",
        f"**Risk tier:** {c['risk_tier']} ({c['annex']})",
        f"**Medical device:** {'No' if not c['is_medical_device'] else 'Yes'} — decision-support software only.",
        "",
        "## Rationale",
        "",
        c["rationale"],
        "",
        "## Obligations & how OrgPulse meets them",
        "",
        "| Article | Requirement | How OrgPulse meets it |",
        "|---------|-------------|------------------------|",
    ]
    for o in c["obligations"]:
        lines.append(f"| {o['article']} | {o['title']} | {o['how_orgpulse_meets_it']} |")
    lines += [
        "",
        "> This is decision-support software for a regulated environment, **not a medical "
        "device**. Recommendations require human sign-off (Article 14 human oversight); the "
        "system never takes automated/kinetic action.",
        "",
        "_Generated from `orgpulse.governance.classification` — the machine-readable source "
        "of truth. Regenerate with `python -c \"from orgpulse.governance.classification import "
        "as_markdown; print(as_markdown())\"`._",
    ]
    return "\n".join(lines)
