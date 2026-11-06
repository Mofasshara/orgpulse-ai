# OrgPulse Data Quality Report

**Run ID:** tick-6-2026-09-01T07:00:00
**Generated:** 2026-09-01T07:00:00

🚨 **Overall Severity:** CRITICAL
**🔔 ALERT TRIGGERED** — Human review required.

## KPI Findings

| KPI | Severity | Detail | Latest Value |
|-----|----------|--------|--------------|
| daily_enrollment | CRITICAL | low CUSUM peak=23.20 (warn=5.12, crit=7.96, n=30); first cross idx 7 | 59.0 |
| pass_rate | WARNING | low CUSUM peak=5.65 (warn=5.12, crit=7.96, n=30); first cross idx 6 | 0.8333333333333334 |
| calibration_offset | CRITICAL | high CUSUM peak=51.81 (warn=5.12, crit=7.96, n=30); first cross idx 3 | 1.5666132067140943 |
| lead_time_days | SAFE | z=0.26 (warn>=3.0, crit>=4.5) | 5.905073566969942 |

## Analysis

Automated interpretation (offline mode): the pipeline observed — KPI Findings:. These movements are assessed against GxP quality expectations; where a KPI is Warning/Critical the trend and forecast suggest a deviation trigger may apply and should be reviewed by the quality team.

## Forecast (Daily Enrollment)

- **30 days:** 45.80763482376548 (90% interval: [8.800817252638552, 81.47621375868509])
- **60 days:** 45.26331997882907 (90% interval: [8.302841617194183, 85.6275373558986])
- **90 days:** 44.54807977354475 (90% interval: [4.672662888568254, 82.85941217856767])

## Recommendations

### Recommendation 1: daily_enrollment

**CAPA Frame:** CAPA-001
**Status:** PENDING_HUMAN_REVIEW

Open a CAPA review for 'daily_enrollment': confirm the signal against source data, assess whether an ICH E6(R3) deviation threshold is crossed, and, if confirmed, document root cause and corrective action. Escalate to the responsible quality owner for sign-off.

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).

### Recommendation 2: pass_rate

**CAPA Frame:** CAPA-002
**Status:** PENDING_HUMAN_REVIEW

Open a CAPA review for 'pass_rate': confirm the signal against source data, assess whether an ICH E6(R3) deviation threshold is crossed, and, if confirmed, document root cause and corrective action. Escalate to the responsible quality owner for sign-off.

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).

### Recommendation 3: calibration_offset

**CAPA Frame:** CAPA-003
**Status:** PENDING_HUMAN_REVIEW

Open a CAPA review for 'calibration_offset': confirm the signal against source data, assess whether an ICH E6(R3) deviation threshold is crossed, and, if confirmed, document root cause and corrective action. Escalate to the responsible quality owner for sign-off.

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).

### Recommendation 4: daily_enrollment

**CAPA Frame:** LINEAGE-RISK
**Status:** PENDING_HUMAN_REVIEW

The clinical records behind 'daily_enrollment' include site(s) whose data did not come from a verified system of record (scanned/manual case report forms rather than EDC entry). Treat this forecast as lower-trust: do not act on it without confirming the underlying site data with the responsible quality owner, and note the widened interval already published reflects this genealogy risk, not just statistical uncertainty.

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).

---

**Decision support only**

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).
