# OrgPulse Data Quality Report

**Run ID:** tick-3-2026-06-01T07:00:00
**Generated:** 2026-06-01T07:00:00

🚨 **Overall Severity:** CRITICAL
**🔔 ALERT TRIGGERED** — Human review required.

## KPI Findings

| KPI | Severity | Detail | Latest Value |
|-----|----------|--------|--------------|
| daily_enrollment | WARNING | low CUSUM peak=5.76 (warn=5.12, crit=7.96, n=30); first cross idx 26 | 41.0 |
| pass_rate | CRITICAL | low CUSUM peak=8.98 (warn=5.12, crit=7.96, n=30); first cross idx 22 | 0.9 |
| calibration_offset | CRITICAL | high CUSUM peak=52.65 (warn=5.12, crit=7.96, n=30); first cross idx 3 | 1.1047388661180226 |
| lead_time_days | SAFE | z=-0.23 (warn>=3.0, crit>=4.5) | 5.126739399254856 |

## Analysis

Automated interpretation (offline mode): the pipeline observed — KPI Findings:. These movements are assessed against GxP quality expectations; where a KPI is Warning/Critical the trend and forecast suggest a deviation trigger may apply and should be reviewed by the quality team.

## Forecast (Daily Enrollment)

- **30 days:** 49.312590142961945 (90% interval: [34.72129414738826, 66.15225513281416])
- **60 days:** 50.553118702229895 (90% interval: [34.527430844806766, 66.55289477726112])
- **90 days:** 51.38050999585493 (90% interval: [34.51026880056053, 68.13866320640228])

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

---

**Decision support only**

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).
