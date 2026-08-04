# OrgPulse Data Quality Report

**Run ID:** tick-5-2026-08-03T07:00:00
**Generated:** 2026-08-03T07:00:00

🚨 **Overall Severity:** CRITICAL
**🔔 ALERT TRIGGERED** — Human review required.

## KPI Findings

| KPI | Severity | Detail | Latest Value |
|-----|----------|--------|--------------|
| daily_enrollment | CRITICAL | low CUSUM peak=21.76 (warn=5.12, crit=7.96, n=30); first cross idx 5 | 50.0 |
| pass_rate | SAFE | z=-0.26 (warn>=3.0, crit>=4.5) | 0.9285714285714286 |
| calibration_offset | CRITICAL | high CUSUM peak=47.25 (warn=5.12, crit=7.96, n=30); first cross idx 4 | 1.4342095627135805 |
| lead_time_days | SAFE | z=-0.50 (warn>=3.0, crit>=4.5) | 4.559527803117606 |

## Analysis

Automated interpretation (offline mode): the pipeline observed — KPI Findings:. These movements are assessed against GxP quality expectations; where a KPI is Warning/Critical the trend and forecast suggest a deviation trigger may apply and should be reviewed by the quality team.

## Forecast (Daily Enrollment)

- **30 days:** 44.0457505846532 (90% interval: [23.944522784495966, 63.59396239155913])
- **60 days:** 43.55085932204261 (90% interval: [22.82832617755352, 63.19364167636914])
- **90 days:** 44.05486098536907 (90% interval: [24.869168074221957, 64.23068285374154])

## Recommendations

### Recommendation 1: daily_enrollment

**CAPA Frame:** CAPA-001
**Status:** PENDING_HUMAN_REVIEW

Open a CAPA review for 'daily_enrollment': confirm the signal against source data, assess whether an ICH E6(R3) deviation threshold is crossed, and, if confirmed, document root cause and corrective action. Escalate to the responsible quality owner for sign-off.

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).

### Recommendation 2: calibration_offset

**CAPA Frame:** CAPA-002
**Status:** PENDING_HUMAN_REVIEW

Open a CAPA review for 'calibration_offset': confirm the signal against source data, assess whether an ICH E6(R3) deviation threshold is crossed, and, if confirmed, document root cause and corrective action. Escalate to the responsible quality owner for sign-off.

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).

---

**Decision support only**

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).
