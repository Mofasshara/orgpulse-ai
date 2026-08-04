# OrgPulse Data Quality Report

**Run ID:** tick-4-2026-07-01T07:00:00
**Generated:** 2026-07-01T07:00:00

🚨 **Overall Severity:** CRITICAL
**🔔 ALERT TRIGGERED** — Human review required.

## KPI Findings

| KPI | Severity | Detail | Latest Value |
|-----|----------|--------|--------------|
| daily_enrollment | SAFE | z=0.64 (warn>=3.0, crit>=4.5) | 51.0 |
| pass_rate | SAFE | z=0.81 (warn>=3.0, crit>=4.5) | 1.0 |
| calibration_offset | CRITICAL | high CUSUM peak=49.76 (warn=5.12, crit=7.96, n=30); first cross idx 3 | 1.2426418067536011 |
| lead_time_days | SAFE | z=-0.73 (warn>=3.0, crit>=4.5) | 4.178104160445831 |

## Analysis

Automated interpretation (offline mode): the pipeline observed — KPI Findings:. These movements are assessed against GxP quality expectations; where a KPI is Warning/Critical the trend and forecast suggest a deviation trigger may apply and should be reviewed by the quality team.

## Forecast (Daily Enrollment)

- **30 days:** 51.74379609972096 (90% interval: [33.66945138282166, 69.38354916275918])
- **60 days:** 50.91520789233117 (90% interval: [33.16572348179294, 69.7383404654168])
- **90 days:** 50.31182281687053 (90% interval: [31.199717362326112, 69.45918436449229])

## Recommendations

### Recommendation 1: calibration_offset

**CAPA Frame:** CAPA-001
**Status:** PENDING_HUMAN_REVIEW

Open a CAPA review for 'calibration_offset': confirm the signal against source data, assess whether an ICH E6(R3) deviation threshold is crossed, and, if confirmed, document root cause and corrective action. Escalate to the responsible quality owner for sign-off.

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).

---

**Decision support only**

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).
