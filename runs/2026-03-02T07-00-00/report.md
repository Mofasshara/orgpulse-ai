# OrgPulse Data Quality Report

**Run ID:** tick-0-2026-03-02T07:00:00
**Generated:** 2026-03-02T07:00:00

✅ **Overall Severity:** SAFE

## KPI Findings

| KPI | Severity | Detail | Latest Value |
|-----|----------|--------|--------------|
| daily_enrollment | SAFE | bootstrap baseline (no prior period remembered) | 59.0 |
| pass_rate | SAFE | bootstrap baseline (no prior period remembered) | 1.0 |
| calibration_offset | SAFE | bootstrap baseline (no prior period remembered) | 0.6218649353347324 |
| lead_time_days | SAFE | bootstrap baseline (no prior period remembered) | 5.853665900661905 |

## Analysis

Automated interpretation (offline mode): the pipeline observed — KPI Findings:. These movements are assessed against GxP quality expectations; where a KPI is Warning/Critical the trend and forecast suggest a deviation trigger may apply and should be reviewed by the quality team.

## Forecast (Daily Enrollment)

- **30 days:** 56.648920908850386 (95% CI: [44.50709870338794, 67.40346560219919])
- **60 days:** 62.387535411980856 (95% CI: [51.51761214740675, 73.86871353235873])
- **90 days:** 65.46957074466971 (95% CI: [53.78738301619851, 77.0921062829761])

## Recommendations

### Recommendation 1: daily_enrollment

**CAPA Frame:** CAPA-001
**Status:** PENDING_HUMAN_REVIEW

Open a CAPA review for 'daily_enrollment': confirm the signal against source data, assess whether an ICH E6(R3) deviation threshold is crossed, and, if confirmed, document root cause and corrective action. Escalate to the responsible quality owner for sign-off.

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).

---

**Decision support only**

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).
