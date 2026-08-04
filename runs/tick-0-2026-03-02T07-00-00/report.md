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

- **30 days:** 56.648920908850386 (90% interval: [37.92571877984399, 73.23288258211667])
- **60 days:** 62.387535411980856 (90% interval: [45.62565511382847, 80.0919964878687])
- **90 days:** 65.46957074466971 (90% interval: [47.45514437209759, 83.39201092316043])

## Recommendations

### Recommendation 1: daily_enrollment

**CAPA Frame:** CAPA-001
**Status:** PENDING_HUMAN_REVIEW

Open a CAPA review for 'daily_enrollment': confirm the signal against source data, assess whether an ICH E6(R3) deviation threshold is crossed, and, if confirmed, document root cause and corrective action. Escalate to the responsible quality owner for sign-off.

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).

---

**Decision support only**

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).
