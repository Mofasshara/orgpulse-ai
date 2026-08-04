# OrgPulse Data Quality Report

**Run ID:** tick-1-2026-04-01T07:00:00
**Generated:** 2026-04-01T07:00:00

🚨 **Overall Severity:** CRITICAL
**🔔 ALERT TRIGGERED** — Human review required.

## KPI Findings

| KPI | Severity | Detail | Latest Value |
|-----|----------|--------|--------------|
| daily_enrollment | CRITICAL | high CUSUM peak=10.59 (warn=5.12, crit=7.96, n=30); first cross idx 17 | 44.0 |
| pass_rate | SAFE | z=0.73 (warn>=3.0, crit>=4.5) | 1.0 |
| calibration_offset | CRITICAL | high CUSUM peak=66.39 (warn=5.12, crit=7.96, n=30); first cross idx 3 | 0.7871989506703393 |
| lead_time_days | WARNING | high CUSUM peak=7.21 (warn=5.12, crit=7.96, n=30); first cross idx 6 | 5.122132912638447 |

## Analysis

Automated interpretation (offline mode): the pipeline observed — KPI Findings:. These movements are assessed against GxP quality expectations; where a KPI is Warning/Critical the trend and forecast suggest a deviation trigger may apply and should be reviewed by the quality team.

## Forecast (Daily Enrollment)

- **30 days:** 54.656969449077195 (95% CI: [44.4023298848758, 65.91218590409491])
- **60 days:** 59.38855159464471 (95% CI: [47.39579358781093, 71.01129663484167])
- **90 days:** 64.15868995468153 (95% CI: [52.60432410605864, 75.38996523363004])

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

### Recommendation 3: lead_time_days

**CAPA Frame:** CAPA-003
**Status:** PENDING_HUMAN_REVIEW

Open a CAPA review for 'lead_time_days': confirm the signal against source data, assess whether an ICH E6(R3) deviation threshold is crossed, and, if confirmed, document root cause and corrective action. Escalate to the responsible quality owner for sign-off.

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).

---

**Decision support only**

> Decision support only — not an automated decision. A qualified human must review and approve before any action (EU AI Act Art. 14 human oversight).
