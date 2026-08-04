# OrgPulse Data Quality Report

**Run ID:** tick-2-2026-05-01T07:00:00
**Generated:** 2026-05-01T07:00:00

🚨 **Overall Severity:** CRITICAL
**🔔 ALERT TRIGGERED** — Human review required.

## KPI Findings

| KPI | Severity | Detail | Latest Value |
|-----|----------|--------|--------------|
| daily_enrollment | CRITICAL | high CUSUM peak=9.08 (warn=5.12, crit=7.96, n=30); first cross idx 22 | 54.0 |
| pass_rate | SAFE | z=-0.47 (warn>=3.0, crit>=4.5) | 0.9375 |
| calibration_offset | CRITICAL | high CUSUM peak=61.77 (warn=5.12, crit=7.96, n=30); first cross idx 3 | 0.9798371325300718 |
| lead_time_days | CRITICAL | high CUSUM peak=10.37 (warn=5.12, crit=7.96, n=30); first cross idx 7 | 5.572996888813004 |

## Analysis

Automated interpretation (offline mode): the pipeline observed — KPI Findings:. These movements are assessed against GxP quality expectations; where a KPI is Warning/Critical the trend and forecast suggest a deviation trigger may apply and should be reviewed by the quality team.

## Forecast (Daily Enrollment)

- **30 days:** 54.60976001274288 (90% interval: [38.7859553788471, 68.59048273432485])
- **60 days:** 57.18918325322031 (90% interval: [41.02523323024609, 72.47968870473271])
- **90 days:** 61.06941633735468 (90% interval: [43.33112467003036, 77.51669881706613])

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
