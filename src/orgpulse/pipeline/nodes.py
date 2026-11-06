from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
import pandas as pd

from orgpulse.core.domain import (
    DetectionMode,
    Domain,
    KpiSpec,
    _default_dq_recommendation,
    _default_offline_recommend,
)
from orgpulse.detect.orchestrator import detect_kpi, overall_severity
from orgpulse.detect.rules import datemath_detect, min_history_for, threshold_detect
from orgpulse.detect.types import Severity
from orgpulse.detect.validation import MIN_ASSESSMENT_DAYS, kpi_domain_reason, kpi_series_reason
from orgpulse.domains.pharma import PHARMA
from orgpulse.forecast.calibration import interval_calibration
from orgpulse.forecast.prophet_forecaster import forecast_kpi, series_from_generator
from orgpulse.lineage import UNSAFE_LINEAGE, lineage_confidence
from orgpulse.pipeline.llm import LLMClient
from orgpulse.pipeline.state import (
    HUMAN_OVERSIGHT_DISCLAIMER,
    KpiFinding,
    PipelineState,
    Recommendation,
    data_quality_finding,
)


def validated_clinical_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Validate RAW clinical rows before aggregating to a daily series — the single builder
    used by BOTH detection and forecasting. Aggregation hides per-row corruption (a negative
    site total sums into a plausible positive), so per-site validity is checked here, first.

    Raises ValueError if: a date is missing/NaT; (date, site_id) is not unique; or any present
    per-site daily_enrollment is negative or non-integer (enrollment is a nonnegative count).
    NaN per-site values are allowed — they propagate to a NaN daily total (→ DATA_QUALITY).
    Returns the daily (ds, y) frame with any-NaN-in-a-day → NaN.
    """
    if "date" not in df.columns or "daily_enrollment" not in df.columns:
        raise ValueError("clinical frame missing required columns")
    if df["date"].isna().any():
        raise ValueError("clinical frame has missing/NaT dates")
    if "site_id" in df.columns and df.duplicated(["date", "site_id"]).any():
        raise ValueError("duplicate (date, site_id) clinical records")
    # Coerce to numeric and AGGREGATE THE COERCED VALUES — not the raw column. A string dtype
    # column would otherwise be "summed" by pandas via concatenation ("1"×8 → "11111111"),
    # a wrong-dtype fail-open. Non-numeric strings coerce to NaN (→ more NaN than the original
    # had → reject); numeric strings coerce correctly.
    coerced = pd.to_numeric(df["daily_enrollment"], errors="coerce")
    if int(coerced.isna().sum()) != int(df["daily_enrollment"].isna().sum()):
        raise ValueError("non-numeric enrollment values")
    present = coerced[coerced.notna()]
    if ((present < 0) | (present != present.round())).any():
        raise ValueError("impossible per-site enrollment (negative or non-integer count)")
    return series_from_generator(df.assign(daily_enrollment=coerced), "date", "daily_enrollment")


class DetectNodeOutput(TypedDict):
    kpi_findings: list[KpiFinding]
    overall_severity: str


class ForecastNodeOutput(TypedDict):
    forecast: dict


def _kpi_finding(spec: KpiSpec, series, bounds: dict) -> KpiFinding:
    """
    Assess one KPI series and ALWAYS return a finding. This is the single fail-safe gate:
    the default is DATA_QUALITY, and SAFE/WARNING/CRITICAL are returned only after the data
    is affirmatively validated (1-D, numeric, >=120 points, in-domain) AND detected. A short,
    missing, ragged, non-numeric, or out-of-domain series can never be omitted or read as SAFE.
    Detection dispatches on the KPI's mode (TIMESERIES / THRESHOLD / DATEMATH — all wired).
    """
    reason = kpi_series_reason(spec.name, series, min_len=min_history_for(spec.detection_mode), bounds=bounds)
    if reason is not None:
        return data_quality_finding(spec.name, reason)
    s = np.asarray(series, dtype=float)
    if spec.detection_mode is DetectionMode.TIMESERIES:
        baseline, recent = s[:-30], s[-30:]
        results = detect_kpi(baseline, recent, kpi=spec.name)  # returns DATA_QUALITY on non-finite data
        sev = overall_severity(results)
        sev_name = sev.name
        detail = next((r for r in results if r.severity == sev), results[0]).detail  # driving detector
    elif spec.detection_mode is DetectionMode.THRESHOLD:
        r = threshold_detect(s, spec.thresholds, spec.name)
        sev_name, detail = r.severity.name, r.detail
    else:  # DATEMATH
        r = datemath_detect(s, spec.thresholds, spec.name)
        sev_name, detail = r.severity.name, r.detail
    v = float(s[-1])
    latest = v if math.isfinite(v) else None
    return KpiFinding(kpi=spec.name, severity=sev_name, detail=detail, latest_value=latest)


def detect_node(state: PipelineState, domain: Domain = PHARMA) -> DetectNodeOutput:
    """
    Detect anomalies for each of the domain's KPIs. Every KPI ALWAYS yields a finding: an
    unusable series (missing, malformed, short, non-finite, or out of domain) becomes
    DATA_QUALITY rather than being skipped — the "bad data never crashes, never reads SAFE"
    contract at the node level. Extraction happens inside the guard so a missing stream/column
    or a ragged array is DATA_QUALITY, not a crash.
    """
    # Streams live under a generic "streams" key when driven by the graph; direct callers may
    # pass the streams dict as the state itself. Support both.
    streams = state.get("streams") or state
    bounds = domain.bounds()
    findings: list[KpiFinding] = []
    for spec in domain.kpi_specs:
        try:
            series = domain.extract(streams, spec.name)
        except Exception:
            # extract is UNTRUSTED domain code: ANY exception (missing stream, wrong shape, overflow,
            # or an arbitrary RuntimeError/OSError from inside it) means "this KPI is unusable" ->
            # DATA_QUALITY, never a crash. Only BaseException (KeyboardInterrupt/SystemExit) escapes.
            findings.append(data_quality_finding(spec.name, "unusable input (extract failed)"))
            continue
        findings.append(_kpi_finding(spec, series, bounds))  # always returns a finding (never None)

    # DATA_QUALITY dominates (mirrors detect.orchestrator.overall_severity): if any KPI
    # could not be assessed — or nothing was analyzable at all — the run is DATA_QUALITY,
    # not a lower severity. Otherwise take the max of SAFE/WARNING/CRITICAL. (A bare
    # numeric max is wrong here because Severity.DATA_QUALITY = -1, the lowest value.)
    if not findings or any(f.severity == "DATA_QUALITY" for f in findings):
        overall = Severity.DATA_QUALITY
    else:
        overall = max(Severity[f.severity] for f in findings)

    return {"kpi_findings": findings, "overall_severity": overall.name}


def forecast_node(state: PipelineState, domain: Domain = PHARMA) -> ForecastNodeOutput:
    """
    Forecast the domain's hero KPI at horizons 30, 60, 90 days, with a calibration factor
    APPLIED to the published interval (not just diagnosed).

    Raw Prophet intervals are overconfident (~70% coverage against a nominal 90%). We
    fit a widening scale via rolling-origin folds (`interval_calibration`) and apply it
    to the production bounds so the published interval reflects that correction. The
    scale is fit IN-SAMPLE, so `calibration.in_sample` is recorded and the out-of-sample
    coverage is explicitly not claimed. If the history is too short to calibrate, the raw
    interval is published unchanged with `calibration.applied=False`.

    Returns forecast[kpi] with per-horizon {yhat, lower, upper} plus a `calibration` block.
    """
    forecast_result = {}
    hero = domain.hero()

    # PARITY (engine invariant): detection runs before forecast. If detection ruled the hero
    # unusable (DATA_QUALITY), publish NO forecast — regardless of whether the domain's
    # `extract` and `hero_frame` happen to agree. This encodes "one validated verdict, both
    # consumers" at the engine level instead of trusting each domain to keep them consistent.
    hero_finding = next((f for f in state.get("kpi_findings", []) if getattr(f, "kpi", None) == hero), None)
    if hero_finding is not None and hero_finding.severity == "DATA_QUALITY":
        return {"forecast": forecast_result}

    if domain.hero_frame is not None:
        # Build the hero's dated series and validate it FULLY before Prophet sees it — both the
        # `y` values (via the shared series check) AND the `ds` dates (present, parseable,
        # non-null, unique). Prophet crashes on NaT/missing ds and silently mis-forecasts on
        # duplicate ds, so a corrupt hero_frame must yield NO forecast, never a crash or a
        # bogus one. (Pharma's builder already guarantees clean dates; this protects any domain.)
        streams = state.get("streams") or state
        try:
            daily_df = domain.hero_frame(streams)
            # hero_frame is domain-supplied: a buggy one may return the WRONG TYPE (None, an ndarray,
            # a dict) or raise while building. Fail closed to "no forecast", never crash the run.
            # (AttributeError is caught below because `.columns`/`.dt` on a non-DataFrame raises it.)
            if not isinstance(daily_df, pd.DataFrame):
                return {"forecast": forecast_result}
            if "ds" not in daily_df.columns or "y" not in daily_df.columns:
                return {"forecast": forecast_result}
            ds = pd.to_datetime(daily_df["ds"], errors="coerce")
            if ds.isna().any() or ds.duplicated().any():
                return {"forecast": forecast_result}
            # Prophet rejects timezone-aware ds ("Column ds has timezone specified"). A tz-aware
            # frame is VALID input, so normalize it to tz-naive (canonicalized to UTC) and write it
            # BACK to the frame that forecast_kpi/interval_calibration consume — else the original
            # tz-aware frame reaches Prophet and crashes. Pharma builds tz-naive dates, so this is a
            # no-op there (guaranteed byte-identical: the frame is only rebuilt when tz is present).
            if getattr(ds.dt, "tz", None) is not None:
                daily_df = daily_df.assign(ds=ds.dt.tz_convert("UTC").dt.tz_localize(None))
            # Validate AND normalize `y` to float, then write it BACK — symmetric with the ds fix.
            # A numeric-string/object y column coerces here for validation but, if the ORIGINAL
            # frame were handed to Prophet, would crash it ("'>=' not supported between str and
            # float"). Assigning the coerced float array back guarantees Prophet/calibration see
            # floats. (Pharma's y is already float64, so the values are identical -> byte-identical.)
            y = daily_df["y"].to_numpy(dtype=float)
            daily_df = daily_df.assign(y=y)
        except Exception:
            # hero_frame is UNTRUSTED domain code, and so is everything that consumes its output
            # here. ANY exception (wrong return type, overflow, non-numeric y, or an arbitrary
            # RuntimeError/OSError from inside it) means "no forecast", never a crash. Only
            # BaseException (KeyboardInterrupt/SystemExit) escapes.
            return {"forecast": forecast_result}
        if kpi_series_reason(hero, y, min_len=MIN_ASSESSMENT_DAYS, bounds=domain.bounds()) is not None:
            return {"forecast": forecast_result}

        # Forecast at horizons 30, 60, 90
        fc = forecast_kpi(daily_df, hero, horizons=(30, 60, 90))

        # Fit the interval-calibration scale on this history and APPLY it to the
        # published bounds. Guard short histories: fall back to raw, uncalibrated.
        try:
            cal = interval_calibration(daily_df)
            scale, raw_cov, applied = cal.scale_factor, cal.raw_coverage, True
            cal_meta = {
                "applied": True, "in_sample": True, "scale_factor": float(scale),
                "raw_coverage": float(raw_cov), "nominal": float(cal.nominal),
                "note": "interval widened by an in-sample calibration factor; "
                        "out-of-sample coverage not validated",
            }
        except ValueError:
            scale = 1.0
            cal_meta = {
                "applied": False, "in_sample": True, "scale_factor": 1.0,
                "raw_coverage": None, "nominal": 0.9,
                "note": "history too short to calibrate; raw Prophet interval published",
            }

        # Lineage confidence: a SEPARATE widening reason from the statistical calibration above,
        # composed on top of it, never conflated into one number. None means the domain makes no
        # genealogy-awareness claim at all, so no lineage key is published. A domain that DOES
        # implement lineage_extract and then fails to deliver usable data (raises, or hands back
        # data lineage_confidence can't parse) falls back to UNSAFE_LINEAGE, never a silent pass.
        combined_scale = scale
        lineage_meta = None
        if domain.lineage_extract is not None:
            try:
                raw_dates, raw_tiers = domain.lineage_extract(streams)
                lineage_result = lineage_confidence(raw_dates, raw_tiers)
            except Exception:
                lineage_result = UNSAFE_LINEAGE
            lineage_meta = {
                "confidence": lineage_result.confidence,
                "tier_counts": lineage_result.tier_counts,
                "reason": lineage_result.reason,
                "low_trust": lineage_result.low_trust,
                "widen_factor": lineage_result.widen_factor,
            }
            combined_scale = scale * lineage_result.widen_factor

        forecast_result[hero] = {"calibration": cal_meta}
        if lineage_meta is not None:
            forecast_result[hero]["lineage"] = lineage_meta
        for h in (30, 60, 90):
            fc_at_h = fc.at_horizon(h)
            yhat = float(fc_at_h["yhat"])
            lower_raw = float(fc_at_h["yhat_lower"])
            upper_raw = float(fc_at_h["yhat_upper"])
            lower = yhat - combined_scale * (yhat - lower_raw)  # statistical + lineage widening
            upper = yhat + combined_scale * (upper_raw - yhat)
            if not (math.isfinite(yhat) and math.isfinite(lower) and math.isfinite(upper)):
                # An extreme hero value (e.g. a finite value at the float64 boundary) can push
                # Prophet's trend/uncertainty arithmetic, or the calibration scaling above, past
                # float64 range into inf/NaN. A non-finite bound is not a valid forecast —
                # publishing one would crash downstream JSON serialization (allow_nan=False) or
                # display a bogus number. Suppress the WHOLE forecast rather than a partial one
                # with some horizons finite and others not.
                return {"forecast": {}}
            forecast_result[hero][f"h{h}"] = {"yhat": yhat, "lower": lower, "upper": upper}

    return {"forecast": forecast_result}


def analyst_node(state: PipelineState, llm: LLMClient, domain: Domain = PHARMA) -> dict:
    """
    Analyze findings and forecast via LLM, using the domain's analyst framing.

    Builds a prompt from kpi_findings + forecast, calls llm.analyze,
    returns {"analysis": <str>}.
    """
    findings = state.get("kpi_findings", [])
    forecast = state.get("forecast", {})

    # Build a readable prompt from findings and forecast
    prompt_lines = ["KPI Findings:"]
    for f in findings:
        prompt_lines.append(f"- {f.kpi} ({f.severity}): {f.detail} [latest={f.latest_value}]")

    hero = domain.hero()
    prompt_lines.append(f"\nForecast (hero KPI {hero}):")
    hero_fc = forecast.get(hero, {})
    horizons = [h for h in ("h30", "h60", "h90") if h in hero_fc]  # skip the `calibration` meta key
    if horizons:
        for horizon in horizons:
            data = hero_fc[horizon]
            yhat = data.get("yhat", "N/A")
            lower = data.get("lower", "N/A")
            upper = data.get("upper", "N/A")
            prompt_lines.append(f"- {horizon}: yhat={yhat}, [{lower}, {upper}]")
    else:
        prompt_lines.append("- No forecast available")

    prompt = "\n".join(prompt_lines)

    # Guard the analyst LLM call (symmetric with recommendation_node): the online Claude path can
    # raise (network/API), and a raise here would crash the whole run with no artifact. Fail closed
    # to a neutral note; the KPI findings + forecast + governed recommendations still carry the run.
    # (Offline the stub never raises, so this is a byte-identical no-op for the default path.)
    try:
        analysis = llm.analyze(domain.analyst_system, prompt)
    except Exception:
        analysis = "Automated interpretation unavailable (the analyst step failed); refer to the KPI findings and forecast."
    if not isinstance(analysis, str):
        analysis = ""

    return {"analysis": analysis}


def recommendation_node(state: PipelineState, llm: LLMClient, domain: Domain = PHARMA) -> dict:
    """
    Generate recommendations for flagged KPIs, using the domain's framing (not pharma-specific).

    Splits flagged findings by severity:
    - anomaly_kpis (WARNING/CRITICAL) get an LLM-authored recommendation (the online system prompt
      and offline stub supply the domain's language; the engine's user prompt is domain-neutral).
    - dq_kpis (DATA_QUALITY) get a deterministic data-quality recommendation from
      ``domain.dq_recommendation`` — a data problem, not a genuine anomaly.
    Hero fallback: if both groups are empty, the domain's hero KPI is used (as a data-quality case
    when the run is DATA_QUALITY, else a routine review).

    Every recommendation carries HUMAN_OVERSIGHT_DISCLAIMER and defaults to
    review_status="PENDING_HUMAN_REVIEW". Alert fires on a critical anomaly OR a
    systemic data failure: overall_severity in ("CRITICAL", "DATA_QUALITY").
    Returns {"recommendations": [...], "alert": <bool>}.
    """
    findings = state.get("kpi_findings", [])
    overall_severity = state.get("overall_severity", "SAFE")

    # Split flagged findings by severity
    anomaly_kpis = [f.kpi for f in findings if f.severity in ("WARNING", "CRITICAL")]
    dq_kpis = [f.kpi for f in findings if f.severity == "DATA_QUALITY"]

    # Hero fallback when nothing is flagged: if the run is DATA_QUALITY (nothing was
    # analyzable), treat the hero as a data-quality case (not a process CAPA); otherwise
    # it is a genuinely calm run and the hero gets a routine CAPA-style review rec.
    if not anomaly_kpis and not dq_kpis:
        if overall_severity == "DATA_QUALITY":
            dq_kpis = [domain.hero()]
        else:
            anomaly_kpis = [domain.hero()]

    recommendations = []

    # Anomaly KPIs -> LLM-authored recommendations (domain framing comes from recommend_system /
    # the offline stub; the engine's user prompt is domain-neutral).
    if anomaly_kpis:
        prompt_lines = ["Generate recommendations for the following flagged KPIs:"]
        for kpi in anomaly_kpis:
            prompt_lines.append(f"- {kpi}")
        prompt = "\n".join(prompt_lines)

        # The recommend call runs a Domain-supplied callable (offline_recommend) / a not-runtime-
        # schema-guaranteed online LLM. A RAISE here must not crash the run (the strict per-item
        # validation below only runs AFTER a successful return, so it can't catch an exception).
        try:
            rec_dicts = llm.recommend(domain.recommend_system, prompt, kpis=anomaly_kpis)
        except Exception:
            rec_dicts = []
        requested = set(anomaly_kpis)
        seen: set[str] = set()
        for rec_dict in rec_dicts if isinstance(rec_dicts, list) else []:
            # Be strict about the LLM's output (the online Claude path is not schema-guaranteed
            # at runtime): the item must be a dict whose kpi is one we REQUESTED and whose text
            # and capa_frame are non-empty strings. Malformed values (None text, numeric
            # capa_frame) and invented KPIs are SKIPPED, not coerced into a governed rec — the
            # anomaly still surfaces via the finding + alert (which fire on overall_severity).
            if not isinstance(rec_dict, dict):
                continue
            kpi, text, frame = rec_dict.get("kpi"), rec_dict.get("text"), rec_dict.get("capa_frame")
            if not (isinstance(kpi, str) and kpi in requested
                    and isinstance(text, str) and text.strip()
                    and isinstance(frame, str) and frame.strip()):
                continue
            recommendations.append(
                Recommendation(
                    kpi=kpi, text=text, capa_frame=frame,
                    disclaimer=HUMAN_OVERSIGHT_DISCLAIMER,
                    review_status="PENDING_HUMAN_REVIEW",
                )
            )
            seen.add(kpi)

        # A flagged anomaly KPI MUST carry a governed recommendation. Any that the LLM/stub failed
        # to produce a VALID one for (raise, malformed, or invented-away) gets a deterministic,
        # domain-neutral safe fallback — so a WARNING/CRITICAL run can never emit an EMPTY governed
        # stage (which verify_run would otherwise accept as a valid, semantically-hollow artifact).
        missing = [k for k in anomaly_kpis if k not in seen]
        for d in _default_offline_recommend(missing):
            recommendations.append(
                Recommendation(
                    kpi=d["kpi"], text=d["text"], capa_frame=d["capa_frame"],
                    disclaimer=HUMAN_OVERSIGHT_DISCLAIMER,
                    review_status="PENDING_HUMAN_REVIEW",
                )
            )

    # Data-quality KPIs -> deterministic data-quality recommendation (no LLM). dq_recommendation is
    # a Domain callable: a raise or a non-string/empty return must not crash or write a blank
    # governed rec — fall back to the deterministic default text (SAFE is earned, never a fallback).
    for kpi in dq_kpis:
        try:
            dq_text = domain.dq_recommendation(kpi)
        except Exception:
            dq_text = None
        if not (isinstance(dq_text, str) and dq_text.strip()):
            dq_text = _default_dq_recommendation(kpi)
        recommendations.append(
            Recommendation(
                kpi=kpi,
                capa_frame="DATA-QUALITY",
                text=dq_text,
                disclaimer=HUMAN_OVERSIGHT_DISCLAIMER,
            )
        )

    # Alert fires on a critical anomaly OR a systemic data failure
    alert = overall_severity in ("CRITICAL", "DATA_QUALITY")

    return {"recommendations": recommendations, "alert": alert}
