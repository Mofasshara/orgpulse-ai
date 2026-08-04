from orgpulse.governance.classification import SELF_CLASSIFICATION, as_markdown


def test_self_classification_high_risk():
    assert SELF_CLASSIFICATION["risk_tier"] == "high-risk"
    assert SELF_CLASSIFICATION["annex"] == "Annex III"
    assert SELF_CLASSIFICATION["is_medical_device"] is False


def test_obligations_cover_articles_11_12_14():
    obs = SELF_CLASSIFICATION["obligations"]
    arts = {o["article"] for o in obs}
    assert {"Article 11", "Article 12", "Article 14"}.issubset(arts)
    assert all(o["how_orgpulse_meets_it"] for o in obs)


def test_as_markdown_contains_key_terms():
    md = as_markdown()
    assert "Annex III" in md and "high-risk" in md.lower()
    assert "human oversight" in md.lower() and "Article 12" in md
