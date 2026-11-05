from pathlib import Path

ROOT = Path(__file__).parent


def test_user_guide_is_offline_and_illustrated():
    html = (ROOT / "docs" / "USER_GUIDE.html").read_text(encoding="utf-8")
    for asset in ("phase_vector.png", "groove_closure.png", "mass_moment.png", "steam_turbine_rotor_960.jpg", "turbine_blades.jpg"):
        assert (ROOT / "assets" / "help" / asset).is_file()
        assert f"../assets/help/{asset}" in html
    assert "https://" not in html.replace("href=\"https://", "")  # only external links may be source/credit hyperlinks
    assert "CC BY-SA 2.0" in html and "CC BY-SA 3.0" in html
    assert "EP2549060A2" in html and "EP2253800" in html


def test_guide_covers_operator_workflow_and_safety():
    text = (ROOT / "docs" / "USER_GUIDE.html").read_text(encoding="utf-8").lower()
    for term in ("step-by-step", "csv", "closing blade", "last slot", "45°", "g·mm", "not a release-to-service", "solver", "residual"):
        assert term in text


def test_closure_note_documents_three_interpretations_and_sources():
    text = (ROOT / "docs" / "CLOSURE_RESEARCH.md").read_text(encoding="utf-8")
    for term in ("Measured effective CG radius", "Demonstration-only nominal", "Drawing/measurement pending", "total bare-rotor", "EP2549060A2", "EP2253800 A2"):
        assert term in text
    assert "https://patents.google.com/patent/EP2549060A2/en" in text
    assert "https://data.epo.org/publication-server/rest/v1.2/patents/EP2253800NWA2/document.pdf" in text
