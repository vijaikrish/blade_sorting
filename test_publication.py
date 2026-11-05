"""Publishing checks for personal branding and asset/license clarity."""
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent


def test_personal_project_branding_and_readme_tutorial():
    gui = (ROOT / "gui.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'setWindowTitle("Blade sorting")' in gui
    assert 'QLabel("BLADE SORTING")' in gui
    assert gui.count('setWindowTitle("Blade sorting")') == 1
    assert readme.startswith("# Blade sorting\n")
    assert "## Tutorial" in readme
    assert "docs/images/" in readme
    assert "assets/help/" in readme
    assert "Start_Blade_Sorting.bat" in readme
    assert (ROOT / "Start_Blade_Sorting.bat").is_file()


def test_code_license_and_photo_attribution_are_separate():
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assets = (ROOT / "ASSET_LICENSES.md").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Vijaikrishna Tatchanamoorthy" in license_text
    assert "CC BY-SA 2.0" in assets
    assert "CC BY-SA 3.0" in assets
    assert "not covered by the code's MIT license" in assets


def test_tutorial_local_links_resolve():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide_path = ROOT / "docs" / "USER_GUIDE.html"
    guide = guide_path.read_text(encoding="utf-8")
    markdown_links = re.findall(r"!?\[[^]]*\]\(([^)]+)\)", readme)
    html_links = re.findall(r'(?:href|src)="([^"]+)"', guide)
    assert len(markdown_links) >= 10
    assert len(html_links) >= 10
    for base, links in ((ROOT, markdown_links), (guide_path.parent, html_links)):
        for link in links:
            parsed = urlsplit(link)
            if parsed.scheme or not parsed.path:
                continue
            resolved = base / unquote(parsed.path)
            assert resolved.exists(), f"Broken local tutorial link: {link}"
