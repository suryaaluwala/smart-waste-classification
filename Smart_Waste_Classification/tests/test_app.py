"""Headless smoke test of the Streamlit app. Run from the project root: python tests/test_app.py"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent

app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
assert not app.exception, app.exception
assert [tab.label for tab in app.tabs] == ["Try it", "How it works", "Results", "Data & limits"]

for sample in sorted(path.stem.title() for path in (ROOT / "assets" / "samples").glob("*.jpg")):
    next(button for button in app.button if button.label == sample).click()
    app.run()
    assert not app.exception, app.exception
    headings = [item.value for item in app.markdown if item.value.startswith("###")]
    print(sample, headings)
print("all passed")
