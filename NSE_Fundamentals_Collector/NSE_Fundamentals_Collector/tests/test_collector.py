import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "collect_nse_filings.py"
spec = importlib.util.spec_from_file_location("collector", SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)

def test_default_symbols_include_uploaded_universe():
    expected = {"ABB","ADANIGREEN","APOLLOHOSP","HDFCBANK","INFY","MARUTI",
                "MOTILALOFS","POWERINDIA","RELIANCE","SBIN","SIEMENS",
                "SOLARINDS","TCS","TITAN","ULTRACEMCO","YESBANK"}
    assert set(collector.DEFAULT_SYMBOLS) == expected

def test_iter_urls_finds_nested_links():
    obj = {"xbrl": "https://example.com/a.xml", "child": [{"attachment": "https://example.com/b.pdf"}]}
    found = list(collector.iter_urls(obj))
    assert ("xbrl", "https://example.com/a.xml") in found
    assert ("attachment", "https://example.com/b.pdf") in found

def test_iter_urls_ignores_non_urls():
    assert list(collector.iter_urls({"xbrl": "not a URL", "value": 4})) == []

def test_safe_name():
    name = collector.safe_name("2025 Q1 / Original Filing")
    assert "/" not in name and name

def test_date_window_is_valid():
    assert collector.date.fromisoformat("2020-01-01") <= collector.date.fromisoformat("2026-10-31")
