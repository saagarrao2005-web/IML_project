#!/usr/bin/env python3
"""
Free-first NSE filing collector.
Downloads filing metadata, linked XBRL/attachments where discoverable, and annual-report metadata.
Does not claim completeness; every outcome is written to a manifest.
Run on a machine with internet access. NSE endpoints are public but undocumented and may throttle/block requests.
"""
from __future__ import annotations
import argparse, hashlib, json, logging, re, sys, time
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

DEFAULT_SYMBOLS = [
    "ABB", "ADANIGREEN", "APOLLOHOSP", "HDFCBANK", "INFY", "MARUTI",
    "MOTILALOFS", "POWERINDIA", "RELIANCE", "SBIN", "SIEMENS",
    "SOLARINDS", "TCS", "TITAN", "ULTRACEMCO", "YESBANK"
]
START = date(2020, 1, 1)
END = date(2026, 10, 31)
ACTION_TERMS = re.compile(r"dividend|bonus|split|rights|merger|demerger|delist|suspend|capital reduction|face value", re.I)

def json_default(x):
    if hasattr(x, "isoformat"):
        return x.isoformat()
    return str(x)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def safe_name(value: str, max_len: int = 140) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return (value or "unnamed")[:max_len]

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

def normalize_text(x):
    return str(x or "").strip()

def iter_urls(obj: Any, parent_key=""):
    """Yield (key, URL) pairs from nested NSE responses without assuming one schema."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            if isinstance(v, str) and v.strip().lower().startswith(("http://", "https://")):
                yield key, v.strip()
            else:
                yield from iter_urls(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_urls(v, parent_key)

def get_any(d: dict, candidates: list[str], default=""):
    low = {str(k).lower(): v for k, v in d.items()}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="nse_collection", help="Output folder")
    ap.add_argument("--start", default=START.isoformat())
    ap.add_argument("--end", default=END.isoformat())
    ap.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    ap.add_argument("--sleep", type=float, default=1.25, help="Delay between source requests")
    ap.add_argument("--max-attachments-per-symbol", type=int, default=0,
                    help="0 means no cap; use a cap for a quick pilot")
    args = ap.parse_args()

    from nse import NSE

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if start > end:
        raise SystemExit("--start must be <= --end")

    root = Path(args.out)
    raw = root / "raw"
    docs = root / "documents"
    reports = root / "reports"
    for p in [raw, docs, reports]:
        p.mkdir(parents=True, exist_ok=True)

    log_path = root / "collection.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()])
    manifest = []
    source_summary = []

    def record(status, symbol, data_type, detail, url="", filepath="", extra=None):
        entry = {
            "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "symbol": symbol, "data_type": data_type, "status": status,
            "detail": str(detail), "source_url": url, "local_path": filepath,
            "sha256": sha256(Path(filepath)) if filepath and Path(filepath).exists() else "",
            "extra": extra or {}
        }
        manifest.append(entry)

    with NSE(download_folder=str(root / ".nse_session")) as nse:
        for symbol in args.symbols:
            logging.info("Collecting %s", symbol)
            sym_dir = raw / symbol
            sym_dir.mkdir(parents=True, exist_ok=True)

            # 1) Quarterly and annual financial-result filing metadata, by broadcast date.
            for period in ("quarterly", "annual", "half-yearly"):
                try:
                    result = nse.financial_results(
                        segment="equities", period=period, symbol=symbol,
                        from_date=start, to_date=end
                    )
                    path = sym_dir / f"financial_results_{period}_{start}_{end}.json"
                    write_json(path, result)
                    record("metadata_saved", symbol, f"financial_results_{period}",
                           f"{len(result) if isinstance(result, list) else 'non-list'} filing records",
                           filepath=str(path))
                    # Download URLs specifically associated with XBRL or details/attachments.
                    items = result if isinstance(result, list) else result.get("data", []) if isinstance(result, dict) else []
                    n_downloaded = 0
                    seen = set()
                    for idx, item in enumerate(items):
                        if not isinstance(item, dict):
                            continue
                        broadcast = get_any(item, ["broadcastDateTime", "broadCastDate", "broadcastDate", "filingDate"])
                        period_end = get_any(item, ["to_date", "toDate", "periodEnd", "periodEnded", "re_to_dt"])
                        consolidated = get_any(item, ["consolidated", "consolidatedNonConsolidated", "isConsolidated"])
                        # Keep the filing row itself, including original/revised status and all links.
                        for key, url in iter_urls(item):
                            k = key.lower()
                            if not any(t in k for t in ("xbrl", "attachment", "attch", "file", "detail", "document")):
                                continue
                            if url in seen:
                                continue
                            seen.add(url)
                            if args.max_attachments_per_symbol and n_downloaded >= args.max_attachments_per_symbol:
                                record("not_downloaded_cap", symbol, f"{period}_attachment", "Pilot attachment cap reached", url,
                                       extra={"period_end": period_end, "broadcast": broadcast})
                                continue
                            try:
                                dest = docs / symbol / period
                                dest.mkdir(parents=True, exist_ok=True)
                                name = safe_name(f"{period_end}_{broadcast}_{idx}_{key}")
                                # nse.download_document handles zip/gz and returns actual saved path.
                                saved = nse.download_document(url, folder=str(dest))
                                saved_path = Path(saved)
                                # Record the downloaded path returned by the library; preserve source file.
                                record("downloaded", symbol, f"{period}_attachment", key, url, str(saved_path),
                                       {"period_end": period_end, "broadcast": broadcast, "consolidated": consolidated})
                                n_downloaded += 1
                                time.sleep(args.sleep)
                            except Exception as e:
                                record("download_failed", symbol, f"{period}_attachment", repr(e), url,
                                       extra={"period_end": period_end, "broadcast": broadcast})
                                logging.warning("%s %s attachment failed: %s", symbol, period, e)
                    source_summary.append({"symbol": symbol, "type": f"financial_results_{period}",
                                           "count": len(items), "status": "ok"})
                    time.sleep(args.sleep)
                except Exception as e:
                    record("source_failed", symbol, f"financial_results_{period}", repr(e))
                    source_summary.append({"symbol": symbol, "type": f"financial_results_{period}",
                                           "count": None, "status": "failed", "error": repr(e)})
                    logging.warning("%s %s metadata failed: %s", symbol, period, e)

            # 2) Corporate announcements across the full window; preserve all metadata and download relevant attachments.
            try:
                announcements = nse.announcements(index="equities", symbol=symbol, from_date=start, to_date=end)
                apath = sym_dir / f"announcements_{start}_{end}.json"
                write_json(apath, announcements)
                record("metadata_saved", symbol, "announcements",
                       f"{len(announcements) if isinstance(announcements, list) else 'non-list'} announcement records",
                       filepath=str(apath))
                items = announcements if isinstance(announcements, list) else announcements.get("data", []) if isinstance(announcements, dict) else []
                n_downloaded = 0
                for idx, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    subject = normalize_text(get_any(item, ["subject", "desc", "details", "announcementTitle"]))
                    if not ACTION_TERMS.search(subject):
                        continue
                    for key, url in iter_urls(item):
                        if not any(t in key.lower() for t in ("attachment", "attch", "file", "document")):
                            continue
                        if args.max_attachments_per_symbol and n_downloaded >= args.max_attachments_per_symbol:
                            record("not_downloaded_cap", symbol, "corporate_action_attachment",
                                   "Pilot attachment cap reached", url, extra={"subject": subject})
                            continue
                        try:
                            dest = docs / symbol / "corporate_actions"
                            dest.mkdir(parents=True, exist_ok=True)
                            saved = nse.download_document(url, folder=str(dest))
                            record("downloaded", symbol, "corporate_action_attachment", subject, url, str(saved),
                                   {"subject": subject, "announcement": item})
                            n_downloaded += 1
                            time.sleep(args.sleep)
                        except Exception as e:
                            record("download_failed", symbol, "corporate_action_attachment", repr(e), url,
                                   extra={"subject": subject})
                source_summary.append({"symbol": symbol, "type": "announcements",
                                       "count": len(items), "status": "ok"})
                time.sleep(args.sleep)
            except Exception as e:
                record("source_failed", symbol, "announcements", repr(e))
                source_summary.append({"symbol": symbol, "type": "announcements",
                                       "count": None, "status": "failed", "error": repr(e)})
                logging.warning("%s announcements failed: %s", symbol, e)

            # 3) Corporate-actions endpoint. Depending on NSE endpoint behavior, this may only return current/upcoming actions;
            # this is intentionally kept as a separate source and must not be treated as complete historical action coverage.
            try:
                actions = nse.actions(segment="equities", symbol=symbol, from_date=start, to_date=end)
                path = sym_dir / f"corporate_actions_endpoint_{start}_{end}.json"
                write_json(path, actions)
                record("metadata_saved", symbol, "corporate_actions_endpoint",
                       f"{len(actions) if isinstance(actions, list) else 'non-list'} records; historical coverage must be checked",
                       filepath=str(path))
                source_summary.append({"symbol": symbol, "type": "corporate_actions_endpoint",
                                       "count": len(actions) if isinstance(actions, list) else None,
                                       "status": "retrieved_but_coverage_unverified"})
                time.sleep(args.sleep)
            except Exception as e:
                record("source_failed", symbol, "corporate_actions_endpoint", repr(e))
                source_summary.append({"symbol": symbol, "type": "corporate_actions_endpoint",
                                       "count": None, "status": "failed", "error": repr(e)})

            # 4) Annual report links/metadata. Download PDFs in scope only when the report metadata indicates a year in range.
            try:
                annual = nse.annual_reports(symbol=symbol, segment="equities")
                path = sym_dir / "annual_reports_metadata.json"
                write_json(path, annual)
                items = annual.get("data", []) if isinstance(annual, dict) else []
                record("metadata_saved", symbol, "annual_reports", f"{len(items)} report records", filepath=str(path))
                for idx, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    label = " ".join(str(v) for v in item.values())
                    # Try year extraction but don't silently discard unknown year formats.
                    years = [int(y) for y in re.findall(r"20\d{2}", label)]
                    if years and max(years) < start.year - 1:
                        continue
                    if years and min(years) > end.year:
                        continue
                    url = get_any(item, ["fileName", "url", "attachment", "file"])
                    if isinstance(url, str) and url.startswith(("http://", "https://")):
                        try:
                            dest = docs / symbol / "annual_reports"
                            dest.mkdir(parents=True, exist_ok=True)
                            saved = nse.download_document(url, folder=str(dest))
                            record("downloaded", symbol, "annual_report", label, url, str(saved))
                            time.sleep(args.sleep)
                        except Exception as e:
                            record("download_failed", symbol, "annual_report", repr(e), url)
            except Exception as e:
                record("source_failed", symbol, "annual_reports", repr(e))
                logging.warning("%s annual reports failed: %s", symbol, e)

    # Manifest is append-style per run by writing a run-specific file; never overwrites raw filings.
    manifest_path = reports / "collection_manifest.jsonl"
    run_path = reports / f"collection_manifest_run_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with run_path.open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
    # Also save a convenience CSV.
    try:
        import pandas as pd
        pd.DataFrame(manifest).to_csv(reports / "collection_manifest.csv", index=False)
        pd.DataFrame(source_summary).to_csv(reports / "source_summary.csv", index=False)
    except Exception as e:
        logging.warning("Could not write CSV reports: %s", e)
    write_json(reports / "collection_summary.json", {
        "start": start.isoformat(), "end": end.isoformat(),
        "symbols": args.symbols, "manifest_rows": len(manifest),
        "source_summary": source_summary,
        "limitations": [
            "NSE programmatic endpoints are unofficial/undocumented and may change or block automated access.",
            "The corporate-actions endpoint may not provide complete historical coverage; use dated announcements and filing attachments to verify.",
            "Financial-result filing metadata is not itself a normalized fundamental dataset; linked XBRL and older filing formats require parsing.",
            "Downloaded filing date is not automatically proof of the exact first-public availability time; use broadcast date/time and revision metadata.",
            "Do not assume the collection is complete because a script exits successfully."
        ]
    })
    logging.info("Collection finished. Review %s and the manifest before using any data.", reports)
    print(f"\nDone. Review {reports / 'collection_summary.json'} and {reports / 'collection_manifest.csv'}")
    print("This script records failures and does not certify completeness or point-in-time validity.")

if __name__ == "__main__":
    main()
