# NSE Fundamentals & Corporate Filings Collector

This is a **free-first collection toolkit** tailored to the 16 symbols in the user's uploaded market dataset. It downloads available NSE filing metadata, linked XBRL/attachment documents, corporate announcement metadata and annual-report links, then records every attempt in a manifest.

## Important honesty / limitations

- The toolkit has **not downloaded data by itself** when generated. It must be run on a computer with internet access.
- NSE's browser-facing data endpoints are public but undocumented; automated access may fail, change, throttle or be blocked. The script logs failures rather than claiming complete coverage.
- The corporate-actions endpoint may only return upcoming/current actions for some periods. Historical corporate actions must be confirmed from dated announcements and filings.
- Filing metadata and raw XBRL are not yet a complete normalized fundamental table. `parse_integrated_xbrl.py` attempts to parse compatible modern Integrated Filing documents using `nse-xbrl`; older XBRL taxonomies/PDFs may fail and must be handled separately.
- A filing's period-end date is **not** its publication date. Only a verified broadcast/publication timestamp can be used for point-in-time alignment. Revised filings must remain separate.
- This collection does not establish survivorship-free universe history, adjusted total-return targets, or leakage-free model-ready features on its own.
- Respect NSE terms, request slowly, and use the source documents for academic research only as permitted.

## Symbols

ABB, ADANIGREEN, APOLLOHOSP, HDFCBANK, INFY, MARUTI, MOTILALOFS, POWERINDIA, RELIANCE, SBIN, SIEMENS, SOLARINDS, TCS, TITAN, ULTRACEMCO, YESBANK.

## Run on Windows

Open PowerShell in the extracted folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install nse-xbrl
pytest -q
python scripts/collect_nse_filings.py --out nse_collection --start 2020-01-01 --end 2026-10-31 --sleep 1.5
python scripts/parse_integrated_xbrl.py --root nse_collection
```

For a short pilot before the full collection:

```powershell
python scripts/collect_nse_filings.py --symbols RELIANCE HDFCBANK INFY --start 2024-01-01 --end 2026-10-31 --max-attachments-per-symbol 20 --out pilot_collection
```

The collector requires an active internet connection. If the NSE client fails with access/anti-bot errors, do not attempt to evade controls; use the official NSE web interface to download filings for the affected company/date range and place them in the source archive, retaining their original URL and broadcast metadata.

## Output layout

- `nse_collection/raw/<SYMBOL>/financial_results_*.json`: filing metadata including broadcast and revision fields where NSE returns them.
- `nse_collection/documents/<SYMBOL>/...`: original downloaded XBRL/attachments.
- `nse_collection/reports/collection_manifest.csv`: success/failure log, source URL, local path, hash, and context.
- `nse_collection/reports/source_summary.csv`: per-symbol/source counts.
- `nse_collection/processed/parsed_integrated_fundamentals.csv`: records parsed by `nse-xbrl`.
- `nse_collection/processed/parse_failures.csv`: documents requiring a compatible parser or manual review.

## Required checks after collection

1. Check source summary for failures and empty filing counts.
2. Check every symbol has adequate quarterly and annual filings through the relevant periods.
3. Inspect consolidated vs standalone records; choose one basis consistently.
4. Keep original and revised filings separate and use the version available at each prediction cutoff.
5. Validate units and taxonomies; don't mix Rupees, Lakhs, Crores, or percentages.
6. Manually verify a sample of parsed rows against source documents.
7. Build a point-in-time join only where `broadcast_datetime <= prediction_cutoff` (if date-only, conservatively allow use from the next trading session).
8. Do not mark the study ready until feature and target alignment, corporate actions, survivorship, and target leakage are separately audited.

## Sources

- NSE Financial Results: https://www.nseindia.com/companies-listing/corporate-filings-financial-results
- NSE Integrated Filing Financials: https://www.nseindia.com/companies-listing/corporate-integrated-filing
- NSE Corporate Announcements: https://www.nseindia.com/companies-listing/corporate-filings-announcements
- NSE Corporate Actions: https://www.nseindia.com/static/investor-relations/corporate-actions
- `nse` package API docs: https://bennythadikaran.github.io/NseIndiaApi/api.html
- `nse-xbrl` parser docs: https://github.com/ShantanuAnant/nse-xbrl
