#!/usr/bin/env python3
"""Parse compatible NSE Integrated Filing XBRL documents while preserving provenance.

This parser deliberately records failures. Older XBRL taxonomies and PDFs may need
separate parsers/manual extraction and are not silently dropped.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from dataclasses import asdict, is_dataclass
import pandas as pd

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", default="nse_collection")
    args=ap.parse_args()
    root=Path(args.root)
    manifest_files=list((root/"reports").glob("collection_manifest_run_*.jsonl"))
    if not manifest_files:
        raise SystemExit("No run manifest found. Run collection first.")
    manifest=[]
    for mf in sorted(manifest_files):
        for line in mf.read_text(encoding="utf-8").splitlines():
            try: manifest.append(json.loads(line))
            except json.JSONDecodeError: pass
    by_path={}
    for row in manifest:
        p=row.get("local_path")
        if p:
            by_path[str(Path(p).resolve())]=row
    xmls=list((root/"documents").rglob("*.xml"))
    xmls += [p for p in (root/"documents").rglob("*") if p.is_file() and p.suffix.lower() not in (".pdf",".zip",".gz",".csv",".json") and p.read_bytes()[:1000].lstrip().startswith(b"<?xml")]
    rows=[]; failures=[]
    try:
        from nse_xbrl import FilingResult
    except ImportError:
        raise SystemExit("Install parser dependency: pip install nse-xbrl")
    for p in sorted(set(xmls)):
        meta=by_path.get(str(p.resolve()),{})
        try:
            text=p.read_text(encoding="utf-8",errors="replace")
            parsed=FilingResult.from_xbrl(
                text,
                symbol=meta.get("symbol") or p.parent.parent.name,
                is_consolidated=str(meta.get("extra",{}).get("consolidated","")).lower() in ("true","yes","consolidated","1")
            )
            values=asdict(parsed) if is_dataclass(parsed) else dict(vars(parsed))
            rows.append({
                "symbol":meta.get("symbol",""),
                "period_end":values.get("period_end"),
                "broadcast_datetime":meta.get("extra",{}).get("broadcast",""),
                "revision_status":meta.get("extra",{}).get("revision_status",""),
                "source_url":meta.get("source_url",""),
                "source_file":str(p),
                "sha256":sha256(p),
                "parser":"nse-xbrl",
                **values
            })
        except Exception as e:
            failures.append({
                "source_file":str(p),"symbol":meta.get("symbol",""),
                "source_url":meta.get("source_url",""),"sha256":sha256(p),
                "error_type":type(e).__name__,"error":str(e),
                "note":"Not parsed by nse-xbrl; may be an older taxonomy or unsupported attachment."
            })
    out=root/"processed"; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"parsed_integrated_fundamentals.csv",index=False)
    pd.DataFrame(failures).to_csv(out/"parse_failures.csv",index=False)
    print(f"Parsed: {len(rows)}; failures requiring review: {len(failures)}")
    print(f"Outputs: {out/'parsed_integrated_fundamentals.csv'} and {out/'parse_failures.csv'}")
    print("A parser success does not establish point-in-time safety; verify broadcast_datetime and revisions.")

if __name__=="__main__": main()
