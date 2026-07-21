#!/usr/bin/env python
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
from graphkan_amo_nqs.reporting.latex_report import make_report_tex


def main():
    ap=argparse.ArgumentParser(description="Generate and compile LaTeX analysis report.")
    ap.parse_args()
    tex=make_report_tex(Path("."))
    paper=tex.parent
    cmd=["latexmk","-pdf","-interaction=nonstopmode","-halt-on-error",tex.name]
    proc=subprocess.run(cmd,cwd=paper,text=True,capture_output=True)
    (paper/"production_analysis_report_build.log").write_text(proc.stdout+"\n"+proc.stderr,encoding="utf-8")
    if proc.returncode!=0:
        print(proc.stdout[-2000:]); print(proc.stderr[-2000:]); raise SystemExit(proc.returncode)
    print(f"report written to {paper/'production_analysis_report.pdf'}")

if __name__=="__main__":
    main()
