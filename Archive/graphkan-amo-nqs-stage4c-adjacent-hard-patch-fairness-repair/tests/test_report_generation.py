from pathlib import Path
from graphkan_amo_nqs.reporting.latex_report import make_report_tex

def test_report_tex_generation(tmp_path):
    (tmp_path/"paper").mkdir()
    tex=make_report_tex(tmp_path)
    assert tex.exists()
    assert "Production Analysis" in tex.read_text()
