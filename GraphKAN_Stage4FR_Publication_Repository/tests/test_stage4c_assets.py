from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def test_stage4c_config_is_frozen_patch():
    cfg=yaml.safe_load((ROOT/'configs/netket_stage4c_hard_patch.yaml').read_text())
    assert cfg['stage']=='4C'
    assert cfg['chains']['n_sites']==[6,8]
    assert cfg['parameters']['delta']==[1.5,2.0,2.5]
    assert cfg['parameters']['c6']==[3.0,4.0,5.0]
    assert len(cfg['training']['seeds'])==5
    assert 'smooth_endpoint_sum_21p' not in cfg['model_specs']
    assert cfg['stage4c_gate']['min_candidate_exact_valid_rate']==0.90
def test_stage4c_runner_and_summary_are_native_and_runtime_corrected():
    runner=(ROOT/'scripts/run_stage4c_patch.py').read_text(); summary=(ROOT/'scripts/summarize_stage4c.py').read_text()
    assert 'stage4c_metrics.csv' in runner
    assert 'scientific_validity_exact' in runner
    assert '--resume' in runner
    assert 'control_runtime_ratio' in summary
    assert 'control/candidate wall_seconds <= 1.25' in summary
def test_stage4c_vmc_batch_logging_exists():
    assert 'tee "$LOG"' in (ROOT/'scripts/run_stage4c_batch.sh').read_text()
