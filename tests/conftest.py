import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


# Stage 4I makes the two supported test profiles explicit without rewriting
# historical test modules.  Collection-time labels are deterministic and are
# themselves exercised by tests/test_stage4i_profiles.py.
NQS_MODULES = {
    "test_flax_nqs_optional.py",
    "test_netket_conventions_optional.py",
    "test_netket_optional_builders.py",
    "test_periodic_flax_features_optional.py",
    "test_periodic_netket_operator_optional.py",
    "test_stage3e_parameter_matching.py",
    "test_stage4d_model_builds.py",
    "test_stage4h_transplant.py",
}
SLOW_MODULES = {
    "test_stage4f1_transactional_scheduler.py",
    "test_stage4f1b_semantic_lease.py",
}
OPTIONAL_MODULES = {
    "test_bose_hubbard_limits.py",
    "test_quspin_optional_crosscheck.py",
    "test_tenpy_optional_reference.py",
}


def pytest_collection_modifyitems(items):
    import pytest

    for item in items:
        name = Path(str(item.fspath)).name
        if name in NQS_MODULES:
            item.add_marker(pytest.mark.nqs)
        if name in SLOW_MODULES:
            item.add_marker(pytest.mark.slow)
        if name in OPTIONAL_MODULES:
            item.add_marker(pytest.mark.optional)
