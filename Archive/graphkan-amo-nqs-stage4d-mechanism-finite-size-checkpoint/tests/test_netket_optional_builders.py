import numpy as np
import pytest

from graphkan_amo_nqs.exceptions import DependencyUnavailable
from graphkan_amo_nqs.hamiltonians.netket_builders import build_netket_rydberg_operator
from graphkan_amo_nqs.hamiltonians.rydberg import RydbergParams


def test_netket_optional_builder_constructs_small_operator_when_available():
    """The builder requires an explicit geometry and Rydberg parameter set.

    When NetKet is unavailable, the project must still remain import-safe.
    """
    positions = np.array([[0.0], [1.0]], dtype=float)
    params = RydbergParams(omega=1.0, delta=0.5, c6=1.0)
    try:
        hilbert, operator = build_netket_rydberg_operator(positions, params)
    except DependencyUnavailable:
        pytest.skip("NetKet is optional in the base test environment.")
    assert hilbert is not None
    assert operator is not None
