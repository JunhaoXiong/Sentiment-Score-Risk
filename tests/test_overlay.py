import numpy as np
import pandas as pd
import pytest

from ssr.risk import overlay


def test_step_smooth_flat():
    idx = pd.date_range("2020-01-01", periods=4)
    var = pd.Series([1.0, 1.0, 2.0, 2.0], index=idx)
    z = pd.Series([0.0, -2.5, -1.0, -3.0], index=idx)
    np.testing.assert_allclose(overlay.step(var, z, 1.5), [1.0, 1.5, 2.0, 3.0])
    np.testing.assert_allclose(overlay.smooth(var, z, 0.2), [1.0, 1.3, 2.0, 2.8])
    np.testing.assert_allclose(overlay.flat(var, 0.1), [1.1, 1.1, 2.2, 2.2])


def test_missing_z_means_no_uplift_and_matching_flat():
    idx = pd.date_range("2020-01-01", periods=3)
    var = pd.Series([1.0, 2.0, 3.0], index=idx)
    z = pd.Series([-3.0], index=idx[:1])
    s = overlay.smooth(var, z, 0.5)
    assert list(s) == [2.0, 2.0, 3.0]
    u = overlay.matching_flat_uplift(var, s)
    assert overlay.flat(var, u).mean() == pytest.approx(s.mean())
