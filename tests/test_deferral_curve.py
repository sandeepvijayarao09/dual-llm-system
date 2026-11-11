"""router.deferral_curve — the sweep, the oracle baseline and the area."""

import numpy as np
import pytest

from router.deferral_curve import at, auc, curve, oracle

# 4 queries: router confidence, router right?, classifier right?
CONF = np.array([0.55, 0.62, 0.80, 0.95])
ML_OK = np.array([False, True, True, True])
CLF_OK = np.array([True, False, True, False])


def test_endpoints_are_router_alone_and_classifier_alone():
    pts = curve(CONF, ML_OK, CLF_OK)
    assert pts[0][1:] == (0.0, ML_OK.mean())          # never defer
    assert pts[-1][1:] == (1.0, CLF_OK.mean())        # always defer


def test_operating_point_defers_below_the_threshold():
    deferred, acc = at(0.65, CONF, ML_OK, CLF_OK)
    assert deferred == 0.5                            # 0.55 and 0.62
    assert acc == pytest.approx((1 + 0 + 1 + 1) / 4)


def test_oracle_defers_helpful_queries_first_and_peaks_above_both():
    orc = oracle(ML_OK, CLF_OK)
    assert orc[1] == (0.25, 1.0)                      # defer the one the classifier fixes
    assert max(a for _, a in orc) >= max(ML_OK.mean(), CLF_OK.mean())


def test_auc_of_a_flat_curve_is_its_height():
    assert auc([(0.0, 0.9), (0.5, 0.9), (1.0, 0.9)]) == pytest.approx(0.9)
