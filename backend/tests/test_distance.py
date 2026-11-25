"""Ground geometry tests (pure maths, no model needed)."""
import math

import numpy as np

from skyir.distance import GroundModel, plausible_size


def test_centre_pixel_matches_pitch_geometry():
    g = GroundModel(1000, 640, 512)
    r, f, s, ok = g.project(320, 256, 100, 30)
    assert ok and abs(r) < 1e-9
    assert math.isclose(f, 100 / math.tan(math.radians(30)), rel_tol=1e-9)
    assert math.isclose(s, 100 / math.sin(math.radians(30)), rel_tol=1e-9)


def test_nadir_is_symmetric_and_scaled_by_altitude():
    g = GroundModel(1000, 640, 512)
    r, f, s, ok = g.project([420, 320], [256, 156], 50, 90)
    assert ok.all()
    assert np.allclose(r, [5.0, 0.0]) and np.allclose(f, [0.0, 5.0], atol=1e-9)  # 100 px * 50 m / 1000 px


def test_rays_above_horizon_are_invalid():
    g = GroundModel(1000, 640, 512)
    _, f, _, ok = g.project(320, 0, 100, 5)  # top row is ~9 deg above the optical axis
    assert not ok and np.isnan(f)


def test_ground_velocity_recovers_metric_speed():
    g = GroundModel(1000, 640, 512)
    # at nadir from 50 m, 1 px = 5 cm; 4 px/frame at 10 fps = 2 m/s towards the top of the image
    vr, vf = g.ground_velocity(320, 256, 0, -4, 50, 90, 10)
    assert math.isclose(float(vr), 0, abs_tol=1e-9) and math.isclose(float(vf), 2.0, rel_tol=1e-6)


def test_footprint_is_clipped_polygon():
    g = GroundModel(1000, 640, 512)
    fp = g.footprint(100, 20, max_range_m=1000)
    assert len(fp) > 4 and max(f for _, f in fp) <= 1000 * 1.2 and min(f for _, f in fp) > 0


def test_size_ranges():
    assert plausible_size("Person", 1.7) and not plausible_size("Person", 5.0)
    assert plausible_size("Bicycle", 3.2, slack_m=0.5)
    assert plausible_size("DontCare", 100.0)  # unknown classes are never rejected
