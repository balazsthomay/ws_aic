"""v39 unit tests — state-vector dim consistency + seat-success filter.

These tests cover the pure-Python parts of v39 that don't require ROS
imports. End-to-end recording behavior is verified via the Modal smoke
test (`modal_collect_demos.py::smoke_test`).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def _make_episode(*, push_contact_step: int | None = -1,
                  delta_tail_fz: float | None = 1.0,
                  T: int = 20, state_dim: int = 32) -> dict:
    """Build a minimal in-memory episode dict matching `_load_one`."""
    return {
        "scene_id": "test",
        "states": np.zeros((T, state_dim), dtype=np.float32),
        "actions": np.zeros((T, 6), dtype=np.float32),
        "wrench_z": np.zeros((T,), dtype=np.float32),
        "images_left_camera": np.zeros((T, 4, 4, 3), dtype=np.uint8),
        "images_center_camera": np.zeros((T, 4, 4, 3), dtype=np.uint8),
        "images_right_camera": np.zeros((T, 4, 4, 3), dtype=np.uint8),
        "port_world": np.zeros(3, dtype=np.float32),
        "cable_type": "sfp_sc",
        "port_type": "sfp",
        "push_contact_step": push_contact_step,
        "tail_fz": 0.0,
        "delta_tail_fz": delta_tail_fz,
    }


# --- _is_seated: filter logic ---


def test_is_seated_accepts_full_push_with_low_tail():
    from aggregate_seating_demos import _is_seated
    ep = _make_episode(push_contact_step=-1, delta_tail_fz=1.0)
    assert _is_seated(ep) is True


def test_is_seated_rejects_bottomed_out():
    from aggregate_seating_demos import _is_seated
    ep = _make_episode(push_contact_step=37, delta_tail_fz=1.0)
    assert _is_seated(ep) is False


def test_is_seated_rejects_skipped_push():
    from aggregate_seating_demos import _is_seated
    ep = _make_episode(push_contact_step=-2, delta_tail_fz=0.0)
    assert _is_seated(ep) is False


def test_is_seated_rejects_high_tail_fz():
    from aggregate_seating_demos import _is_seated, SEATED_DELTA_TAIL_FZ_MAX
    ep = _make_episode(
        push_contact_step=-1,
        delta_tail_fz=SEATED_DELTA_TAIL_FZ_MAX + 1.0,
    )
    assert _is_seated(ep) is False


def test_is_seated_handles_legacy_episodes_missing_metrics():
    """Old v38 episodes have no metrics — fall back to True so filter
    doesn't silently drop them; the state_dim check rejects them anyway."""
    from aggregate_seating_demos import _is_seated
    ep = _make_episode(push_contact_step=None, delta_tail_fz=None)
    assert _is_seated(ep) is True


def test_is_seated_accepts_large_negative_delta_insertion_signal():
    """Large NEGATIVE delta_tail_fz indicates the cable inserted and the
    gripper continued compressing past the seat. From observed v39 smoke:
    delta down to -120 N is a strong insertion signal, not a failure."""
    from aggregate_seating_demos import _is_seated
    ep = _make_episode(push_contact_step=-1, delta_tail_fz=-120.0)
    assert _is_seated(ep) is True


def test_is_seated_rejects_positive_delta_faceplate_compression():
    """Positive delta means the gripper is still pressing on the faceplate
    after push — cable didn't insert."""
    from aggregate_seating_demos import _is_seated, SEATED_DELTA_TAIL_FZ_MAX
    ep = _make_episode(
        push_contact_step=-1,
        delta_tail_fz=SEATED_DELTA_TAIL_FZ_MAX + 0.5,
    )
    assert _is_seated(ep) is False


# --- State-dim layout consistency (no ROS imports needed) ---


def _read_wrench_from_obs(obs) -> np.ndarray:
    """Local copy of _read_wrench helper to avoid pulling in ROS deps."""
    try:
        w = obs.wrist_wrench.wrench
        return np.array([
            w.force.x, w.force.y, w.force.z,
            w.torque.x, w.torque.y, w.torque.z,
        ], dtype=np.float64)
    except Exception:
        return np.zeros(6, dtype=np.float64)


def test_wrench_helper_returns_six_floats():
    obs = SimpleNamespace(
        wrist_wrench=SimpleNamespace(wrench=SimpleNamespace(
            force=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            torque=SimpleNamespace(x=4.0, y=5.0, z=6.0),
        ))
    )
    w = _read_wrench_from_obs(obs)
    assert w.shape == (6,)
    assert np.allclose(w, [1, 2, 3, 4, 5, 6])


def test_wrench_helper_falls_back_to_zeros_on_missing_field():
    obs = SimpleNamespace()
    w = _read_wrench_from_obs(obs)
    assert w.shape == (6,)
    assert np.all(w == 0.0)


def test_state_dim_constants_consistent():
    """Both the collector and inference policy must declare STATE_DIM = 32.

    We can't import the modules (ROS dep), so grep the source files.
    """
    coll_src = (
        ROOT
        / "src/aic/aic_example_policies/aic_example_policies/ros/SeatingCollector.py"
    ).read_text()
    inf_src = (
        ROOT
        / "src/aic/aic_example_policies/aic_example_policies/ros/DINOv2ACT.py"
    ).read_text()
    assert "STATE_DIM = 32" in coll_src, "SeatingCollector must declare STATE_DIM=32"
    assert "STATE_DIM = 32" in inf_src, "DINOv2ACT must declare STATE_DIM=32"
    assert "WRENCH_DIM = 6" in coll_src
    assert "WRENCH_DIM = 6" in inf_src


def test_aggregator_default_state_dim_is_32():
    from aggregate_seating_demos import aggregate
    import inspect
    sig = inspect.signature(aggregate)
    assert sig.parameters["expected_state_dim"].default == 32


# --- Aggregator end-to-end on synthetic episodes ---


def test_filter_seated_drops_failures(tmp_path: Path):
    """Run the aggregator on synthetic episodes covering the failure modes
    and the insertion-signal mode, verify --filter-seated keeps only the
    seated ones."""
    from aggregate_seating_demos import aggregate

    in_dir = tmp_path / "episodes"
    in_dir.mkdir()
    out_path = tmp_path / "out.npz"

    def _write(name: str, *, push_contact_step: int, delta_tail_fz: float):
        T = 30
        np.savez_compressed(
            in_dir / f"episode_{name}.npz",
            scene_id=np.array(name),
            states=np.zeros((T, 32), dtype=np.float32),
            actions=np.zeros((T, 6), dtype=np.float32),
            wrench=np.zeros((T, 6), dtype=np.float32),
            wrench_z=np.zeros((T,), dtype=np.float32),
            images_left_camera=np.zeros((T, 4, 4, 3), dtype=np.uint8),
            images_center_camera=np.zeros((T, 4, 4, 3), dtype=np.uint8),
            images_right_camera=np.zeros((T, 4, 4, 3), dtype=np.uint8),
            port_world=np.zeros(3, dtype=np.float32),
            cable_type=np.array("sfp_sc"),
            port_type=np.array("sfp"),
            target_module_name=np.array("nic_rail_1"),
            port_name=np.array("port_sfp_0"),
            gripper_offset=np.zeros(3, dtype=np.float32),
            push_contact_step=np.array(push_contact_step, dtype=np.int32),
            tail_fz=np.array(0.0, dtype=np.float32),
            delta_tail_fz=np.array(delta_tail_fz, dtype=np.float32),
            fz_baseline=np.array(0.0, dtype=np.float32),
            lateral_noise_xy=np.array([0.01, -0.01], dtype=np.float32),
        )

    # Good cases (kept):
    _write("good_small_neg", push_contact_step=-1, delta_tail_fz=-3.0)
    _write("good_big_neg",   push_contact_step=-1, delta_tail_fz=-100.0)
    # Bad cases (dropped):
    _write("bad_bottomed",   push_contact_step=15, delta_tail_fz=1.0)
    _write("bad_skipped",    push_contact_step=-2, delta_tail_fz=0.0)
    _write("bad_compressed", push_contact_step=-1, delta_tail_fz=20.0)

    aggregate(in_dir, out_path, filter_seated=True)

    z = np.load(out_path, allow_pickle=True)
    kept = sorted(str(s) for s in z["scene_ids"])
    assert kept == ["good_big_neg", "good_small_neg"], f"got {kept}"


def test_aggregator_without_filter_keeps_all(tmp_path: Path):
    from aggregate_seating_demos import aggregate

    in_dir = tmp_path / "episodes"
    in_dir.mkdir()
    out_path = tmp_path / "out.npz"

    for name, pcs, dtf in (("a", -1, 1.0), ("b", 5, 9.0), ("c", -2, 0.0)):
        T = 30
        np.savez_compressed(
            in_dir / f"episode_{name}.npz",
            scene_id=np.array(name),
            states=np.zeros((T, 32), dtype=np.float32),
            actions=np.zeros((T, 6), dtype=np.float32),
            wrench_z=np.zeros((T,), dtype=np.float32),
            images_left_camera=np.zeros((T, 4, 4, 3), dtype=np.uint8),
            images_center_camera=np.zeros((T, 4, 4, 3), dtype=np.uint8),
            images_right_camera=np.zeros((T, 4, 4, 3), dtype=np.uint8),
            port_world=np.zeros(3, dtype=np.float32),
            cable_type=np.array("sfp_sc"),
            port_type=np.array("sfp"),
            push_contact_step=np.array(pcs, dtype=np.int32),
            tail_fz=np.array(0.0, dtype=np.float32),
            delta_tail_fz=np.array(dtf, dtype=np.float32),
        )

    aggregate(in_dir, out_path, filter_seated=False)

    z = np.load(out_path, allow_pickle=True)
    assert int(z["states"].shape[0]) == 3
