"""Tests for the FRED loader (HDF5 events + coordinates.txt annotations).

Synthetic HDF5 files are written for each known Prophesee/Metavision
layout so we can verify the loader handles all of them. No real FRED
data is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from ev_drone_detector.data.fred import (
    FREDAnnotation,
    annotations_in_window,
    load_fred_annotations,
    load_fred_events,
)


h5py = pytest.importorskip("h5py")


def _make_event_arrays(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 1280, size=n).astype(np.uint16)
    y = rng.integers(0, 720, size=n).astype(np.uint16)
    t = np.sort(rng.integers(0, 1_000_000, size=n)).astype(np.int64)
    p = rng.integers(0, 2, size=n).astype(np.uint8)
    return x, y, t, p


def _write_layout_cd_structured(path, x, y, t, p):
    """Layout 1: /CD/events as structured dataset."""
    dt = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "u1"), ("t", "<i8")])
    rec = np.empty(len(x), dtype=dt)
    rec["x"], rec["y"], rec["p"], rec["t"] = x, y, p, t
    with h5py.File(path, "w") as f:
        f.create_group("CD").create_dataset("events", data=rec)


def _write_layout_events_structured(path, x, y, t, p):
    """Layout 2: /events as structured dataset."""
    dt = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "u1"), ("t", "<i8")])
    rec = np.empty(len(x), dtype=dt)
    rec["x"], rec["y"], rec["p"], rec["t"] = x, y, p, t
    with h5py.File(path, "w") as f:
        f.create_dataset("events", data=rec)


def _write_layout_events_group(path, x, y, t, p):
    """Layout 3: /events as group with x,y,t,p children."""
    with h5py.File(path, "w") as f:
        g = f.create_group("events")
        g.create_dataset("x", data=x)
        g.create_dataset("y", data=y)
        g.create_dataset("t", data=t)
        g.create_dataset("p", data=p)


def _write_layout_root(path, x, y, t, p):
    """Layout 4: root-level x,y,t,p."""
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("t", data=t)
        f.create_dataset("p", data=p)


@pytest.mark.parametrize(
    "writer",
    [
        _write_layout_cd_structured,
        _write_layout_events_structured,
        _write_layout_events_group,
        _write_layout_root,
    ],
    ids=["cd_structured", "events_structured", "events_group", "root_level"],
)
def test_load_fred_events_layouts(tmp_path, writer):
    """Loader must handle all four known Prophesee HDF5 layouts."""
    x, y, t, p = _make_event_arrays(n=200, seed=1)
    f_path = tmp_path / "events.hdf5"
    writer(f_path, x, y, t, p)

    rx, ry, rt, rp = load_fred_events(f_path)
    assert rx.dtype == np.int64
    assert ry.dtype == np.int64
    assert rt.dtype == np.int64
    assert rp.dtype == np.int64
    assert len(rx) == 200
    assert np.array_equal(rx, x.astype(np.int64))
    assert np.array_equal(ry, y.astype(np.int64))
    assert np.array_equal(rt, t.astype(np.int64))
    assert np.array_equal(rp, p.astype(np.int64))


def test_load_fred_events_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fred_events(tmp_path / "does_not_exist.hdf5")


def test_load_fred_events_unknown_layout_raises(tmp_path):
    """An HDF5 with none of the known layouts should error clearly."""
    path = tmp_path / "bad.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("foo", data=np.arange(10))
        f.create_dataset("bar", data=np.arange(10))
    with pytest.raises(ValueError, match="Could not locate events"):
        load_fred_events(path)


def test_load_fred_events_time_range_slice(tmp_path):
    """time_range_us must restrict the output to events in the window."""
    x = np.arange(10, dtype=np.uint16)
    y = np.arange(10, dtype=np.uint16)
    t = np.arange(10, dtype=np.int64) * 1000  # 0, 1000, 2000, ... us
    p = np.zeros(10, dtype=np.uint8)
    path = tmp_path / "events.hdf5"
    _write_layout_root(path, x, y, t, p)

    rx, ry, rt, rp = load_fred_events(path, time_range_us=(3000, 7000))
    # Events at t=3000, 4000, 5000, 6000 only
    assert len(rt) == 4
    assert rt.tolist() == [3000, 4000, 5000, 6000]


def test_load_fred_events_max_events_cap(tmp_path):
    """max_events should subsample to the requested size."""
    x, y, t, p = _make_event_arrays(n=1000)
    path = tmp_path / "events.hdf5"
    _write_layout_root(path, x, y, t, p)
    rx, ry, rt, rp = load_fred_events(path, max_events=100)
    assert len(rx) == 100
    # Timestamps remain monotonically non-decreasing (sel.sort() guarantees this)
    assert np.all(np.diff(rt) >= 0)


def test_load_fred_annotations_parses_each_line(tmp_path):
    path = tmp_path / "coordinates.txt"
    path.write_text(
        "0.033333: 100, 200, 150, 250, 1, DJI Mini 2\n"
        "0.066666: 105, 205, 155, 255, 1, DJI Mini 2\n"
        "1.500000: 600, 360, 640, 400, 2, Betafpv air75\n"
        "\n"
        "# comment line, should be skipped\n"
    )
    anns = load_fred_annotations(path)
    assert len(anns) == 3
    assert anns[0] == FREDAnnotation(
        t_us=33333, bbox=(100, 200, 150, 250), track_id=1, class_name="DJI Mini 2"
    )
    assert anns[2].t_us == 1_500_000
    assert anns[2].class_name == "Betafpv air75"
    assert anns[2].track_id == 2


def test_load_fred_annotations_sorted_by_time(tmp_path):
    path = tmp_path / "coordinates.txt"
    path.write_text(
        "1.000000: 0, 0, 10, 10, 1, x\n"
        "0.500000: 0, 0, 10, 10, 1, x\n"
        "0.100000: 0, 0, 10, 10, 1, x\n"
    )
    anns = load_fred_annotations(path)
    ts = [a.t_us for a in anns]
    assert ts == sorted(ts)


def test_load_fred_annotations_ignores_malformed_lines(tmp_path):
    path = tmp_path / "coordinates.txt"
    path.write_text(
        "0.033333: 100, 200, 150, 250, 1, drone\n"
        "this line is garbage\n"
        "0.066666: not, valid, bbox, fields, broken\n"
        "1.500000: 600, 360, 640, 400, 2, drone\n"
    )
    anns = load_fred_annotations(path)
    assert len(anns) == 2


def test_load_fred_annotations_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fred_annotations(tmp_path / "nope.txt")


def test_annotations_in_window():
    anns = [
        FREDAnnotation(t_us=0, bbox=(0, 0, 1, 1), track_id=1, class_name="d"),
        FREDAnnotation(t_us=5_000_000, bbox=(0, 0, 1, 1), track_id=1, class_name="d"),
        FREDAnnotation(t_us=8_000_000, bbox=(0, 0, 1, 1), track_id=1, class_name="d"),
        FREDAnnotation(t_us=12_000_000, bbox=(0, 0, 1, 1), track_id=1, class_name="d"),
    ]
    w1 = annotations_in_window(anns, 0, 8_000_000)
    w2 = annotations_in_window(anns, 8_000_000, 16_000_000)
    assert [a.t_us for a in w1] == [0, 5_000_000]
    assert [a.t_us for a in w2] == [8_000_000, 12_000_000]
