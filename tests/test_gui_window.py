"""Headless (offscreen) tests of the master window: scan -> table rows,
check cascade, optics spread, and a real Start run over a synthetic SER."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from lunation.gui.window import (COL_DRZ, COL_NAME, COL_PHASE,  # noqa: E402
                                 COL_PROGRESS, MasterWindow)

from .test_stack_e2e import _make_capture  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp, tmp_path):
    # keep QSettings out of the real registry: file-backed store in tmp
    s = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    w = MasterWindow(str(tmp_path), settings=s)
    w.out_edit.setText(str(tmp_path / "out").replace("\\", "/"))
    yield w
    w.scheduler = None
    w.close()


def _capture_tree(tmp_path):
    d = tmp_path / "caps" / "2026-01-01"
    d.mkdir(parents=True)
    ser, _, _ = _make_capture(d)
    dst = str(d / "2026-01-01-Capture_L.ser")
    os.replace(ser, dst)
    return str(tmp_path / "caps"), dst


def test_scan_builds_grouped_rows(window, tmp_path):
    caps, _ = _capture_tree(tmp_path)
    window.paths.addItem(caps)
    window.scan(auto_align=False)
    g = window._groups["2026-01-01"]
    assert g.childCount() == 1
    leaf = g.child(0)
    assert leaf.text(COL_NAME) == "L"
    assert g.text(COL_PHASE) != ""
    assert leaf.text(COL_DRZ) == "2x"  # default 440mm/3.76um rig
    # rescan is append-only dedup
    window.scan(auto_align=False)
    assert g.childCount() == 1


def test_group_check_cascades(window, tmp_path):
    caps, _ = _capture_tree(tmp_path)
    window.paths.addItem(caps)
    window.scan(auto_align=False)
    g = window._groups["2026-01-01"]
    g.setCheckState(COL_NAME, Qt.Unchecked)
    assert g.child(0).checkState(COL_NAME) == Qt.Unchecked
    assert window._checked_leaves() == []


def test_optics_spread_scopes(window, tmp_path):
    caps, _ = _capture_tree(tmp_path)
    window.paths.addItem(caps)
    window.scan(auto_align=False)
    leaf = window._groups["2026-01-01"].child(0)
    window.spread_optics(leaf, "focalLength", 200.0, scope=0)
    assert window._entry[leaf]["optics"]["focalLength"] == 200.0
    # 200mm f/2.86 @3.76um -> Q~4.6 -> clamped 3x
    assert leaf.text(COL_DRZ) == "3x"


def test_groups_ordered_by_phase(window):
    # ages: 04-21 -> 4.7d, 06-30 -> 15.6d, 06-05 -> 20.2d
    for d in ("2026-06-30", "2026-04-21", "2026-06-05", ""):
        window.add_row(d, "SER", f"x_{d or 'un'}.ser",
                       {"type": "ser", "path": f"x_{d or 'un'}.ser",
                        "label": "L", "key": f"ser:{d or 'un'}"})
    order = [window.tree.topLevelItem(i).text(0)
             for i in range(window.tree.topLevelItemCount())]
    assert order == ["2026-04-21", "2026-06-30", "2026-06-05",
                     "(undated)"]  # thinnest -> fullest, undated last


def test_blink_cycles_and_highlights(window, tmp_path):
    from .test_blink import disk_ser

    for d, off in (("2026-04-21", -12), ("2026-06-30", 10)):
        cap = tmp_path / "caps" / d
        cap.mkdir(parents=True)
        disk_ser(cap / f"{d}-Capture_L.ser", dx=off)
    window.paths.addItem(str(tmp_path / "caps"))
    window.scan(auto_align=False)
    window.build_blink_set()
    assert [b["key"] for b in window._blink] == ["2026-04-21",
                                                 "2026-06-30"]
    # shown frame's group is highlighted; stepping follows in phase order
    assert window.tree.currentItem() is window._blink[0]["item"]
    window._show_blink(1)
    assert window.tree.currentItem() is window._blink[1]["item"]
    window._show_blink(1)  # wraps
    assert window.tree.currentItem() is window._blink[0]["item"]
    assert window.preview.pixmap() and not window.preview.pixmap().isNull()
    # play toggles; a manual selection stops it
    window._toggle_blink_play()
    assert window.blink_timer.isActive()
    window.tree.setCurrentItem(window.tree.topLevelItem(1))  # user click
    assert not window.blink_timer.isActive()


def test_scan_auto_assembles_aligned_preview(window, tmp_path):
    """Scan builds the table AND the aligned blink set in one go; a
    rescan that finds nothing new keeps the existing set."""
    from .test_blink import disk_ser

    for d, off in (("2026-04-21", -12), ("2026-06-30", 10)):
        cap = tmp_path / "caps" / d
        cap.mkdir(parents=True)
        disk_ser(cap / f"{d}-Capture_L.ser", dx=off)
    window.paths.addItem(str(tmp_path / "caps"))
    window.scan()  # no Align press needed
    assert [b["key"] for b in window._blink] == ["2026-04-21",
                                                 "2026-06-30"]
    assert window.preview.pixmap() and not window.preview.pixmap().isNull()
    # the bar reported the build, then was handed back to the scheduler
    assert window.overall.format() == "%p%  overall"
    assert window.overall.maximum() == 1000
    first = window._blink
    window.scan()  # nothing new -> no rebuild
    assert window._blink is first


def test_lunation_gets_no_table_row(window, tmp_path):
    """The gif is an output, not an input: no row, jobs still scheduled."""
    caps, _ = _capture_tree(tmp_path)
    window.paths.addItem(caps)
    window.scan(auto_align=False)
    window.lun_check.setChecked(True)
    window.start()
    window.timer.stop()  # never tick: don't actually launch children
    assert "(undated)" not in window._groups
    kinds = sorted(j.kind for j in window.scheduler.jobs)
    assert kinds == ["encode", "gif", "stack"]
    # gif/encode status rides the overall bar, not a row
    window.scheduler.tick = lambda: None  # poll nothing
    window._tick()
    assert "lunation:" in window.overall.format()


def test_start_runs_synthetic_session(window, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr("lunation.gui.window.QDesktopServices",
                        type("D", (), {"openUrl":
                                       staticmethod(opened.append)}))
    caps, _ = _capture_tree(tmp_path)
    window.paths.addItem(caps)
    window.scan(auto_align=False)
    window.start()
    assert window.scheduler is not None
    assert not window.start_btn.isEnabled()
    deadline = time.time() + 120
    while not window.scheduler.done():
        window._tick()
        assert time.time() < deadline, "scheduler did not finish"
        time.sleep(0.05)
    window._tick()  # final row refresh

    from lunation.master.job import State

    assert all(j.state == State.OK for j in window.scheduler.jobs)
    leaf = window._groups["2026-01-01"].child(0)
    assert "done" in leaf.text(COL_PROGRESS)
    out = tmp_path / "out" / "2026-01-01" / "L1_stack.xisf"
    assert out.exists()
    # discovered config stored WITH the output, in this run's
    # lunation_<id> dir (PIPP/AutoStakkert model)
    import glob
    import json

    hits = glob.glob(str(tmp_path / "out" / "lunation_*"
                         / "2026-01-01.json"))
    assert len(hits) == 1
    cfg = json.load(open(hits[0]))
    assert cfg["defaults"]["drizzle"] == 2
    assert cfg["jobs"][0]["id"] == "L1"
    # the output dir opened exactly once when the run finished
    assert len(opened) == 1
    assert opened[0].toLocalFile().rstrip("/") == window.out_root
