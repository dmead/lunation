"""The master window — ports pjsr/master/UI.jsh (MasterDialog) and the
session-picker additions of pjsr/lunar-master.js.

Same architecture as the original dialog: ONE two-level table (date-group
parents, checkable item leaves), per-row progress cells aggregating the
row's bound jobs (text block-glyph bars, like the original — TreeBox cell
icons squashed real bars, and they read cleanly at any DPI), preview pane
to the right, log tail below following the selected entry's most
interesting job, and a 0.75 s QTimer driving Scheduler.tick() in the event
loop — no worker threads; the scheduler is non-blocking by design.

Deviations from the original (deliberate):
- no survey verdict column: the survey pipeline is out of the port's scope;
  failure reasons surface in the progress cell + tooltip instead.
- settings (search paths, output root) persist via QSettings instead of
  local-settings.json.
- default concurrent jobs is 1, not 3: a Python stack job already fans out
  over frames internally, so serial jobs replace the child-PI concurrency.
"""

import os
import time

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog,
                               QDoubleSpinBox, QFileDialog, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSpinBox, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from ..master.discovery import (default_optics, derive_drizzle,
                                find_finish_config, new_run_dir, phase_name,
                                plate_scale, scan_search_paths, ser_channel,
                                session_from_sers, write_discovered_config,
                                write_prep_config)
from ..assemble.collect import lunar_age
from ..master.job import State
from ..master.pipeline import lunation_jobs, prep_job, session_jobs
from ..master.scheduler import Scheduler
from .blink import build_blink, pick_group_ser
from .model import fmt_elapsed, overall_fraction, rollup
from .preview import array_to_qimage, preview_qimage

TICK_MS = 750
BAR_CELLS = 12

COL_NAME, COL_PHASE, COL_TYPE, COL_FL, COL_PX, COL_DRZ, COL_SOURCE, \
    COL_PROGRESS, COL_TIME = range(9)
HEADERS = ["date / item", "phase", "type", "FL mm", "px µm", "drizzle",
           "source", "progress", "time"]

STATE_COLORS = {  # UI.jsh:31-40
    State.PENDING: "#5A5A5A", State.READY: "#7A7A48",
    State.LAUNCHING: "#3C6EA5", State.RUNNING: "#3C6EA5",
    State.OK: "#3F8F3F", State.FAILED: "#A43535",
    State.CANCELLED: "#8A6A2A", State.SKIPPED: "#4A4A4A",
}


def text_bar(fraction: float) -> str:
    full = round(BAR_CELLS * max(0.0, min(1.0, fraction)))
    return "▮" * full + "▯" * (BAR_CELLS - full)


class OpticsPrompt(QDialog):
    """Edit FL/px with an apply scope (UI.jsh:138-189)."""

    SCOPES = ["this item only", "all items on this date",
              "all items in the table"]

    def __init__(self, parent, title, value, unit, decimals):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.num = QDoubleSpinBox(self)
        self.num.setSuffix(f" {unit}")
        self.num.setDecimals(decimals)
        self.num.setRange(0.1 if decimals else 10, 100000)
        self.num.setValue(value)
        self.scope = QComboBox(self)
        self.scope.addItems(self.SCOPES)
        ok = QPushButton("OK", self)
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel", self)
        cancel.clicked.connect(self.reject)
        btns = QHBoxLayout()
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(cancel)
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("apply to:"))
        row.addWidget(self.scope, 1)
        lay.addWidget(self.num)
        lay.addLayout(row)
        lay.addLayout(btns)


class MasterWindow(QWidget):
    def __init__(self, output: str | None = None,
                 settings: QSettings | None = None):
        super().__init__()
        self.setWindowTitle("Lunation")
        self.settings = settings or QSettings("lunation", "lunation")
        self.scheduler: Scheduler | None = None
        self.bound: list[tuple[QTreeWidgetItem, list]] = []
        self._lun_jobs: list = []
        self._opened_output = False
        self.selected_job = None
        self._blink: list[dict] = []  # [{item, key, qimage}]
        self._blink_i = 0
        self._blink_guard = False
        self._known_keys: set[str] = set()
        self._groups: dict[str, QTreeWidgetItem] = {}
        # entries live OUTSIDE Qt item-data: setData deep-copies dicts into
        # QVariant maps, so mutations (optics edits, job binding) would be
        # silently lost
        self._entry: dict[QTreeWidgetItem, dict] = {}

        # no pipeline root: each Start mints <output>/lunation_<id>/ and
        # stores the generated configs WITH the output (the PIPP/
        # AutoStakkert model). equipment.json is read from the output dir.
        out = (output or self.settings.value("outputRoot")
               or f"{os.getcwd()}/out").replace("\\", "/").rstrip("/")
        self._build_ui(out)
        self._restore_settings()

    @property
    def out_root(self) -> str:
        return self.out_edit.text().replace("\\", "/").rstrip("/")

    # ---- layout ---------------------------------------------------------

    def _build_ui(self, out: str) -> None:
        # paths + settings
        self.out_edit = QLineEdit(out, self)
        out_browse = QPushButton("…", self)
        out_browse.clicked.connect(lambda: self._browse_into(self.out_edit))

        self.paths = QListWidget(self)
        self.paths.setMaximumHeight(64)
        add_path = QPushButton("Add search path", self)
        add_path.clicked.connect(self._add_search_path)
        del_path = QPushButton("Remove", self)
        del_path.clicked.connect(self._remove_search_path)
        scan = QPushButton("Scan", self)
        scan.setToolTip("find captures on the search paths and assemble "
                        "the aligned preview")
        # lambda: QPushButton.clicked passes checked=False, which would
        # land in auto_align
        scan.clicked.connect(lambda: self.scan())
        add_file = QPushButton("Add file(s)…", self)
        add_file.clicked.connect(self._add_files)

        self.jobs_spin = QSpinBox(self)
        self.jobs_spin.setRange(1, 6)
        self.jobs_spin.setValue(1)
        self.jobs_spin.setToolTip(
            "concurrent heavy jobs — each stack job already runs frame "
            "workers in parallel, so 1 is usually right. Live during a "
            "run: lowering suspends the newest jobs, raising resumes/"
            "launches.")
        self.jobs_spin.valueChanged.connect(
            lambda v: self.scheduler and self.scheduler.set_cap("heavy", v))
        self.lun_check = QCheckBox("render lunation + encode", self)
        self.px_spin = QSpinBox(self)
        self.px_spin.setRange(240, 2300)
        self.px_spin.setValue(1080)

        # the table
        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(len(HEADERS))
        self.tree.setHeaderLabels(HEADERS)
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.currentItemChanged.connect(self._on_select)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        fm = self.tree.fontMetrics()
        widths = {COL_NAME: "2026-04-21MMMM", COL_PHASE: "waning crescentM",
                  COL_TYPE: "TIFFM", COL_FL: "8888M", COL_PX: "88.88M",
                  COL_DRZ: "drizzleM",
                  COL_SOURCE: "2026-04-21-8888_8-Capture_L.serM",
                  COL_PROGRESS: text_bar(1) + "MM100% assembleM",
                  COL_TIME: "88:88M"}
        for c, s in widths.items():
            self.tree.setColumnWidth(c, fm.horizontalAdvance(s))

        # preview
        self.preview = QLabel("select an item", self)
        self.preview.setFixedWidth(340)
        self.preview.setMinimumHeight(300)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet(
            "background: #161616; color: #BBBBBB;")
        self.preview.setWordWrap(True)

        # overall + log + buttons
        self.overall = QProgressBar(self)
        self.overall.setRange(0, 1000)
        self.overall.setFormat("%p%  overall")
        self.log_box = QPlainTextEdit(self)
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(160)
        self.log_box.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 8pt;")

        self.start_btn = QPushButton("Start", self)
        self.start_btn.clicked.connect(self.start)
        self.pause_btn = QPushButton("Pause", self)
        self.pause_btn.clicked.connect(self._toggle_pause)
        cancel_btn = QPushButton("Cancel entry", self)
        cancel_btn.clicked.connect(self._cancel_entry)
        cancel_all_btn = QPushButton("Cancel all", self)
        cancel_all_btn.clicked.connect(
            lambda: self.scheduler and self.scheduler.cancel_all())
        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.close)

        # numbered areas, top to bottom (fixed order):
        # ① inputs  ② processing  ③ options  ④ output  ⑤ run
        g1 = QGroupBox("① choose inputs", self)
        paths_row = QHBoxLayout(g1)
        paths_row.addWidget(self.paths, 1)
        pb = QVBoxLayout()
        for b in (add_path, del_path, scan, add_file):
            pb.addWidget(b)
        paths_row.addLayout(pb)

        # blink controls (one aligned frame per date group, phase order)
        self.blink_btn = QPushButton("Blink", self)
        self.blink_btn.setToolTip(
            "pick one decent frame per date group, disk-align, and cycle "
            "— like PixInsight Blink with one frame per day")
        self.blink_btn.clicked.connect(lambda: self.build_blink_set())
        self.align_btn = QPushButton("Align", self)
        self.align_btn.setToolTip(
            "blink + the output's rotation chain (physics-seeded NCC) in "
            "phase order — a draft of how the gif will play; registration "
            "only, no processing or finishing")
        self.align_btn.clicked.connect(
            lambda: self.build_blink_set(align=True))
        blink_prev = QPushButton("◀", self)
        blink_prev.clicked.connect(lambda: self._show_blink(-1))
        blink_next = QPushButton("▶", self)
        blink_next.clicked.connect(lambda: self._show_blink(1))
        self.blink_play_btn = QPushButton("Play", self)
        self.blink_play_btn.clicked.connect(self._toggle_blink_play)
        self.blink_timer = QTimer(self)
        self.blink_timer.setInterval(667)  # the lunation's 1.5 fps
        self.blink_timer.timeout.connect(lambda: self._show_blink(1))

        g2 = QGroupBox("② processing", self)
        proc = QVBoxLayout(g2)
        table_row = QHBoxLayout()
        table_row.addWidget(self.tree, 1)
        prevcol = QVBoxLayout()
        prevcol.addWidget(self.preview, 1)
        blink_row = QHBoxLayout()
        for b in (self.blink_btn, self.align_btn, blink_prev, blink_next,
                  self.blink_play_btn):
            blink_row.addWidget(b)
        prevcol.addLayout(blink_row)
        table_row.addLayout(prevcol)
        proc.addLayout(table_row, 1)
        proc.addWidget(self.log_box)
        proc.addWidget(self.overall)

        g3 = QGroupBox("③ options", self)
        settings_row = QHBoxLayout(g3)
        for w in (QLabel("max concurrent jobs:"), self.jobs_spin,
                  self.lun_check, QLabel("outPx:"), self.px_spin):
            settings_row.addWidget(w)
        settings_row.addStretch()

        g4 = QGroupBox("④ output", self)
        out_row = QHBoxLayout(g4)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(out_browse)

        g5 = QGroupBox("⑤ run", self)
        buttons = QHBoxLayout(g5)
        for b in (self.start_btn, self.pause_btn, cancel_btn,
                  cancel_all_btn):
            buttons.addWidget(b)
        buttons.addStretch()
        buttons.addWidget(close_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(g1)
        lay.addWidget(g2, 1)
        lay.addWidget(g3)
        lay.addWidget(g4)
        lay.addWidget(g5)
        self.resize(1400, 900)

        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._tick)

    # ---- settings -------------------------------------------------------

    def _restore_settings(self) -> None:
        for p in self.settings.value("searchPaths") or []:
            self.paths.addItem(p)
        out = self.settings.value("outputRoot")
        if out:
            self.out_edit.setText(out)

    def _save_settings(self) -> None:
        self.settings.setValue("searchPaths", [
            self.paths.item(i).text() for i in range(self.paths.count())])
        self.settings.setValue("outputRoot", self.out_edit.text())

    def _browse_into(self, edit: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "choose", edit.text())
        if d:
            edit.setText(d.replace("\\", "/"))
            self._save_settings()

    def _add_search_path(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "add search path")
        if d:
            self.paths.addItem(d.replace("\\", "/"))
            self._save_settings()

    def _remove_search_path(self) -> None:
        for it in self.paths.selectedItems():
            self.paths.takeItem(self.paths.row(it))
        self._save_settings()

    # ---- table build ----------------------------------------------------

    @staticmethod
    def _group_age(key: str) -> float:
        import re

        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", key):
            return lunar_age(key)
        return float("inf")  # undated groups sink to the bottom

    def _date_group(self, date: str) -> QTreeWidgetItem:
        key = date or "(undated)"
        g = self._groups.get(key)
        if g is None:
            g = QTreeWidgetItem([key])
            g.setFlags(g.flags() | Qt.ItemIsUserCheckable)
            g.setCheckState(COL_NAME, Qt.Checked)
            g.setText(COL_PHASE, phase_name(date))
            # groups sit in OUTPUT order: by synodic age, thinnest ->
            # fullest, exactly how the lunation orders its frames
            age = self._group_age(key)
            idx = 0
            while (idx < self.tree.topLevelItemCount()
                   and self._group_age(
                       self.tree.topLevelItem(idx).text(COL_NAME)) <= age):
                idx += 1
            self.tree.insertTopLevelItem(idx, g)
            g.setExpanded(True)
            self._entry[g] = {"group": True}
            self._groups[key] = g
        return g

    def add_row(self, date: str, type_: str, source: str,
                entry: dict) -> QTreeWidgetItem | None:
        if entry["key"] in self._known_keys:
            return None
        self._known_keys.add(entry["key"])
        g = self._date_group(date)
        it = QTreeWidgetItem(g)
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
        it.setCheckState(COL_NAME, Qt.Checked)  # added deliberately -> run
        it.setText(COL_NAME, entry.get("label", ""))
        it.setText(COL_PHASE, phase_name(date))
        it.setText(COL_TYPE, type_)
        it.setText(COL_SOURCE, os.path.basename(source))
        it.setToolTip(COL_SOURCE, source)
        entry["optics"] = default_optics(self.out_root)
        self._entry[it] = entry
        self._apply_optics(it)
        return it

    def _apply_optics(self, it: QTreeWidgetItem) -> None:
        e = self._entry[it]
        o = e["optics"]
        o["drizzle"] = derive_drizzle(o["focalLength"], o["pixelSize"],
                                      o["aperture"])
        it.setText(COL_FL, str(round(o["focalLength"])))
        it.setText(COL_PX, f"{o['pixelSize']:.2f}")
        it.setText(COL_DRZ, f"{o['drizzle']}x")
        it.setToolTip(COL_DRZ, f"{plate_scale(o['focalLength'], o['pixelSize']):.2f}\"/px")

    def scan(self, auto_align: bool = True) -> None:
        paths = [self.paths.item(i).text()
                 for i in range(self.paths.count())]
        if not paths:
            QMessageBox.information(self, "Lunation",
                                    "add a search path first")
            return
        found = scan_search_paths(paths)
        added = 0
        for s in found["sers"]:
            added += self.add_row(
                s["date"], "SER", s["path"],
                {"type": "ser", "path": s["path"],
                 "label": s["channel"],
                 "key": f"ser:{s['path']}"}) is not None
        for img in found["images"]:
            ext = os.path.splitext(img["path"])[1][1:].upper()
            added += self.add_row(
                img["date"] or "?", ext, img["path"],
                {"type": "image", "image": img,
                 "label": img["name"], "key": f"img:{img['path']}"}) \
                is not None
        # scan assembles the preview itself: aligned draft of the gif,
        # ready to Play (rescans that find nothing new keep the old set)
        if auto_align and added:
            self.build_blink_set(align=True)

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "add captures / images / session configs", "",
            "inputs (*.ser *.json *.xisf *.tif *.tiff *.png *.jpg)")
        from ..master.discovery import DATE_RE

        for f in files:
            f = f.replace("\\", "/")
            base = os.path.basename(f)
            m = DATE_RE.search(f)
            date = m.group(1) if m else ""
            if f.lower().endswith(".ser"):
                self.add_row(date, "SER", f,
                             {"type": "ser", "path": f,
                              "label": ser_channel(f), "key": f"ser:{f}"})
            elif f.lower().endswith(".json"):
                name = base[:-len(".json")]
                if name.startswith(("finish-", "prep-")):
                    continue
                self.add_row(date, "config", f,
                             {"type": "config", "label": name,
                              "session": {
                                  "name": name, "config": f,
                                  "finish_config": find_finish_config(
                                      self.out_root, name, near=f)},
                              "key": f"cfg:{f}"})
            else:
                img = {"path": f, "date": date,
                       "name": os.path.splitext(base)[0]}
                self.add_row(date or "?", "IMG", f,
                             {"type": "image", "image": img,
                              "label": img["name"], "key": f"img:{f}"})

    # ---- table behaviors ------------------------------------------------

    def _on_item_changed(self, it: QTreeWidgetItem, col: int) -> None:
        # group checkbox cascades to checkable children
        e = self._entry.get(it, {})
        if col == COL_NAME and e.get("group"):
            st = it.checkState(COL_NAME)
            for i in range(it.childCount()):
                c = it.child(i)
                if c.flags() & Qt.ItemIsUserCheckable:
                    c.setCheckState(COL_NAME, st)

    def _on_double_click(self, it: QTreeWidgetItem, col: int) -> None:
        e = self._entry.get(it, {})
        if col not in (COL_FL, COL_PX) or e.get("group") or e.get("jobs"):
            return
        field = "focalLength" if col == COL_FL else "pixelSize"
        unit, dec = ("mm", 0) if col == COL_FL else ("µm", 2)
        dlg = OpticsPrompt(self, HEADERS[col], e["optics"][field], unit, dec)
        if dlg.exec() != QDialog.Accepted:
            return
        self.spread_optics(it, field, dlg.num.value(),
                           dlg.scope.currentIndex())

    def spread_optics(self, it: QTreeWidgetItem, field: str, value: float,
                      scope: int) -> None:
        """scope: 0 = this item, 1 = whole date, 2 = all items
        (lunar-master.js:351-371); rows with live jobs are skipped."""
        targets = [it]
        if scope == 1:
            targets = [it.parent().child(i)
                       for i in range(it.parent().childCount())]
        elif scope == 2:
            targets = [g.child(i) for g in self._groups.values()
                       for i in range(g.childCount())]
        for t in targets:
            te = self._entry.get(t, {})
            if "optics" in te and not te.get("jobs"):
                te["optics"][field] = value
                self._apply_optics(t)

    # ---- blink ----------------------------------------------------------

    def build_blink_set(self, align: bool = False) -> None:
        from PySide6.QtWidgets import QApplication

        self.blink_timer.stop()
        self.blink_play_btn.setText("Play")
        groups = []
        for i in range(self.tree.topLevelItemCount()):
            g = self.tree.topLevelItem(i)
            sers = [self._entry[g.child(j)] for j in range(g.childCount())
                    if self._entry.get(g.child(j), {}).get("type") == "ser"]
            e = pick_group_ser(sers)
            if e:
                groups.append((g.text(COL_NAME), e["path"], g))
        if not groups:
            self.preview.setText("no SER groups to blink")
            return

        def prog(step, total, label):
            # feedback while the build blocks the loop: caption in the
            # preview, live fraction on the overall bar
            self.preview.setText(f"blink: {label}…")
            self.overall.setRange(0, max(1, total))
            self.overall.setValue(step)
            self.overall.setFormat(f"%p%  assembling preview — {label}")
            self.preview.repaint()
            QApplication.processEvents()

        frames = build_blink([(k, p) for k, p, _ in groups],
                             progress=prog, align=align)
        # hand the bar back to the scheduler display
        self.overall.setRange(0, 1000)
        self.overall.setFormat("%p%  overall")
        self.overall.setValue(
            round(1000 * overall_fraction(self.scheduler.jobs))
            if self.scheduler else 0)
        self._blink = [
            {"item": g, "key": f["key"],
             "qimage": array_to_qimage(f["image"])}
            for f, (_, _, g) in zip(frames, groups)]
        self._blink_i = 0
        self._show_blink()

    def _show_blink(self, step: int = 0) -> None:
        if not self._blink:
            return
        self._blink_i = (self._blink_i + step) % len(self._blink)
        b = self._blink[self._blink_i]
        pm = QPixmap.fromImage(b["qimage"]).scaled(
            self.preview.width() - 8, self.preview.height() - 8,
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pm)
        self.preview.setToolTip(
            f"{b['key']}  ({self._blink_i + 1}/{len(self._blink)})")
        # the shown frame's group follows in the table
        self._blink_guard = True
        try:
            self.tree.setCurrentItem(b["item"])
        finally:
            self._blink_guard = False

    def _toggle_blink_play(self) -> None:
        if self.blink_timer.isActive():
            self.blink_timer.stop()
            self.blink_play_btn.setText("Play")
        elif self._blink:
            self.blink_timer.start()
            self.blink_play_btn.setText("Stop")

    def _on_select(self, it, _prev) -> None:
        if self._blink_guard:
            return  # blink is driving the selection — keep its frame up
        if self.blink_timer.isActive():
            self._toggle_blink_play()  # manual click stops the playback
        e = self._entry.get(it, {}) if it else {}
        jobs = e.get("jobs") or []
        pick = next((j for j in jobs if j.state == State.RUNNING), None) \
            or next((j for j in jobs if j.log_tail), None) \
            or (jobs[0] if jobs else None)
        self.selected_job = pick
        self._refresh_log()
        self._update_preview(e)

    def _update_preview(self, e: dict) -> None:
        if e.get("group") or e.get("type") not in ("ser", "image"):
            self.preview.setPixmap(QPixmap())
            self.preview.setText("date group — select an item"
                                 if e.get("group") else "select an item")
            return
        path = e["path"] if e["type"] == "ser" else e["image"]["path"]
        name = os.path.basename(path)
        self.preview.setText(f"loading {name}…")
        self.preview.repaint()
        q = preview_qimage(path, e["type"] == "ser",
                           max_px=self.preview.width() * 2)
        if q is None:
            self.preview.setText(f"preview failed: {name}")
            return
        pm = QPixmap.fromImage(q).scaled(
            self.preview.width() - 8, self.preview.height() - 24,
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pm)
        self.preview.setToolTip(name + ("  (frame 0)"
                                        if e["type"] == "ser" else ""))

    def _refresh_log(self) -> None:
        if self.selected_job:
            self.log_box.setPlainText("\n".join(self.selected_job.log_tail))
            sb = self.log_box.verticalScrollBar()
            sb.setValue(sb.maximum())

    # ---- start / run ----------------------------------------------------

    def _checked_leaves(self) -> list[QTreeWidgetItem]:
        out = []
        for g in self._groups.values():
            for i in range(g.childCount()):
                c = g.child(i)
                e = self._entry.get(c, {})
                if e.get("jobs"):
                    continue
                if (c.checkState(COL_NAME) == Qt.Checked
                        or g.checkState(COL_NAME) == Qt.Checked):
                    out.append(c)
        return out

    def _bind(self, it: QTreeWidgetItem, jobs: list) -> None:
        e = self._entry.setdefault(it, {})
        e["jobs"] = (e.get("jobs") or []) + jobs
        it.setFlags(it.flags() & ~Qt.ItemIsUserCheckable)
        self.bound.append((it, e["jobs"]))

    def start(self) -> None:
        out_root = self.out_root
        picked = self._checked_leaves()
        if not picked and not self.lun_check.isChecked():
            QMessageBox.information(self, "Lunation",
                                    "nothing checked to run")
            return
        self._save_settings()
        self._opened_output = False
        # this run's config home, stored with the output
        run_dir = new_run_dir(out_root)
        all_jobs, finish_ids = [], []

        # SER leaves regroup by date into one session each
        ser_by_date: dict[str, list[QTreeWidgetItem]] = {}
        for it in picked:
            e = self._entry[it]
            if e["type"] == "ser":
                date = it.parent().text(COL_NAME)
                ser_by_date.setdefault(date, []).append(it)
        for date, items in sorted(ser_by_date.items()):
            ses = session_from_sers(
                date, [self._entry[i]["path"]
                       for i in items])
            drz = max((self._entry[i]["optics"]["drizzle"]
                       for i in items), default=2)
            cfg = write_discovered_config(ses, run_dir, out_root, drz)
            jobs = session_jobs(date, cfg,
                                find_finish_config(out_root, date))
            by_ser = {self._entry[i]["path"]: i
                      for i in items}
            for j in jobs:
                if j.kind == "stack" and j.meta.get("ser") in by_ser:
                    self._bind(by_ser[j.meta["ser"]], [j])
                elif j.kind == "finish":
                    self._bind_group(date, [j])
                    finish_ids.append(j.id)
            all_jobs += jobs

        for it in picked:
            e = self._entry[it]
            if e["type"] == "config":
                s = e["session"]
                jobs = session_jobs(s["name"], s["config"],
                                    s["finish_config"])
                finish_ids += [j.id for j in jobs if j.kind == "finish"]
                self._bind(it, jobs)
                all_jobs += jobs
            elif e["type"] == "image":
                pj = prep_job(write_prep_config(e["image"], run_dir,
                                                out_root))
                self._bind(it, [pj])
                all_jobs.append(pj)

        if self.lun_check.isChecked():
            # the lunation is an OUTPUT, not an input: no table row — its
            # status rides the overall bar and the files land under
            # <output>/lunation/run-*/
            self._lun_jobs = lunation_jobs(out_root, finish_ids,
                                           self.px_spin.value())
            all_jobs += self._lun_jobs

        if not all_jobs:
            QMessageBox.information(self, "Lunation", "no jobs to run")
            return
        self.scheduler = Scheduler(all_jobs,
                                   heavy_cap=self.jobs_spin.value(),
                                   log=lambda *_: None)
        self.start_btn.setEnabled(False)
        self.timer.start()

    def _bind_group(self, date: str, jobs: list) -> None:
        g = self._groups.get(date or "(undated)")
        if g:
            e = self._entry.setdefault(g, {"group": True})
            e["jobs"] = (e.get("jobs") or []) + jobs
            self.bound.append((g, e["jobs"]))

    # ---- tick -----------------------------------------------------------

    def _tick(self) -> None:
        if not self.scheduler:
            return
        self.scheduler.tick()
        now = time.time()
        for it, jobs in self.bound:
            r = rollup(jobs, now)
            it.setText(COL_PROGRESS, f"{text_bar(r.fraction)}  {r.label}")
            it.setToolTip(COL_PROGRESS, r.tooltip)
            color = STATE_COLORS.get(r.state, "#AAAAAA")
            it.setForeground(COL_PROGRESS, QBrush(QColor(color)))
            it.setText(COL_TIME, fmt_elapsed(r.elapsed_s))
        self.overall.setValue(
            round(1000 * overall_fraction(self.scheduler.jobs)))
        if self._lun_jobs:
            lr = rollup(self._lun_jobs, now)
            self.overall.setFormat(f"%p%  overall — lunation: {lr.label}")
        self._refresh_log()
        if self.scheduler.done():
            self.timer.stop()
            self.overall.setValue(1000)
            # show the results: open the output dir once, unless the whole
            # run was cancelled/failed with nothing produced
            if not self._opened_output and any(
                    j.state == State.OK for j in self.scheduler.jobs):
                self._opened_output = True
                QDesktopServices.openUrl(
                    QUrl.fromLocalFile(self.out_root))

    def _toggle_pause(self) -> None:
        if self.scheduler:
            self.scheduler.paused = not self.scheduler.paused
            self.pause_btn.setText(
                "Resume" if self.scheduler.paused else "Pause")

    def _cancel_entry(self) -> None:
        it = self.tree.currentItem()
        if not (it and self.scheduler):
            return
        e = self._entry.get(it, {})
        for j in e.get("jobs") or []:
            self.scheduler.cancel(j)

    # ---- close ----------------------------------------------------------

    def closeEvent(self, ev) -> None:
        live = self.scheduler and any(
            j.state in (State.RUNNING, State.LAUNCHING)
            for j in self.scheduler.jobs)
        if live:
            a = QMessageBox.warning(
                self, "Lunation", "Jobs are still running. Cancel them"
                " and close?", QMessageBox.Yes | QMessageBox.No)
            if a != QMessageBox.Yes:
                ev.ignore()
                return
            self.scheduler.cancel_all()
        self.timer.stop()
        self._save_settings()
        ev.accept()
