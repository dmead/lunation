"""Discovery port: scan rules, channel inference, session assembly,
config generation (never clobber), prep configs, phase names."""

import json

import numpy as np

from lunation.master.discovery import (default_optics, derive_drizzle,
                                       phase_name, plate_scale,
                                       scan_search_paths, ser_channel,
                                       session_from_sers,
                                       write_discovered_config,
                                       write_prep_config)

from .test_ser import build_ser


def _ser(path, color_id=0):
    fr = np.zeros((8, 8), dtype="u1")
    build_ser(str(path), [fr], color_id=color_id, depth=8)
    return str(path)


def test_channel_from_filename_and_header(tmp_path):
    assert ser_channel(_ser(tmp_path / "2026-01-01-Capture_L.ser")) == "L"
    assert ser_channel(_ser(tmp_path / "x_r.ser")) == "R"  # case-insensitive
    # letterless: colorId >= 8 -> OSC, else MONO
    assert ser_channel(_ser(tmp_path / "osc.ser", color_id=8)) == "OSC"
    assert ser_channel(_ser(tmp_path / "mono.ser", color_id=0)) == "MONO"


def test_scan_skip_rules_and_image_filters(tmp_path):
    (tmp_path / "caps" / "2026-01-01").mkdir(parents=True)
    (tmp_path / "caps" / ".git").mkdir()
    (tmp_path / "caps" / "#work").mkdir()
    (tmp_path / "caps" / "moon_files").mkdir()
    _ser(tmp_path / "caps" / "2026-01-01" / "a_L.ser")
    _ser(tmp_path / "caps" / ".git" / "hidden_L.ser")
    _ser(tmp_path / "caps" / "#work" / "tmp_L.ser")
    _ser(tmp_path / "caps" / "moon_files" / "tile_L.ser")

    big = b"x" * (301 * 1024)
    d = tmp_path / "caps" / "2026-01-01"
    (d / "final_2026-01-01.png").write_bytes(big)       # kept
    (d / "small.png").write_bytes(b"x" * 1024)          # < 300 KB
    (d / "003_reject_frame.png").write_bytes(big)       # thumbnail
    (d / "capture_L.ser.png").write_bytes(big)          # ser preview
    (d / "frame_04_x.png").write_bytes(big)             # render byproduct
    (d / "moon.preS.xisf").write_bytes(big)             # backup

    found = scan_search_paths([str(tmp_path / "caps")])
    assert [s["channel"] for s in found["sers"]] == ["L"]
    assert found["sers"][0]["date"] == "2026-01-01"
    assert [i["name"] for i in found["images"]] == ["final_2026-01-01"]
    assert found["images"][0]["date"] == "2026-01-01"


def test_session_job_ids():
    ses = session_from_sers("2026-01-01", [
        "a_L.ser", "b_L.ser", "c_R.ser", "d_G.ser", "e_B.ser", "f_H.ser"])
    ids = [j["id"] for j in ses["jobs"]]
    assert ids == ["L1", "L2", "R", "G", "B", "H"]
    by_id = {j["id"]: j for j in ses["jobs"]}
    assert by_id["L1"]["localAlign"] and by_id["H"]["localAlign"]
    assert by_id["R"]["bestFraction"] == 0.35
    assert "localAlign" not in by_id["R"]


def test_session_osc_mono_suffixes(tmp_path):
    osc = _ser(tmp_path / "cap1.ser", color_id=9)
    osc2 = _ser(tmp_path / "cap2.ser", color_id=10)
    mono = _ser(tmp_path / "plain.ser", color_id=0)
    ses = session_from_sers("d", [osc, osc2, mono])
    assert [j["id"] for j in ses["jobs"]] == ["OSC1", "OSC2", "M1"]
    assert ses["jobs"][0]["channel"] == "osc"


def test_config_generation_never_clobbers(tmp_path):
    ses = session_from_sers("2026-01-01", ["a_L.ser"])
    p = write_discovered_config(ses, str(tmp_path), f"{tmp_path}/out",
                                drizzle=3)
    cfg = json.load(open(p))
    assert cfg["defaults"] == {
        "bestFraction": 0.10, "maxFrames": 400, "minFrames": 20,
        "alignOnGradient": True, "drizzle": 3, "drizzleMargin": 16}
    assert cfg["concurrency"] == 2
    assert cfg["outDir"].endswith("/2026-01-01")

    # hand-tuned config wins: second write is a no-op
    json.dump({"hand": "tuned"}, open(p, "w"))
    p2 = write_discovered_config(ses, str(tmp_path), f"{tmp_path}/out")
    assert p2 == p and json.load(open(p)) == {"hand": "tuned"}


def test_prep_config_shape(tmp_path):
    img = {"path": "Z:/finished/nice moon (v2).tif", "date": "2026-02-02",
           "name": "nice moon (v2)"}
    prep = write_prep_config(img, str(tmp_path), f"{tmp_path}/out")
    cfg = json.load(open(prep["config"]))
    assert cfg["targetR"] == 979 and cfg["canvas"] == 2300
    assert cfg["items"] == [{"src": img["path"], "out": prep["out"]}]
    assert prep["name"].startswith("FIN_2026-02-02_")
    assert " " not in prep["name"] and "(" not in prep["name"]


def test_optics_derivation(tmp_path):
    # house rig: TS-70 440mm f/6.3, IMX585 2.9um -> Q~1.7 -> 2x
    assert derive_drizzle(440, 2.9, 70) == 2
    # heavily undersampled (short focal, big pixels) -> clamped at 3
    assert derive_drizzle(200, 6.0, 70) == 3
    # oversampled (long focal) needs no drizzle
    assert derive_drizzle(2000, 2.9, 70) == 1
    assert abs(plate_scale(440, 3.76) - 1.7626) < 1e-3
    o = default_optics(str(tmp_path))  # no equipment.json -> defaults
    assert o["focalLength"] == 440 and o["drizzle"] >= 1

    (tmp_path / "equipment.json").write_text(
        '{"focalLength": 1000, "pixelSize": 2.9}')
    assert default_optics(str(tmp_path))["focalLength"] == 1000


def test_phase_names():
    assert phase_name("not-a-date") == ""
    # exercise the bins over one synodic month; all names must appear
    import datetime

    names = set()
    d0 = datetime.date(2026, 1, 1)
    for k in range(30):
        names.add(phase_name((d0 + datetime.timedelta(days=k)).isoformat()))
    assert {"new", "waxing crescent", "first quarter", "waxing gibbous",
            "full", "waning gibbous", "last quarter",
            "waning crescent"} <= names
