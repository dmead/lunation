"""Frozen-app entry (PyInstaller bundles): double-click opens the GUI,
any argument makes it the CLI (which is how the scheduler's job children
run inside a bundle — the exe re-invokes itself with stage commands).

`--smoke` builds the main window offscreen and exits 0 — CI's proof that
the bundle's Qt/numpy/cv2 stack actually loads on the target OS.
"""

import multiprocessing
import os
import sys


def main() -> int:
    # frame workers spawn grandchildren of this exe; without this every
    # worker would relaunch the GUI
    multiprocessing.freeze_support()
    argv = sys.argv[1:]
    if argv == ["--smoke"]:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        from lunation.gui.window import MasterWindow

        app = QApplication([])
        w = MasterWindow()
        w.show()
        return 0
    if argv:
        from lunation.cli import main as cli_main

        sys.argv = ["lunation", *argv]
        try:
            cli_main()
        except SystemExit:
            raise
        except BaseException:  # noqa: BLE001 — frozen CLI boundary
            # a windowed bundle turns unhandled exceptions into a MODAL
            # dialog; scheduler children run headless and would block on
            # it forever. Print and die instead.
            import traceback

            traceback.print_exc(file=sys.stderr)
            os._exit(1)
        return 0
    from lunation.gui.app import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
