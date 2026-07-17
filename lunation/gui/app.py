"""GUI entry point — `lunation gui`."""

import os
import sys

# icon source: out/2026-05-30/final/moon_2026-05-30.xisf (full moon,
# 14.2 d), disk-cropped with a feathered circular alpha
ICON = os.path.join(os.path.dirname(__file__), "icon.ico")


def run(output: str | None = None) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is not installed — install the GUI extra:\n"
              "  uv tool install 'lunation[gui]'", file=sys.stderr)
        return 2
    from PySide6.QtGui import QIcon

    from .window import MasterWindow

    if sys.platform == "win32":
        # own AppUserModelID, else the taskbar groups us under python.exe
        # and shows ITS icon instead of ours
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "lunation.gui")
    app = QApplication.instance() or QApplication(sys.argv)
    if os.path.exists(ICON):
        app.setWindowIcon(QIcon(ICON))
    win = MasterWindow(output)
    win.show()
    return app.exec()
