import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from vortex_radioss.animtod3plot.Anim_to_D3plot import readAndConvert

file_stem = sys.argv[1]
print("file_stem =", file_stem)
try:
    a2d = readAndConvert(file_stem, use_shell_mask=True, silent=False,
                         no_warnings=False, show_rigidwall=True)
    print("RESULT OK")
except Exception:  # noqa: BLE001
    traceback.print_exc()
    print("RESULT FAILED")
    sys.exit(2)
