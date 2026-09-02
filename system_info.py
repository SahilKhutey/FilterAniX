"""System and Hardware Diagnostics CLI."""
import sys
from pathlib import Path

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.hardware import get_hardware_report


def main():
    print("==================================================")
    print("       ANIMATED CREATOR: SYSTEM DIAGNOSTICS       ")
    print("==================================================")
    report = get_hardware_report()
    for key, val in report.to_dict().items():
        status_flag = ""
        if key in ["FFmpeg Available", "CUDA Available"] and val is True:
            status_flag = " (ENABLED)"
        elif key in ["FFmpeg Available"] and val is False:
            status_flag = " (REQUIRED)"
        print(f"[*] {key:<20}: {val}{status_flag}")
    print("==================================================")


if __name__ == "__main__":
    main()
