from pathlib import Path
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# IMPORT THE SINGLE SOURCE OF TRUTH
# ============================================================

from src.ml.predict_video import (
    predict_video,
)


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print("\n")
        print("=" * 70)
        print("                  VIDEO INFERENCE")
        print("=" * 70)

        print(
            "\nUsage:"
        )

        print(
            "  python -m src.inference.predict_video "
            "<video_path>"
        )

        print(
            "\nExample:"
        )

        print(
            r'  python -m src.inference.predict_video "D:\Amrita SLR Dataset\sign project\50.power\Power 3(1).mp4"'
        )

        print(
            "\n" + "=" * 70
        )

        return

    video_path = Path(
        sys.argv[1]
    )

    if not video_path.exists():

        print(
            f"\nERROR: Video does not exist:\n"
            f"{video_path}"
        )

        return

    # --------------------------------------------------------
    # Run EXACT SAME prediction pipeline
    # --------------------------------------------------------

    predict_video(
        video_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()