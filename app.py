"""FilterAniX Studio v2.0 Launcher."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ui.app import create_app
from src.ui.theme import get_studio_theme, STUDIO_CUSTOM_CSS


def main():
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=get_studio_theme(),
        css=STUDIO_CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
