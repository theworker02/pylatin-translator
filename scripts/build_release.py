from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
RELEASE_DIR = ROOT / "release-artifacts"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    RELEASE_DIR.mkdir(exist_ok=True)

    run([sys.executable, "-m", "flit", "build"])

    run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--clean",
            "--noconfirm",
            "--onefile",
            "--name",
            "universal-translator",
            str(ROOT / "universal_translator" / "__main__.py"),
        ]
    )

    for artifact in DIST_DIR.glob("*"):
        shutil.copy2(artifact, RELEASE_DIR / artifact.name)

    standalone_dir = ROOT / "dist"
    for artifact_name in ["universal-translator.exe", "universal-translator"]:
        artifact_path = standalone_dir / artifact_name
        if artifact_path.exists():
            shutil.copy2(artifact_path, RELEASE_DIR / artifact_path.name)

    print(f"Release artifacts copied to: {RELEASE_DIR}")


if __name__ == "__main__":
    main()
