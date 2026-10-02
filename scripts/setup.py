from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from project_setup import inspect_model_addons, npm_command, project_root, venv_python


def run(command: list[str], *, cwd=None) -> None:
    print("\n>", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовить PostTech Radar к первому запуску")
    parser.add_argument(
        "--without-qwen",
        action="store_true",
        help="не скачивать Qwen (основной MiniLM-режим будет работать, перепроверка Qwen — нет)",
    )
    parser.add_argument("--skip-data", action="store_true", help="не пересобирать локальную базу")
    args = parser.parse_args()

    root = project_root()
    issues = inspect_model_addons(root)
    if issues:
        joined = "\n- ".join(issues)
        raise SystemExit(
            "Не найдены дообученные файлы проекта:\n- "
            + joined
            + "\nЕсли проект получен через git, выполните: git lfs pull"
        )

    python = venv_python(root)
    if not python.exists():
        run([sys.executable, "-m", "venv", str(root / ".venv")])
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(python), "-m", "pip", "install", "-r", str(root / "requirements-runtime.txt")])

    if not args.without_qwen:
        run([str(python), str(root / "scripts" / "download_models.py")])

    npm = npm_command()
    if shutil.which(npm) is None:
        raise SystemExit("Не найден npm. Установите Node.js 20 или новее и повторите команду.")
    run([npm, "ci"], cwd=root / "frontend")
    run([npm, "run", "build"], cwd=root / "frontend")

    if not args.skip_data:
        run(
            [
                str(python),
                str(root / "scripts" / "prepare.py"),
                str(root / "data" / "raw" / "Обращения_1931.xlsx"),
                "--mode",
                "replace",
            ]
        )

    print("\nПодготовка завершена.")
    print("Запуск:")
    print(f"  {sys.executable} scripts/run.py")
    print("После запуска откройте http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
