from __future__ import annotations

import argparse
import subprocess

from project_setup import (
    inspect_model_addons,
    merged_runtime_environment,
    project_root,
    venv_python,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустить PostTech Radar")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    root = project_root()
    python = venv_python(root)
    if not python.exists():
        raise SystemExit("Сначала выполните команду установки из README: python scripts/setup.py")
    issues = inspect_model_addons(root)
    if issues:
        raise SystemExit("\n".join(issues) + "\nВыполните git lfs pull и повторите запуск.")
    if not (root / "data" / "processed" / "posttech.db").exists():
        raise SystemExit("Локальная база ещё не собрана. Выполните: python scripts/setup.py")
    if not (root / "frontend" / "dist" / "index.html").exists():
        raise SystemExit("Frontend ещё не собран. Выполните: python scripts/setup.py")

    command = [
        str(python),
        "-m",
        "uvicorn",
        "app.main:app",
        "--app-dir",
        "backend",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print(f"PostTech Radar: http://{args.host}:{args.port}")
    raise SystemExit(subprocess.call(command, cwd=root, env=merged_runtime_environment(root, python)))


if __name__ == "__main__":
    main()
