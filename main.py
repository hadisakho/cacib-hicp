# check- environnement

import sys
from importlib import import_module
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent

    print(f"Projet : {project_root.name}")
    print(f"Python : {sys.version.split()[0]}")
    print(f"Interpréteur : {sys.executable}")
    print()

    packages = ("numpy", "pandas", "pyarrow", "httpx")

    for package in packages:
        import_module(package)
        print(f"{package} : {version(package)}")

    print("\nEnvironnement bon")


if __name__ == "__main__":
    main()
