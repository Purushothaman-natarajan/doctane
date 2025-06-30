import os
from pathlib import Path
from setuptools import setup

PKG_NAME = "doctane"
VERSION = os.getenv("BUILD_VERSION", "0.1.0a0")  # Use CI/CD versioning or default

if __name__ == "__main__":
    print(f"Building wheel {PKG_NAME}-{VERSION}")

    # Dynamically create doctane/version.py
    cwd = Path(__file__).parent.absolute()
    version_file = cwd / PKG_NAME / "version.py"
    version_file.parent.mkdir(parents=True, exist_ok=True)

    with open(version_file, "w", encoding="utf-8") as f:
        f.write(f"__version__ = '{VERSION}'\n")
    
    setup(name=PKG_NAME, version=VERSION)