from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path


def main() -> None:
    cache = Path.home() / "Projects" / "Radio2026" / "homebrew-cache"
    cache.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["HOMEBREW_CACHE"] = str(cache)
    environment["HOMEBREW_NO_AUTO_UPDATE"] = "1"
    try:
        subprocess.run(["brew", "install", "llvm@20"], check=True, env=environment)
    except subprocess.CalledProcessError:
        bottles = sorted((cache / "downloads").glob("*llvm@20*.tar.gz"))
        if not bottles:
            raise
        destination = Path.home() / "Projects" / "Radio2026" / "runtime" / "llvm20-bottle"
        staging = destination.with_name(destination.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        with tarfile.open(bottles[-1]) as archive:
            archive.extractall(staging, filter="data")
        shutil.rmtree(destination, ignore_errors=True)
        staging.rename(destination)
    stable = Path("/opt/homebrew/opt/llvm@20/lib/libLLVM.dylib")
    unpacked = sorted(
        (Path.home() / "Projects" / "Radio2026" / "runtime" / "llvm20-bottle").glob(
            "**/libLLVM.dylib"
        )
    )
    library = stable if stable.exists() else (unpacked[-1] if unpacked else stable)
    if not library.exists():
        raise RuntimeError(f"LLVM installation completed without {library}")
    print(library)


if __name__ == "__main__":
    main()
