from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> None:
    cache = Path.home() / "Projects" / "Radio2026" / "homebrew-cache"
    cache.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["HOMEBREW_CACHE"] = str(cache)
    environment["HOMEBREW_NO_AUTO_UPDATE"] = "1"
    subprocess.run(["brew", "install", "llvm@20"], check=True, env=environment)
    library = Path("/opt/homebrew/opt/llvm@20/lib/libLLVM.dylib")
    if not library.exists():
        raise RuntimeError(f"LLVM installation completed without {library}")
    print(library)


if __name__ == "__main__":
    main()
