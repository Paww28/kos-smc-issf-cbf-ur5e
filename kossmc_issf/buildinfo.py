"""kossmc-issf version and git commit of the running code (written to experiment_info.txt)."""

import os
import subprocess
from importlib.metadata import PackageNotFoundError, version

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def kossmc_issf_version():
    try:
        return version("kossmc-issf")
    except PackageNotFoundError:
        return "unknown"


def git_commit():
    """Short hash of HEAD, with "-dirty" if tracked files have uncommitted changes.

    "unknown" when the package is not inside a git checkout (or git is not available).
    """
    def git(*args):
        return subprocess.run(["git", "-C", _PKG_DIR, *args], capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")
