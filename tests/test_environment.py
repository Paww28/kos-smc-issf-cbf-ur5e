import os
import re
import subprocess
import sys

import jax
import jax.numpy as jnp

from conftest import REPO


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins(*files):
    pins = {}
    for fn in files:
        with open(os.path.join(REPO, fn)) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                name, version = line.split("==")
                pins[_norm(name)] = version
    return pins


def test_pip_freeze_matches_requirements():
    """Installed packages == requirements.txt (runtime + testing), exactly (kossmc_issf itself excluded)."""
    env = dict(os.environ, PYTHONPATH="")
    out = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
        capture_output=True, text=True, check=True, env=env,
    ).stdout
    installed = {}
    for line in out.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, version = line.split("==")
        installed[_norm(name)] = version
    expected = _pins("requirements.txt")
    assert installed == expected, {
        "missing": sorted(set(expected) - set(installed)),
        "extra": sorted(set(installed) - set(expected)),
        "version": sorted(k for k in set(expected) & set(installed) if expected[k] != installed[k]),
    }


def test_jax_x64_enabled():
    assert jax.config.jax_enable_x64
    assert jnp.zeros(1).dtype == jnp.float64


def test_x64_enabled_by_any_kossmc_issf_submodule():
    """Importing a submodule alone (fresh interpreter) must already enable x64 via kossmc_issf/__init__."""
    code = (
        "import kossmc_issf.core.safety, jax, jax.numpy as jnp;"
        "assert jax.config.jax_enable_x64;"
        "assert jnp.zeros(1).dtype == jnp.float64"
    )
    env = dict(os.environ, PYTHONPATH="")
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=REPO)
