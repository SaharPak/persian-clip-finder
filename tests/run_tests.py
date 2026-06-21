"""Minimal test runner used when pytest is not installed.

Usage::

    python tests/run_tests.py

It discovers test files under ``tests/`` (excluding this file) and executes
each module's top-level test functions in the order they are defined. It
provides a small subset of pytest fixtures:

* ``tmp_path`` – a fresh :class:`pathlib.Path` under a temporary directory
* ``capsys``   – an object whose ``.readouterr()`` returns ``(out, err)``

Failures are reported with a non-zero exit code so this can be used from CI
without requiring ``pytest`` as a dependency.
"""

from __future__ import annotations

import importlib
import inspect
import io
import shutil
import sys
import tempfile
import traceback
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# A pytest-like result for capsys.readouterr()
Capture = namedtuple("Capture", "out err")


class _ActiveCapture:
    """A ``capsys``-like object that captures stdout/stderr for the test body."""

    def __init__(self) -> None:
        self._old_out = sys.stdout
        self._old_err = sys.stderr
        self._out = io.StringIO()
        self._err = io.StringIO()
        sys.stdout = self._out
        sys.stderr = self._err

    def readouterr(self) -> Capture:
        # Flush & snapshot, but keep capturing for the rest of the test.
        out = self._out.getvalue()
        err = self._err.getvalue()
        self._out = io.StringIO()
        self._err = io.StringIO()
        sys.stdout = self._out
        sys.stderr = self._err
        return Capture(out, err)

    def close(self) -> None:
        sys.stdout = self._old_out
        sys.stderr = self._old_err


def _discover() -> list[str]:
    return [
        "tests." + p.stem
        for p in sorted(Path(__file__).resolve().parent.glob("test_*.py"))
    ]


def _provide_args(fn: Callable) -> tuple[list[Any], dict[str, Any]]:
    sig = inspect.signature(fn)
    args: list[Any] = []
    capsys_obj: _ActiveCapture | None = None
    tmp_path_obj: Path | None = None

    for name in sig.parameters:
        if name == "tmp_path" and tmp_path_obj is None:
            tmp_path_obj = Path(tempfile.mkdtemp(prefix="pcf-test-"))
        elif name == "capsys" and capsys_obj is None:
            capsys_obj = _ActiveCapture()

    for name in sig.parameters:
        if name == "tmp_path":
            assert tmp_path_obj is not None
            args.append(tmp_path_obj)
        elif name == "capsys":
            assert capsys_obj is not None
            args.append(capsys_obj)

    cleanup = {
        "capsys": capsys_obj,
        "tmp_path": tmp_path_obj,
    }
    return args, cleanup


def _run_one(mod_name: str) -> tuple[int, int]:
    mod = importlib.import_module(mod_name)
    passed = 0
    failed = 0
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        if getattr(fn, "__module__", None) != mod_name:
            continue

        sig = inspect.signature(fn)
        unsupported = [
            p.name
            for p in sig.parameters.values()
            if p.name not in ("tmp_path", "capsys")
            and p.default is inspect.Parameter.empty
        ]
        if unsupported:
            print(f"  SKIP {name} (unsupported fixture: {unsupported})")
            continue

        args, cleanup = _provide_args(fn)
        try:
            fn(*args)
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"  PASS {name}")
        finally:
            if cleanup.get("capsys") is not None:
                cleanup["capsys"].close()
            if cleanup.get("tmp_path") is not None:
                shutil.rmtree(cleanup["tmp_path"], ignore_errors=True)
    return passed, failed


def main() -> int:
    total_p = 0
    total_f = 0
    for m in _discover():
        if m == "tests.run_tests":
            continue
        print(f"== {m} ==")
        p, f = _run_one(m)
        total_p += p
        total_f += f
    print(f"\n{total_p} passed, {total_f} failed")
    return 0 if total_f == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
