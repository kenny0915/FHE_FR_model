import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_unhandled_exception(tmp_path, rank, source):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_rank_zero_uncaught_exception_is_written_to_training_log(tmp_path):
    result = _run_unhandled_exception(
        tmp_path,
        0,
        (
            "from utils.utils_logging import init_logging; "
            f"init_logging(0, {str(tmp_path)!r}); "
            "raise FloatingPointError('non-finite embeddings')"
        ),
    )

    assert result.returncode != 0
    log_text = (tmp_path / "training.log").read_text(encoding="utf-8")
    assert "Uncaught exception on rank 0 (main process)" in log_text
    assert "Traceback (most recent call last):" in log_text
    assert "FloatingPointError: non-finite embeddings" in log_text


def test_nonzero_rank_uses_separate_error_log(tmp_path):
    result = _run_unhandled_exception(
        tmp_path,
        2,
        (
            "from utils.utils_logging import init_logging; "
            f"init_logging(2, {str(tmp_path)!r}); "
            "raise RuntimeError('rank two failed')"
        ),
    )

    assert result.returncode != 0
    assert not (tmp_path / "training.log").exists()
    log_text = (tmp_path / "error_rank_2.log").read_text(encoding="utf-8")
    assert "Uncaught exception on rank 2 (main process)" in log_text
    assert "RuntimeError: rank two failed" in log_text


def test_background_thread_exception_is_persisted(tmp_path):
    result = _run_unhandled_exception(
        tmp_path,
        0,
        (
            "import threading; "
            "from utils.utils_logging import init_logging; "
            f"init_logging(0, {str(tmp_path)!r}); "
            "thread = threading.Thread("
            "target=lambda: (_ for _ in ()).throw(ValueError('worker failed')), "
            "name='prefetch-worker'); "
            "thread.start(); thread.join()"
        ),
    )

    assert result.returncode == 0
    log_text = (tmp_path / "training.log").read_text(encoding="utf-8")
    assert "Uncaught exception on rank 0 (thread=prefetch-worker)" in log_text
    assert "ValueError: worker failed" in log_text
