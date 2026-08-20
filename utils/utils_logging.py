import logging
import os
import sys
import threading
import traceback
from datetime import datetime


def _append_uncaught_exception(log_path, rank, exc_type, exc_value,
                               exc_traceback, context):
    """Best-effort persistence for exceptions escaping a process or thread."""
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(
                f"Training: {timestamp}-Uncaught exception on rank {rank}"
                f" ({context})\n"
            )
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=log_file)
            log_file.flush()
            os.fsync(log_file.fileno())
    except Exception:
        # Exception reporting must never replace or mask the original failure.
        return False
    return True


def install_uncaught_exception_logging(rank, models_root):
    """Persist uncaught process and thread exceptions before Python exits.

    Rank zero appends to the normal training log. Other distributed ranks use
    separate error-only files so simultaneous failures cannot interleave four
    tracebacks in ``training.log``.
    """
    rank = int(rank)
    log_path = os.path.join(
        models_root,
        "training.log" if rank == 0 else f"error_rank_{rank}.log",
    )

    previous_sys_hook = sys.excepthook
    if getattr(previous_sys_hook, "_training_exception_hook", False):
        previous_sys_hook = getattr(
            previous_sys_hook, "_previous_hook", sys.__excepthook__)

    def process_exception_hook(exc_type, exc_value, exc_traceback):
        _append_uncaught_exception(
            log_path,
            rank,
            exc_type,
            exc_value,
            exc_traceback,
            context="main process",
        )
        previous_sys_hook(exc_type, exc_value, exc_traceback)

    process_exception_hook._training_exception_hook = True
    process_exception_hook._previous_hook = previous_sys_hook
    sys.excepthook = process_exception_hook

    previous_thread_hook = threading.excepthook
    if getattr(previous_thread_hook, "_training_exception_hook", False):
        previous_thread_hook = getattr(
            previous_thread_hook,
            "_previous_hook",
            threading.__excepthook__,
        )

    def thread_exception_hook(args):
        thread_name = args.thread.name if args.thread is not None else "unknown"
        _append_uncaught_exception(
            log_path,
            rank,
            args.exc_type,
            args.exc_value,
            args.exc_traceback,
            context=f"thread={thread_name}",
        )
        previous_thread_hook(args)

    thread_exception_hook._training_exception_hook = True
    thread_exception_hook._previous_hook = previous_thread_hook
    threading.excepthook = thread_exception_hook


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_logging(rank, models_root):
    install_uncaught_exception_logging(rank, models_root)
    if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
        handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)
        log_root.info('rank_id: %d' % rank)
