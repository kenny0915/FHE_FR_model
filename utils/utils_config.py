"""Load legacy flat and family-organized experiment configs independently."""
from copy import deepcopy
import importlib
from pathlib import PurePosixPath


def get_config(config_file):
    path = PurePosixPath(config_file)
    if (not config_file.startswith('configs/') or '..' in path.parts
            or path.suffix not in ('', '.py') or len(path.parts) < 2):
        raise ValueError('config must be a path under configs/ with an optional .py suffix')
    module_path = path.with_suffix('') if path.suffix else path
    module_name = '.'.join(module_path.parts)
    cfg = deepcopy(importlib.import_module('configs.base').config)
    cfg.update(deepcopy(importlib.import_module(module_name).config))
    if cfg.output is None:
        # Keep legacy output paths when switching to a family-organized config path.
        cfg.output = 'work_dirs/' + module_path.name
    return cfg
