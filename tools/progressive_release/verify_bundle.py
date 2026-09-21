"""Check file hashes and smoke-test strict loading/inference without datasets."""
import hashlib
import json
from pathlib import Path

import torch
from loader import load_model
from controlled_degree2.model import DirectQuadratic
from controlled_degree2.rescale_residual_graph import BOUNDARIES, capture


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        assert digest(root / name) == expected, name
    torch.set_num_threads(4)
    scaled, metadata = load_model()
    assert sum(isinstance(m, DirectQuadratic) for m in scaled.modules()) == 25
    assert not any(isinstance(m, torch.nn.PReLU) for m in scaled.modules())
    inputs = torch.zeros(1, 3, 112, 112)
    result, outputs = capture(scaled, inputs, BOUNDARIES)
    assert result.shape == (1, 512) and bool(torch.isfinite(result).all())
    for name in BOUNDARIES:
        assert bool(torch.isfinite(outputs[name]).all()), name
    import controlled_degree2.model as implementation
    assert Path(implementation.__file__).resolve().is_relative_to(root)
    print(json.dumps(dict(verified_files=len(manifest['files']), strict_loading=True,
                         synthetic_smoke_test_passed=True, embedding_shape=list(result.shape),
                         model_source=str(Path(implementation.__file__).resolve())), indent=2))


if __name__ == '__main__':
    main()
