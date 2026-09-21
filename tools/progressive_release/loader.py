"""Load the unfused, MS1MV3-calibrated progressive degree-2 model."""
from pathlib import Path
from controlled_degree2.model import load_controlled_checkpoint


def load_model(checkpoint=None, device='cpu'):
    if checkpoint is None:
        checkpoint = Path(__file__).resolve().parent / 'checkpoints/student_scaled.pt'
    model, metadata = load_controlled_checkpoint(str(checkpoint), device=device)
    return model.eval(), metadata
