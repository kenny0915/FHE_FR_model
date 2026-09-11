from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from loader import load_model


device = "cuda" if torch.cuda.is_available() else "cpu"
model, metadata = load_model("model.pt", map_location=device)

# Input must be an already aligned 112x112 RGB face.
image = Image.open(Path("aligned_face.jpg")).convert("RGB").resize((112, 112))
inputs = pil_to_tensor(image).float().unsqueeze(0).to(device)
inputs = inputs.div(255.0).sub(0.5).div(0.5)

with torch.no_grad():
    embedding = torch.nn.functional.normalize(model(inputs).float(), dim=1)

print(metadata["format"], embedding.shape, torch.isfinite(embedding).all().item())

