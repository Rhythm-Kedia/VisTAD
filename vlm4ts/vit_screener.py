import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


class ViT4TSScreener:
    """Stage 1 of VLM4TS: Lightweight Vision Transformer coarse-screening filter.
    
    Extracts deep visual features from 2D time-series plots to rapidly score candidate
    anomaly windows without invoking the full multimodal LLM on every window.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", pretrained=True):
        self.device = device
        # Use pre-trained Vision Transformer backbone or initialized feature extractor
        if pretrained:
            try:
                weights = models.ViT_B_16_Weights.DEFAULT
                vit = models.vit_b_16(weights=weights)
            except Exception:
                weights = None
                vit = models.vit_b_16(weights=None)
        else:
            weights = None
            vit = models.vit_b_16(weights=None)
            
        vit.heads = nn.Identity()  # Remove 1000-class classification head, keep 768-dim embedding
        self.model = vit.to(device).eval()

        mean = weights.transforms().mean if weights else [0.485, 0.456, 0.406]
        std = weights.transforms().std if weights else [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.baseline_memory = []

    def extract_visual_embedding(self, pil_image: Image.Image) -> torch.Tensor:
        """Extracts 768-dimensional normalized visual embedding from a time-series plot."""
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = nn.functional.normalize(feat, p=2, dim=-1)
        return feat.squeeze(0)

    def calibrate_nominal_baseline(self, nominal_images: list[Image.Image]):
        """Calibrates nominal visual prototype embeddings from initial baseline operational windows."""
        embs = [self.extract_visual_embedding(img) for img in nominal_images]
        if embs:
            stacked = torch.stack(embs, dim=0)
            self.nominal_prototype = torch.mean(stacked, dim=0, keepdim=True)
            self.nominal_prototype = nn.functional.normalize(self.nominal_prototype, p=2, dim=-1)
        else:
            self.nominal_prototype = None

    def score_candidate(self, pil_image: Image.Image) -> float:
        """Computes visual anomaly score based on cosine distance from nominal prototype."""
        emb = self.extract_visual_embedding(pil_image)
        if self.nominal_prototype is not None:
            similarity = torch.sum(emb * self.nominal_prototype).item()
            score = max(0.0, 1.0 - similarity)
            return float(score)
        return 0.5

    def screen_batch(self, image_list: list[Image.Image]) -> list[float]:
        """Rapidly scores a batch of rendered windows."""
        tensors = torch.stack([self.transform(img) for img in image_list], dim=0).to(self.device)
        with torch.no_grad():
            feats = self.model(tensors)
            feats = nn.functional.normalize(feats, p=2, dim=-1)
            if self.nominal_prototype is not None:
                sims = torch.sum(feats * self.nominal_prototype, dim=-1)
                scores = (1.0 - sims).clamp(min=0.0).cpu().tolist()
                return scores
            return [0.5] * len(image_list)
