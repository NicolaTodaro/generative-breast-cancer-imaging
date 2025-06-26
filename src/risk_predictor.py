import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class RiskPredictor(nn.Module):
    def __init__(self, num_birads=6, attr_dim=1, embed_dim=128):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()
        self.img_proj = nn.Linear(768, embed_dim)
        self.attr_proj = nn.Linear(attr_dim, embed_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_birads)
        )

    def forward(self, image, attributes):
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        img_feat = self.img_proj(self.vit(image))
        attr_feat = self.attr_proj(attributes.float())
        x = torch.cat([img_feat, attr_feat], dim=1)
        logits = self.classifier(x)
        return logits  # [B, num_birads]
