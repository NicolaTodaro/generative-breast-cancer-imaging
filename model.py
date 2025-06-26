# src/model.py

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class MaskedImageViT(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.vit = vit_b_16(weights="IMAGENET1K_V1")  # Using pre-trained weights
        self.vit.heads = nn.Identity()  # Remove the final classifier head
        self.proj = nn.Linear(768, embed_dim)  # Adapt the output to embed_dim

    def forward(self, x):
        # Repeating the channel 3 times to get RGB, because ViT expects 3 channels and input is grayscale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.vit(x)  # [batch, 768]
        return self.proj(features)  # [batch, embed_dim]

class AttributeEmbedding(nn.Module):
    def __init__(self, embed_dim, num_density=4, num_view=2, num_category=None, num_birads=6):
        super().__init__()
        self.density_emb = nn.Embedding(num_density, embed_dim)
        self.view_emb = nn.Embedding(num_view, embed_dim)
        self.category_emb = nn.Embedding(num_category, embed_dim)
        self.birads_emb = nn.Embedding(num_birads, embed_dim)
        self.side_emb = nn.Embedding(2, embed_dim)
        self.positional_enc = nn.Parameter(torch.randn(5, embed_dim))  
        self.coord_proj = nn.Linear(4, embed_dim)  
        self.final_proj = nn.Linear(6 * embed_dim, embed_dim) 
        
    def forward(self, attributes):
        # attributes: [batch, 9]
        idx = attributes[:, :5].long()      # [density, view, category, birads, side]
        coords = attributes[:, 5:9].float()  # [xmin, ymin, xmax, ymax]
        emb_list = [
            self.density_emb(idx[:, 0]),
            self.view_emb(idx[:, 1]),
            self.category_emb(idx[:, 2]),
            self.birads_emb(idx[:, 3]),
            self.side_emb(idx[:, 4])
        ]  # each one [batch, embed_dim]
        stacked = torch.stack(emb_list, dim=1)  # [batch, 5, embed_dim]
        stacked = stacked + self.positional_enc  # positional encoding
        discrete_emb = stacked.view(stacked.size(0), -1)  # [batch, 5*embed_dim]
        coord_emb = self.coord_proj(coords)  # [batch, embed_dim]
        full_emb = torch.cat([discrete_emb, coord_emb], dim=1)  # [batch, 6*embed_dim]
        return self.final_proj(full_emb)  # [batch, embed_dim]

class AdvancedEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        c, h, w = input_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, 4, 2, 1),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * (h // 16) * (w // 16), latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class AdvancedDecoder(nn.Module):
    def __init__(self, output_shape, latent_dim):
        super().__init__()
        c, h, w = output_shape
        self.fc = nn.Linear(latent_dim, 256 * (h // 16) * (w // 16))
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, h // 16, w // 16)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, c, 4, 2, 1), # 224x224
            #nn.Sigmoid() if you have values between 0 and 1, but we have values between -1 and 1
        )

    def forward(self, z):
        x = self.fc(z)
        out = self.decoder(x)
       # assert out.shape[1:] == (1, 224, 224), f"Decoder output shape: {out.shape}"
        return out  # [B, C, H, W] shape as per output_shape
"""

class AttributeTransformer(nn.Module):
    def __init__(self, attr_dim, embed_dim, num_layers=2, num_heads=2, batch_first=True):
        super().__init__()
        self.embedding = nn.Linear(attr_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=batch_first)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, attr):
        x = self.embedding(attr).unsqueeze(1)  # [batch, 1, embed_dim]
        x = self.transformer(x)
        return x.squeeze(1)  # [batch, embed_dim]

class SimpleUNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, 3, padding=1)
        )
        self.proj = nn.Linear(3 * latent_dim, latent_dim)

    def forward(self, x):
        # x: [B, 3*latent_dim]
        x = x.unsqueeze(1)  # [B, 1, 3*latent_dim]
        d = self.down1(x)
        u = self.up1(d)
        u = u.view(u.size(0), -1)  # [B, 3*latent_dim]
        return self.proj(u)  # [B, latent_dim]


        
class LatentDiffusionModel(nn.Module):
    def __init__(self, config, category_list=None):
        super(LatentDiffusionModel, self).__init__()
        
        self.latent_dim = config.latent_dim
        # Latent Diffusion Model components Encoder, Decoder, Denoiser
        self.encoder = AdvancedEncoder(config.input_shape, config.latent_dim)
        self.decoder = AdvancedDecoder(config.input_shape, config.latent_dim)
        self.denoiser = SimpleUNet(config.latent_dim)
        # Vision Transformer to process masked images
        self.masked_image_vit = MaskedImageViT(embed_dim=config.latent_dim)
        '''Transformer for anatomical attributes
        self.attr_transformer = AttributeTransformer(attr_dim=config.attr_dim, embed_dim=config.latent_dim)'''
        #Attribute embedding for anatomical attributes
        self.attr_embedder = AttributeEmbedding(embed_dim=config.latent_dim, num_density=4, num_view=2, num_category=len(category_list), num_birads=6)

        # Diffusion schedule parameters from config
        beta_start = getattr(config, "beta_start", 1e-4)
        beta_end = getattr(config, "beta_end", 0.02)
        num_steps = getattr(config, "num_diffusion_steps", 1000)
        # Buffer for betas
        betas = torch.linspace(beta_start, beta_end, num_steps)
        self.register_buffer('betas', betas)

    def q_sample(self, latent, t, noise):
        t = t.view(-1).long()
        beta_t = self.betas[t].view(-1, *[1]*(latent.dim()-1))
        return torch.sqrt(1 - beta_t) * latent + torch.sqrt(beta_t) * noise

    def forward(self, images=None, masked_images=None, attributes=None, t=None):
        
        Training: images, masked_images, attributes, t
        Inference: attributes (images/masked_images None)
        
        if images is not None and masked_images is not None:
            latent = self.encoder(images)
            masked_emb = self.masked_image_vit(masked_images)
            attr_emb = self.attr_embedder(attributes)
            cond = torch.cat([masked_emb, attr_emb], dim=1)  # [B, 2*latent_dim]
            
            if t is not None:
                noise = torch.randn_like(latent)
                noisy_latent = self.q_sample(latent, t, noise)
                input_denoiser = torch.cat([noisy_latent, cond], dim=1)  # [B, 3*latent_dim]
            else:
                input_denoiser = torch.cat([latent, cond], dim=1)  # [B, 3*latent_dim]
            
            denoised_latent = self.denoiser(input_denoiser)
            recon = self.decoder(denoised_latent)
            return recon, latent

        else:
            # Inference: only attributes
            batch_size = attributes.shape[0]
            latent = torch.randn(batch_size, self.latent_dim, device=attributes.device)
            masked_emb = torch.zeros(batch_size, self.latent_dim, device=attributes.device)
            attr_emb = self.attr_embedder(attributes)
            cond = torch.cat([masked_emb, attr_emb], dim=1)
            input_denoiser = torch.cat([latent, cond], dim=1)  # [B, 3*latent_dim]
            denoised_latent = self.denoiser(input_denoiser)
            recon = self.decoder(denoised_latent)
            return recon, latent
        
        
if __name__ == "__main__":
    from config_tuning import config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    category_list = [...]  # <-- Inserisci qui la tua lista
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LatentDiffusionModel(config, category_list).to(device)
    dummy_input = torch.randn(4, *config.input_shape).to(device)
    dummy_masked = torch.randn(4, *config.input_shape).to(device)
    dummy_attr = torch.cat([
        torch.randint(0, 4, (4, 1)),  # density
        torch.randint(0, 2, (4, 1)),  # view
        torch.randint(0, len(category_list), (4, 1)),  # category
        torch.randint(0, 6, (4, 1)),  # birads
        torch.randint(0, 2, (4, 1)),  # side
        torch.randn(4, 4)  # xmin, ymin, xmax, ymax (float)
    ], dim=1).to(device)
    # Test training mode
    recon, latent = model(dummy_input, dummy_masked, dummy_attr)
    print("[TRAIN] Reconstructed image shape:", recon.shape)
    print("[TRAIN] Latent representation shape:", latent.shape)

    # Test inference mode (solo attributi)
    recon_inf, latent_inf = model(attributes=dummy_attr)
    print("[INFER] Generated image shape:", recon_inf.shape)
    print("[INFER] Latent representation shape:", latent_inf.shape)
"""

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value):
        # query: [B, Nq, D], key/value: [B, Nk, D]
        attn_output, _ = self.cross_attn(query, key, value)
        x = self.norm(query + attn_output)
        ff_out = self.ff(x)
        out = self.norm2(x + ff_out)
        return out

class GiantTransformerDenoiser(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        # Denoising head
        self.denoise_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, latent, vit_feat, attr_feat):
        # latent: [B, 1, D], vit_feat: [B, 1, D], attr_feat: [B, 1, D]
        x = latent
        # Concateno key/value: ViT + Attribute + Latent
        kv = torch.cat([vit_feat, attr_feat, latent], dim=1)  # [B, 3, D]
        for layer in self.layers:
            x = layer(x, kv, kv)  # Cross-attention tra latente e tutto il resto
        # Denoising head
        x = self.denoise_head(x)
        return x  # [B, 1, D]

class NewLatentDiffusionTransformer(nn.Module):
    def __init__(self, config, category_list=None):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.encoder = AdvancedEncoder(config.input_shape, config.latent_dim)
        self.decoder = AdvancedDecoder(config.input_shape, config.latent_dim)
        self.vit = MaskedImageViT(embed_dim=config.latent_dim)
        self.attr_embedder = AttributeEmbedding(
            embed_dim=config.latent_dim,
            num_density=4,
            num_view=2,
            num_category=len(category_list),
            num_birads=6
        )
        self.transformer = GiantTransformerDenoiser(
            embed_dim=config.latent_dim,
            num_heads=4,
            num_layers=4
        )

    def forward(self, images=None, masked_images=None, attributes=None):
        # Encoder
        if images is not None:
            latent = self.encoder(images)  # [B, D]
            latent = latent.unsqueeze(1)   # [B, 1, D]
        else:
            # Inference: generate random latent
            batch_size = attributes.shape[0]
            latent = torch.randn(batch_size, 1, self.latent_dim, device=attributes.device)

        # ViT features
        if masked_images is not None:
            vit_feat = self.vit(masked_images).unsqueeze(1)  # [B, 1, D]
        else:
            vit_feat = torch.zeros_like(latent)  # [B, 1, D]

        # Attribute features
        attr_feat = self.attr_embedder(attributes).unsqueeze(1)  # [B, 1, D]

        # Transformer denoising
        denoised_latent = self.transformer(latent, vit_feat, attr_feat)  # [B, 1, D]
        denoised_latent = denoised_latent.squeeze(1)  # [B, D]

        # Decoder
        recon = self.decoder(denoised_latent)
        return recon, denoised_latent

if __name__ == "__main__":
    from config_tuning import config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    category_list = [...]  # Inserisci qui la tua lista
    model = NewLatentDiffusionTransformer(config, category_list).to(device)
    dummy_input = torch.randn(4, *config.input_shape).to(device)
    dummy_masked = torch.randn(4, *config.input_shape).to(device)
    dummy_attr = torch.cat([
        torch.randint(0, 4, (4, 1)),  # density
        torch.randint(0, 2, (4, 1)),  # view
        torch.randint(0, len(category_list), (4, 1)),  # category
        torch.randint(0, 6, (4, 1)),  # birads
        torch.randint(0, 2, (4, 1)),  # side
        torch.randn(4, 4)  # xmin, ymin, xmax, ymax (float)
    ], dim=1).to(device)
    recon, latent = model(dummy_input, dummy_masked, dummy_attr)
    print("[NEW TRANSFORMER] Reconstructed image shape:", recon.shape)
    print("[NEW TRANSFORMER] Latent representation shape:", latent.shape)