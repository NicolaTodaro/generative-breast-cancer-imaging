import torch
import torch.optim as optim
import torch.nn as nn
from metrics import compute_ssim
from config_tuning import config
from data_loader import MammogramDataset
from model import NewLatentDiffusionTransformer as LatentDiffusionModel
from risk_predictor import RiskPredictor
import os

def train_image_generation(train_loader, category_list=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LatentDiffusionModel(config, category_list).to(device)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.epochs):
        print(f"[ImageGen] Starting epoch {epoch+1}/{config.epochs}")
        epoch_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            masked_images = batch["masked_image"].to(device)
            attributes = batch["attributes"].to(device)
            optimizer.zero_grad()
            recon, _ = model(images, masked_images, attributes)
            mse_loss = criterion(recon, images)
            ssim_loss = 0
            for i in range(images.size(0)):
                ssim_loss += 1 - compute_ssim(recon[i].detach().cpu().numpy(), images[i].detach().cpu().numpy())
            ssim_loss = ssim_loss / images.size(0)
            loss = mse_loss + ssim_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"[ImageGen] Epoch [{epoch+1}/{config.epochs}], Loss: {avg_loss:.4f}")
        if (epoch + 1) % 50 == 0 or (epoch + 1) == config.epochs:
            if not os.path.exists(config.model_save_path):
                os.makedirs(config.model_save_path)
            torch.save(model.state_dict(), os.path.join(config.model_save_path, f"model_epoch{epoch+1}.pth"))

def train_risk_prediction(train_loader, num_birads=6, attr_dim=1, embed_dim=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiskPredictor(num_birads=num_birads, attr_dim=attr_dim, embed_dim=embed_dim).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.epochs):
        print(f"[RiskPred] Starting epoch {epoch+1}/{config.epochs}")
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            # Choose the attributes you want to use (e.g., only breast_density)
            attributes = batch["attributes"][:, :attr_dim].to(device)
            birads = batch["attributes"][:, 3].long().to(device)  # Assuming BIRADS is the 4th attribute
            optimizer.zero_grad()
            logits = model(images, attributes)
            loss = criterion(logits, birads)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == birads).sum().item()
            total += birads.size(0)
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0
        print(f"[RiskPred] Epoch [{epoch+1}/{config.epochs}], Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
        # model save (optional)
        if (epoch + 1) % 50 == 0 or (epoch + 1) == config.epochs:
            if not os.path.exists(config.model_save_path):
                os.makedirs(config.model_save_path)
            torch.save(model.state_dict(), os.path.join(config.model_save_path, f"risk_predictor_epoch{epoch+1}.pth"))

if __name__ == "__main__":
    train_loader, _ = MammogramDataset.get_data_loaders(config.data_dir, config.batch_size)
    # Choose which training to run:
    train_image_generation(train_loader, train_loader.dataset.category_list if hasattr(train_loader.dataset, 'category_list') else None)
    # train_risk_prediction(train_loader, num_birads=6, attr_dim=1, embed_dim=128)