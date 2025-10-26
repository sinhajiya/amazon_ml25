import os
import hashlib
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipProcessor, SiglipModel


# setting parameters 
TRAIN_PATH = "/data/train.csv"
VAL_PATH   = "/data/val.csv"
BATCH_SIZE = 128  
NUM_WORKERS = 2
PREFETCH_FACTOR = 2
MAX_LENGTH = 64
BASE_MODEL = "google/siglip-base-patch16-224"
EPOCHS = 50
SAVE_FREQ = 5
CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "cache_images"
RESIZE = (128, 128)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# use cached images 
def _get_cached_path(cache_dir, url):
    hashed = hashlib.md5(url.encode()).hexdigest() + ".jpeg"
    return os.path.join(cache_dir, hashed)

# Loads cached images and text, preprocesses them with SigLIP processor, and returns model-ready tensors with log-transformed price labels
class SiglipPriceDataset(Dataset):
    def __init__(self, df, processor, cache_dir=CACHE_DIR, resize=RESIZE, text_col="cleaned_text"):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.cache_dir = cache_dir
        self.resize = resize
        self.text_col = text_col

    def __len__(self):
        return len(self.df)

    def _get_cached_path(self, url):
        return _get_cached_path(self.cache_dir, url)
# load images
    def _load_image(self, url):
        cached_path = self._get_cached_path(url)
        try:
            return Image.open(cached_path).convert("RGB")
        except Exception:
            return Image.new("RGB", self.resize, (255, 255, 255))
        
# load image, price and text.
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row.get(self.text_col, ""))
        img_url = str(row.get("image_link", ""))
        raw_price = float(row.get("price", 0.0))
        log_price = torch.log1p(torch.tensor(raw_price, dtype=torch.float32))
        image = self._load_image(img_url)

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["price"] = log_price
        return item

# regression head with fused embeddings
class MLP2(nn.Module):
    def __init__(self, dim=512, hidden_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, img_emb, txt_emb):
        fused = 0.3 * img_emb + 0.7 * txt_emb #weighted fusion
        return self.head(fused)

  
# complete model
class PricePredictor(nn.Module):
    def __init__(self, base_model_name=BASE_MODEL):
        super().__init__()
        self.siglip = SiglipModel.from_pretrained(base_model_name)
        embed_dim = self.siglip.config.text_config.hidden_size
        self.mlp_head = MLP2(dim=embed_dim, hidden_dim=embed_dim)
        print(f"Detected SigLIP embedding dimension: {embed_dim}")
# freeze all params in siglip
        for p in self.siglip.parameters():
            p.requires_grad = False

#unfreeze last text layer and text projection layer
        for name, p in self.siglip.named_parameters():
            if "text_projection" in name or "text_model.encoder.layers.11" in name:
                p.requires_grad = True

    def forward(self, pixel_values, input_ids, attention_mask=None):
        out = self.siglip(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        img_emb = out.image_embeds.detach()
        txt_emb = out.text_embeds
        img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-12) #norm
        txt_emb = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + 1e-12)
        price_pred = self.mlp_head(img_emb, txt_emb) #regression head
        return price_pred.squeeze(), img_emb, txt_emb

# smape val
def smape_np(y_true, y_pred, eps=1e-8):
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    num = 2.0 * np.abs(y_pred - y_true)
    return np.mean(num / denom) * 100.0


def main():
    print("Device:", DEVICE)

# load data
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    print(f"Loaded train={len(train_df)} val={len(val_df)}")
# initialize model
    processor = SiglipProcessor.from_pretrained(BASE_MODEL)

    train_dataset = SiglipPriceDataset(train_df, processor, cache_dir=CACHE_DIR, resize=RESIZE)
    val_dataset = SiglipPriceDataset(val_df, processor, cache_dir=CACHE_DIR, resize=RESIZE)
#data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=max(1, NUM_WORKERS // 2),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR,
    )

# compile model
    model = PricePredictor().to(DEVICE)
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile()")
    except Exception as e:
        print("torch.compile not available or failed, continuing without it:", e)

# train use mixed precision training.
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4) # train only 2 layers of siglip and full mlp
    loss_fn = nn.L1Loss()

    resume_path = os.path.join(CHECKPOINT_DIR, "training_state.pt")
    start_epoch = 1
    best_val_smape = float("inf")
    patience_counter = 0
    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch = ck["epoch"] + 1
        best_val_smape = ck.get("best_val_smape", best_val_smape)
        patience_counter = ck.get("patience_counter", 0)
        print(f"Resumed from epoch {ck['epoch']}. Best SMAPE so far: {best_val_smape:.4f}")

    torch.backends.cudnn.benchmark = True

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for batch in pbar:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            prices = batch.pop("price")

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                preds, _, _ = model(**batch)
                loss = loss_fn(preds, prices)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"mae_log": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                trues = batch.pop("price")
                preds, _, _ = model(**batch)
                all_preds.append(preds.detach().cpu().numpy())
                all_trues.append(trues.detach().cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        preds_exp = np.expm1(all_preds)
        trues_exp = np.expm1(all_trues)
        val_smape = smape_np(trues_exp, preds_exp)

        print(f"Epoch {epoch}, Train MAE (log): {avg_train_loss:.4f}, Val SMAPE: {val_smape:.4f}%")
# save model with best smape score
        if val_smape < best_val_smape:
            best_val_smape = val_smape
            patience_counter = 0
            best_path = os.path.join(CHECKPOINT_DIR, "best_by_smape.pt")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path} (SMAPE={best_val_smape:.4f}%)")
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}")

        if epoch % SAVE_FREQ == 0 or epoch == EPOCHS:
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_smape": best_val_smape,
                "patience_counter": patience_counter,
            }
            save_path = os.path.join(CHECKPOINT_DIR, f"training_state_epoch{epoch}.pt")
            torch.save(state, save_path)
            torch.save(state, resume_path)
            print(f"Saved checkpoint: {save_path}")

    print(f"Training finished. Best Val SMAPE: {best_val_smape:.4f}%")
    print(f"Resume checkpoint: {resume_path}")

if __name__ == '__main__':
    main()
