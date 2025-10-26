import os
import re
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipProcessor

from train import PricePredictor, BASE_MODEL, MAX_LENGTH, RESIZE

# configs
TEST_PATH = "/test.csv"  
CHECKPOINT_PATH = "/training_state_epoch50.pt"
BATCH_SIZE = 32
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "predictions.csv"

#load image from url as and when inferencing
def load_image_from_url(url, resize=RESIZE):
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = image.resize(resize)
        return image
    except Exception:
        return Image.new("RGB", resize, (255, 255, 255))

# TEXT PREPROCESSINGS using functions from 01_textpreprocessing.ipynb
def preprocess_text(Sentence):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    Sentence = Sentence.lower()
    Sentence = re.sub(url_pattern, "", Sentence)
    Sentence = re.sub(r"\.{2,}", ".", Sentence)
    Sentence = re.sub(r"\s+", " ", Sentence).strip()
    Sentence = re.sub(r"[^a-zA-Z0-9\s]", "", Sentence)
    return Sentence
def extract_key_value_pairs(text):
    pattern = r'([A-Za-z0-9 +*#\'"–”“\-]+?):\s*(.*?)(?=\n[A-Za-z0-9 +*#\'"–”“\-]+?:|$)'
    if not isinstance(text, str):
        return {}
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return {k.strip(): v.strip() for k, v in matches}
def build_sentence(row):
    def clean(x):
        if pd.isna(x) or str(x).strip().lower() in ["nan", "none", "null", ""]:
            return ""
        return str(x).strip()
    
    name = clean(row.get("Item_Name", ""))
    desc = clean(row.get("ProductDesc", ""))
    bullets = clean(row.get("Bullet_Points", ""))
    value = clean(row.get("Value", ""))
    unit = clean(row.get("Unit", ""))

    if bullets:
        bullets = re.sub(r'[•;|]+', '\n', bullets) 
    if desc:
        desc = re.sub(r'[•;|]+', '\n', desc)

    parts = []
    if name:
        parts.append(name)
    if desc:
        parts.append(desc)
    if bullets:
        parts.append(bullets)
    if value or unit:
        parts.append(f"{value} {unit}".strip())
    
    text = "\n".join([p for p in parts if p])
    
    text = re.sub(r'\n{2,}', '\n', text).strip()
    
    return text
def remove_repeated_phrases(text):
    if pd.isna(text):
        return text

    parts = re.split(r',|\n| {2,}', text.strip())

    seen = set()
    unique_parts = []
    for part in parts:
        p = part.strip()
        if p and p not in seen:
            seen.add(p)
            unique_parts.append(p)
    cleaned_text = '. '.join(unique_parts)
    if cleaned_text and not cleaned_text.endswith('.'):
        cleaned_text += '.'

    return cleaned_text
# Perform full text preprocessing pipeline as in training
def text_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["extracted"] = df["catalog_content"].apply(extract_key_value_pairs)
    
    if "extracted" in df.columns:
        df = df.join(pd.json_normalize(df["extracted"]))
        df.drop(columns=["extracted"], inplace=True)

    bullet_cols = [col for col in df.columns if re.match(r"(?i)bullet point", col)]
    if bullet_cols:
        df["Bullet_Points"] = df[bullet_cols].apply(
            lambda row: "\n".join(str(v).strip() for v in row if pd.notnull(v) and str(v).strip() != "")
            if any(pd.notnull(v) and str(v).strip() != "" for v in row)
            else np.nan,
            axis=1,
        )
        df["Bullet_Points"] = (
            df["Bullet_Points"].astype(str)
            .str.replace(r"(?i)Bullet Point\s*\d*[:\-]*\s*", "", regex=True)
            .str.strip()
        )
        df.drop(columns=bullet_cols, inplace=True)

    item_cols = [col for col in df.columns if re.match(r"(?i)item name", col)]
    if item_cols:
        df["Item_Name"] = df[item_cols].apply(
            lambda row: "\n".join(str(v).strip() for v in row if pd.notnull(v) and str(v).strip() != "")
            if any(pd.notnull(v) and str(v).strip() != "" for v in row)
            else np.nan,
            axis=1,
        )
        df["Item_Name"] = df["Item_Name"].astype(str).str.strip()
        df.drop(columns=item_cols, inplace=True)

    desc_cols = [col for col in df.columns if re.match(r"(?i)product description", col)]
    if desc_cols:
        df["ProductDesc"] = df[desc_cols].apply(
            lambda row: "\n".join(str(v).strip() for v in row if pd.notnull(v) and str(v).strip() != "")
            if any(pd.notnull(v) and str(v).strip() != "" for v in row)
            else np.nan,
            axis=1,
        )
        df["ProductDesc"] = df["ProductDesc"].astype(str).str.strip()
        df.drop(columns=desc_cols, inplace=True)

    df = df.map(lambda x: re.sub(r"[^\x00-\x7F]+", " ", x).strip() if isinstance(x, str) else x)

    df["Item_Name"] = df["Item_Name"].astype(str).apply(preprocess_text)
    df["Bullet_Points"] = df["Bullet_Points"].astype(str).apply(preprocess_text)
    df["ProductDesc"] = df["ProductDesc"].astype(str).apply(preprocess_text)

    df["text"] = df.apply(build_sentence, axis=1)
    df["cleaned_text"] = df["text"].apply(remove_repeated_phrases)
    return df

#dataset loader
class SiglipTestDataset(Dataset):
    def __init__(self, df, processor, resize=RESIZE):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.resize = resize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row.get("cleaned_text", ""))
        img_url = str(row.get("image_link", ""))

        image = load_image_from_url(img_url, self.resize)

        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        del image
        return {k: v.squeeze(0) for k, v in inputs.items()}
    
def run_inference():

    print(f"Device: {DEVICE}")
    test_df = pd.read_csv(TEST_PATH)
    test_df = test_df[50000:62500]
    print(f"Loaded test data: {len(test_df)} samples")

    test_df = text_preprocess(test_df)

    processor = SiglipProcessor.from_pretrained(BASE_MODEL)
    test_dataset = SiglipTestDataset(test_df, processor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Load model
    model = PricePredictor().to(DEVICE)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)


    model.eval()
    print(f"Loaded model from {CHECKPOINT_PATH}")

    preds = []
    #batch based
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            pred_log, _, _ = model(**batch)
            preds.extend(np.expm1(pred_log.detach().cpu().numpy()))  
# save
    test_df["predicted_price"] = preds
    test_df[["sample_id", "predicted_price"]].to_csv(SAVE_PATH, index=False)
    print(f" Saved predictions to {SAVE_PATH}")


if __name__ == "__main__":
    run_inference()
