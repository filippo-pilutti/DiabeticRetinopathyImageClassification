import os
import cv2
import random
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# ==========================
# CONFIG
# ==========================
BASE_DIR = Path("dataset")                 # input root
OUT_DIR = Path("dataset_augmented")        # output root

IMAGE_DIRS = ["train_images", "val_images", "test_images"]  # read from all
EXCEL_FILES = ["train.csv.xls", "val.csv.xls", "test.csv.xls"]

# Column names in your label files
FILENAME_COL = "id_code"
LABEL_COL    = "diagnosis"

# Output image format
OUT_EXT = ".jpg"   # ".png" also fine (larger but lossless)

# ---- Balancing strategy (TRAIN only) ----
BALANCE_STRATEGY = "cap"    # "max" | "cap" | "sqrt"
MAX_TARGET_PER_CLASS = 4000 # used when BALANCE_STRATEGY == "cap"
USE_EXPLICIT_TARGETS = False
EXPLICIT_TARGETS = {0: 2184, 1: 2000, 2: 2080, 3: 2040, 4: 2040}

# Preprocessing target size (retina-friendly)
FINAL_SIZE = (640, 640)  # width, height

# Random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BORDER_COLOR = (0, 0, 0)  # or (10,10,10) if you want it slightly gray

# ==========================
# Helpers: read tables
# ==========================
def try_infer_columns(df: pd.DataFrame):
    global FILENAME_COL, LABEL_COL
    cols = {c.lower(): c for c in df.columns}

    if FILENAME_COL is None:
        for guess in ["image", "filename", "file_name", "file", "id", "image_id", "name", "id_code"]:
            if guess in cols:
                FILENAME_COL = cols[guess]; break
    if LABEL_COL is None:
        for guess in ["level", "label", "grade", "diagnosis", "class", "severity"]:
            if guess in cols:
                LABEL_COL = cols[guess]; break

    if FILENAME_COL is None or LABEL_COL is None:
        raise ValueError(
            f"Could not infer filename/label columns. Found columns: {list(df.columns)}.\n"
            f"Set FILENAME_COL and LABEL_COL at the top of the script."
        )

def _read_table_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        engine_candidates = ["openpyxl", None] if suffix == ".xlsx" else ["xlrd", None]
        last_err = None
        for eng in engine_candidates:
            try:
                return pd.read_excel(path, engine=eng) if eng else pd.read_excel(path)
            except Exception as e:
                last_err = e
        try:
            return pd.read_csv(path)
        except Exception:
            pass
        raise RuntimeError(
            f"Could not read {path.name} as Excel and CSV. Last error: {last_err}"
        )
    try:
        return pd.read_excel(path)
    except Exception as excel_err:
        try:
            return pd.read_csv(path)
        except Exception as csv_err:
            raise RuntimeError(
                f"Could not read {path.name}.\n"
                f"Tried Excel (failed with: {excel_err}) and CSV (failed with: {csv_err})."
            )

# ==========================
# Preprocessing (retina-safe)
# ==========================
def crop_fundus_black_border(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img_bgr
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return img_bgr[y:y+h, x:x+w]

def pad_to_square(img_bgr):
    h, w = img_bgr.shape[:2]
    if h == w:
        return img_bgr
    d = max(h, w)
    top = (d - h) // 2
    bottom = d - h - top
    left = (d - w) // 2
    right = d - w - left
    return cv2.copyMakeBorder(img_bgr, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=BORDER_COLOR)

def clahe_lab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def unsharp(img_bgr, k=4, sigma=10):
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    sharp = cv2.addWeighted(img_bgr, 1 + k, blur, -k, 0)
    return sharp

def preprocess(img_bgr):
    img = crop_fundus_black_border(img_bgr)
    img = pad_to_square(img)
    img = clahe_lab(img)
    img = unsharp(img, k=1.5, sigma=10)
    img = cv2.resize(img, FINAL_SIZE, interpolation=cv2.INTER_AREA)
    return img

# ==========================
# Augmentations (gentle)
# ==========================
def random_hflip(img):
    return cv2.flip(img, 1) if random.random() < 0.5 else img  # horizontal only

def random_rotate(img, max_deg=15):
    h, w = img.shape[:2]
    angle = np.random.uniform(-max_deg, max_deg)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=BORDER_COLOR)

def random_zoom(img, max_zoom=0.2):
    h, w = img.shape[:2]
    s = 1.0 + np.random.uniform(-max_zoom, max_zoom)
    nh, nw = int(h*s), int(w*s)
    z = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if s >= 1.0:
        y0 = (nh - h)//2; x0 = (nw - w)//2
        return z[y0:y0+h, x0:x0+w]
    # zooming out: place on black background, not reflected
    canvas = np.zeros_like(img, dtype=np.uint8)
    y0 = (h - nh)//2; x0 = (w - nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = z
    return canvas

    # zooming out: place on reflected canvas to avoid black borders
    canvas = cv2.copyMakeBorder(img, 0, 0, 0, 0, borderType=cv2.BORDER_REFLECT_101)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = z
    return canvas

def jitter_brightness(img, low=0.8, high=1.2):
    factor = np.random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def jitter_contrast(img, low=0.85, high=1.15):
    factor = np.random.uniform(low, high)
    img_f = img.astype(np.float32)
    mean = np.mean(img_f, axis=(0, 1), keepdims=True)
    out = np.clip((img_f - mean) * factor + mean, 0, 255)
    return out.astype(np.uint8)

def jitter_saturation(img, low=0.8, high=1.2):
    factor = np.random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def jitter_gamma(img, low=0.9, high=1.1):
    gamma = np.random.uniform(low, high)
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def augment_once(img):
    out = img.copy()
    out = random_hflip(out)
    out = random_rotate(out, 15)
    out = random_zoom(out, 0.2)
    if random.random() < 0.9: out = jitter_brightness(out, 0.9, 1.1)
    if random.random() < 0.9: out = jitter_contrast(out, 0.9, 1.1)
    if random.random() < 0.9: out = jitter_saturation(out, 0.9, 1.1)
    if random.random() < 0.5: out = jitter_gamma(out, 0.95, 1.05)

    return out

# ==========================
# Pipeline
# ==========================
def main():
    # Prepare output folders (mirror the original split)
    if OUT_DIR.exists():
        print(f"Removing existing {OUT_DIR} ...")
        shutil.rmtree(OUT_DIR)
    for split in ["train_images", "val_images", "test_images"]:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    print("=== START: preprocess all splits, augment ONLY training ===")

    for split_name, img_subdir, xls_name in zip(
        ["train", "val", "test"],
        ["train_images", "val_images", "test_images"],
        ["train.csv.xls", "val.csv.xls", "test.csv.xls"]
    ):
        print(f"\nProcessing split: {split_name}")

        split_dir = BASE_DIR / img_subdir
        out_dir   = OUT_DIR / img_subdir
        label_path = BASE_DIR / xls_name

        if not split_dir.exists() or not label_path.exists():
            print(f"⚠️  Missing split folder or label file for {split_name}. Skipping.")
            continue

        # --- Read labels ---
        labels_df = _read_table_any(label_path)
        try_infer_columns(labels_df)
        labels_df = labels_df[[FILENAME_COL, LABEL_COL]].copy()

        # Normalize names and labels
        def _norm_name(s):
            s = str(s).replace("\\", "/")
            return os.path.basename(s)
        labels_df[FILENAME_COL] = labels_df[FILENAME_COL].apply(_norm_name)
        labels_df[LABEL_COL] = pd.to_numeric(labels_df[LABEL_COL], errors="coerce").astype("Int64")
        labels_df = labels_df.dropna(subset=[LABEL_COL]).astype({LABEL_COL: int}).reset_index(drop=True)

        # Index images within this split
        img_paths = {p.stem.lower(): p for p in split_dir.rglob("*")
                     if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]}
        print(f"  Found {len(img_paths)} images in {split_dir.name}")

        # Match labels to paths
        matched = []
        for _, row in labels_df.iterrows():
            stem = os.path.splitext(str(row[FILENAME_COL]))[0].lower()
            p = img_paths.get(stem)
            if p is not None:
                matched.append((p, int(row[LABEL_COL])))
        if not matched:
            print(f"⚠️  No matches for {split_name}, skipping.")
            continue
        print(f"  Matched {len(matched)} labeled images")

        # --- Preprocess originals (ALL splits) ---
        preprocessed = []
        for p, label in tqdm(matched, desc=f"Preprocessing {split_name}"):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = preprocess(img)
            preprocessed.append((img, label, p.stem))

        # --- Assemble outputs per policy ---
        all_items = []

        if split_name == "train":
            # TRAIN: preprocess + augment (balance inside train only)
            print("  Augmenting training set...")
            orig_counts = Counter(label for _, label, _ in preprocessed)

            if USE_EXPLICIT_TARGETS:
                targets = dict(EXPLICIT_TARGETS)
            else:
                if BALANCE_STRATEGY == "max":
                    max_count = max(orig_counts.values())
                    targets = {c: max_count for c in orig_counts.keys()}
                elif BALANCE_STRATEGY == "cap":
                    max_count = max(orig_counts.values())
                    cap = min(MAX_TARGET_PER_CLASS, max_count)
                    targets = {c: min(cap, max_count) for c in orig_counts.keys()}
                elif BALANCE_STRATEGY == "sqrt":
                    # proportional to sqrt of originals, scaled to mean of originals
                    scale = np.mean(list(orig_counts.values())) / np.mean([np.sqrt(v) for v in orig_counts.values()])
                    targets = {c: int(max(1, round(np.sqrt(v) * scale))) for c, v in orig_counts.items()}
                else:
                    raise ValueError(f"Unknown BALANCE_STRATEGY: {BALANCE_STRATEGY}")

            print("  Target per-class counts (train):", targets)

            by_class = {}
            for img, label, stem in preprocessed:
                by_class.setdefault(label, []).append((img, stem))
                all_items.append((img, label, f"{stem}_proc{OUT_EXT}"))  # one processed copy

            # upsample to target within train
            per_class_counter = Counter(lbl for _, lbl, _ in all_items)
            for label, target in targets.items():
                cur = per_class_counter.get(label, 0)
                src = by_class.get(label, [])
                if not src:
                    print(f"  Warning: class {label} has no originals in train.")
                    continue
                k = 0
                while cur < target:
                    img_src, stem = random.choice(src)
                    aug = augment_once(img_src)
                    new_name = f"{stem}_aug{k:05d}{OUT_EXT}"
                    all_items.append((aug, label, new_name))
                    cur += 1; k += 1
                print(f"  Class {label}: {cur} total (train)")
        else:
            # VAL/TEST: preprocess ONLY (no augmentation)
            print("  No augmentation for validation/test.")
            for img, label, stem in preprocessed:
                all_items.append((img, label, f"{stem}_proc{OUT_EXT}"))

        # --- Save images & CSV for this split ---
        cache = {filename: img for (img, _label, filename) in all_items}

        for _img, _label, filename in tqdm(all_items, desc=f"Saving {split_name}"):
            out_path = out_dir / filename
            if OUT_EXT.lower() in [".jpg", ".jpeg"]:
                cv2.imwrite(str(out_path), cache[filename], [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(str(out_path), cache[filename])

        df_out = pd.DataFrame([(filename, label) for (_img, label, filename) in all_items],
                              columns=["filename", "label"])
        df_out.to_csv(OUT_DIR / f"{split_name}.csv", index=False)

        # Quick counts log
        counts = Counter(lbl for (_img, lbl, _fn) in all_items)
        print("  Per-class counts written:", dict(counts))
        print(f"✅ {split_name} done — {len(all_items)} images written.")

    print("\n=== ALL SPLITS COMPLETED ===")
    print(f"Augmented dataset written to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
