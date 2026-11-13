from __future__ import annotations

import base64
import csv
import math
import time
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VISION_MODEL_PATH = BASE_DIR / "models" / "money_vision_model.joblib"
REAL_DATASET_DIR = BASE_DIR / "data" / "external" / "ph_banknotes_coins"

BILL_DENOMINATIONS = [20, 50, 100, 200, 500, 1000]

# BSP Official Banknote Color Specs (BGR reference centers + names)
BILL_SPECS: dict[int, dict[str, Any]] = {
    1000: {"name": "₱1,000 Bill", "color_name": "Blue", "bgr": (195, 130, 52), "box_bgr": (245, 165, 60)},
    500: {"name": "₱500 Bill", "color_name": "Yellow / Gold", "bgr": (55, 195, 228), "box_bgr": (0, 215, 255)},
    200: {"name": "₱200 Bill", "color_name": "Green", "bgr": (88, 175, 82), "box_bgr": (75, 210, 85)},
    100: {"name": "₱100 Bill", "color_name": "Violet / Mauve", "bgr": (182, 108, 152), "box_bgr": (220, 120, 200)},
    50: {"name": "₱50 Bill", "color_name": "Red", "bgr": (72, 68, 208), "box_bgr": (70, 70, 245)},
    20: {"name": "₱20 Bill", "color_name": "Orange", "bgr": (52, 128, 224), "box_bgr": (40, 145, 255)},
}

# BSP NGC Coin Diameters in mm
NGC_COIN_MM: list[tuple[float, float, str]] = [
    (20.0, 30.0, "₱20 Coin (Bimetallic)"),
    (10.0, 27.0, "₱10 Coin"),
    (5.0, 25.0, "₱5 Coin"),
    (1.0, 23.0, "₱1 Coin"),
    (0.25, 20.0, "25¢ Coin"),
    (0.05, 16.0, "5¢ Coin"),
]

REAL_CLASS_META: dict[str, dict[str, Any]] = {
    "bill_1000": {"type": "bill", "denomination": 1000.0, "label": "₱1,000 Bill", "box_bgr": (245, 165, 60)},
    "bill_500": {"type": "bill", "denomination": 500.0, "label": "₱500 Bill", "box_bgr": (0, 215, 255)},
    "bill_200": {"type": "bill", "denomination": 200.0, "label": "₱200 Bill", "box_bgr": (75, 210, 85)},
    "bill_100": {"type": "bill", "denomination": 100.0, "label": "₱100 Bill", "box_bgr": (220, 120, 200)},
    "bill_50": {"type": "bill", "denomination": 50.0, "label": "₱50 Bill", "box_bgr": (70, 70, 245)},
    "bill_20": {"type": "bill", "denomination": 20.0, "label": "₱20 Bill", "box_bgr": (40, 145, 255)},
    "coin_20": {"type": "coin", "denomination": 20.0, "label": "₱20 Coin (Bimetallic)", "box_bgr": (56, 189, 248)},
    "coin_10": {"type": "coin", "denomination": 10.0, "label": "₱10 Coin", "box_bgr": (56, 189, 248)},
    "coin_5": {"type": "coin", "denomination": 5.0, "label": "₱5 Coin", "box_bgr": (56, 189, 248)},
    "coin_1": {"type": "coin", "denomination": 1.0, "label": "₱1 Coin", "box_bgr": (56, 189, 248)},
}


def _parse_dataset_class(label: str) -> str:
    parts = label.split("_")
    val = parts[0]
    if val in ("1000", "500", "200", "100", "50"):
        return f"bill_{val}"
    if val == "20":
        return "coin_20" if len(parts) > 1 and parts[1] == "New" else "bill_20"
    return f"coin_{val}"


def _extract_patch_features(bgr_patch: np.ndarray) -> np.ndarray:
    """Extract illumination-resilient Lab + HSV + RGB ratio features from a bill interior patch."""
    if bgr_patch.size == 0:
        return np.zeros(44, dtype=np.float32)

    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2LAB)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat >= 16) & (val >= 30) & ~((sat < 20) & (val > 225))
    if np.count_nonzero(mask) < max(30, int(0.08 * bgr_patch.shape[0] * bgr_patch.shape[1])):
        mask = np.ones(bgr_patch.shape[:2], dtype=bool)

    bgr_sel = bgr_patch[mask].astype(np.float32)
    hsv_sel = hsv[mask].astype(np.float32)
    lab_sel = lab[mask].astype(np.float32)

    rgb_sum = np.maximum(bgr_sel.sum(axis=1, keepdims=True), 1.0)
    bgr_norm = bgr_sel / rgb_sum
    b_mean, g_mean, r_mean = bgr_norm.mean(axis=0)
    b_std, g_std, r_std = bgr_norm.std(axis=0)

    l_med, a_med, b_lab_med = np.median(lab_sel, axis=0)
    l_std, a_std, b_lab_std = lab_sel.std(axis=0)

    hue_hist, _ = np.histogram(hsv_sel[:, 0], bins=24, range=(0, 180), weights=hsv_sel[:, 1] + 1.0)
    hue_hist = hue_hist / max(float(hue_hist.sum()), 1.0)

    s_med = float(np.median(hsv_sel[:, 1])) / 255.0
    v_med = float(np.median(hsv_sel[:, 2])) / 255.0
    hue_rad = (hsv_sel[:, 0] / 180.0) * (2.0 * np.pi)
    hue_weights = hsv_sel[:, 1] + 1.0
    hue_sin = float(np.sum(np.sin(hue_rad) * hue_weights) / np.sum(hue_weights))
    hue_cos = float(np.sum(np.cos(hue_rad) * hue_weights) / np.sum(hue_weights))

    r_over_g = float(np.clip(r_mean / max(g_mean, 0.05), 0.2, 5.0))
    g_over_b = float(np.clip(g_mean / max(b_mean, 0.05), 0.2, 5.0))
    rg_minus_b = float((r_mean + g_mean) * 0.5 - b_mean)
    rb_minus_g = float((r_mean + b_mean) * 0.5 - g_mean)

    vec = np.concatenate(
        [
            np.array(
                [
                    b_mean, g_mean, r_mean,
                    b_std, g_std, r_std,
                    l_med / 255.0, (a_med - 128.0) / 128.0, (b_lab_med - 128.0) / 128.0,
                    l_std / 255.0, a_std / 128.0, b_lab_std / 128.0,
                    s_med, v_med, hue_sin, hue_cos,
                    rg_minus_b, rb_minus_g, r_over_g, g_over_b,
                ],
                dtype=np.float32,
            ),
            hue_hist.astype(np.float32),
        ]
    )
    return vec


def _extract_roi_features(img_bgr: np.ndarray) -> np.ndarray:
    """Extract 93-dim centered spatial Lab/HSV + ring-contrast + gradient-orientation features."""
    img = cv2.resize(img_bgr, (192, 192), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Center 56% ROI avoids cloth/bag borders in handheld/table photos
    c_hsv = hsv[42:150, 42:150]
    c_lab = lab[42:150, 42:150]
    core_hsv = hsv[64:128, 64:128]
    core_lab = lab[64:128, 64:128]

    lab_f = lab.astype(np.float32)
    border = np.concatenate(
        [lab_f[:8].reshape(-1, 3), lab_f[-8:].reshape(-1, 3), lab_f[:, :8].reshape(-1, 3), lab_f[:, -8:].reshape(-1, 3)]
    )
    bg = np.median(border, axis=0)
    de = np.sqrt(np.sum((lab_f - bg) ** 2, axis=2))
    fg_mask = de > max(15.0, float(np.percentile(de, 58)))
    fg_ratio = float(np.mean(fg_mask))
    ys, xs = np.where(fg_mask)
    if len(ys) > 50:
        fg_box_ar = float((xs.max() - xs.min() + 1) / max(ys.max() - ys.min() + 1, 1))
        fg_box_fill = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1) / (192.0 * 192.0))
    else:
        fg_box_ar, fg_box_fill = 1.0, 0.5

    grid_feats: list[float] = []
    for gy in range(3):
        for gx in range(3):
            pl = c_lab[gy * 36 : (gy + 1) * 36, gx * 36 : (gx + 1) * 36].astype(np.float32)
            ph = c_hsv[gy * 36 : (gy + 1) * 36, gx * 36 : (gx + 1) * 36].astype(np.float32)
            grid_feats.extend(
                [
                    float(pl[:, :, 0].mean() / 255.0),
                    float((pl[:, :, 1].mean() - 128.0) / 128.0),
                    float((pl[:, :, 2].mean() - 128.0) / 128.0),
                    float(ph[:, :, 1].mean() / 255.0),
                    float(ph[:, :, 2].mean() / 255.0),
                ]
            )

    hue_hist, _ = np.histogram(
        core_hsv[:, :, 0], bins=20, range=(0, 180), weights=core_hsv[:, :, 1].astype(np.float32) + 1.0
    )
    hue_hist = hue_hist / max(float(hue_hist.sum()), 1.0)

    yy, xx = np.ogrid[:108, :108]
    dist = np.sqrt((xx - 54) ** 2 + (yy - 54) ** 2)
    core = c_lab[dist <= 24].astype(np.float32)
    ring = c_lab[(dist >= 28) & (dist <= 48)].astype(np.float32)
    core_b = float((core[:, 2].mean() - 128.0) / 128.0) if len(core) else 0.0
    ring_b = float((ring[:, 2].mean() - 128.0) / 128.0) if len(ring) else 0.0
    core_a = float((core[:, 1].mean() - 128.0) / 128.0) if len(core) else 0.0
    ring_a = float((ring[:, 1].mean() - 128.0) / 128.0) if len(ring) else 0.0
    core_s = float(c_hsv[dist <= 24][:, 1].mean() / 255.0)
    ring_s = float(c_hsv[(dist >= 28) & (dist <= 48)][:, 1].mean() / 255.0)

    g_roi = gray[42:150, 42:150]
    gx_img = cv2.Sobel(g_roi, cv2.CV_32F, 1, 0, ksize=3)
    gy_img = cv2.Sobel(g_roi, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx_img, gy_img)
    ori_hist, _ = np.histogram((np.arctan2(gy_img, gx_img) + np.pi), bins=8, range=(0, 2 * np.pi), weights=mag)
    ori_hist = ori_hist / max(float(ori_hist.sum()), 1.0)

    sel_lab = core_lab.reshape(-1, 3).astype(np.float32)
    sel_hsv = core_hsv.reshape(-1, 3).astype(np.float32)
    c_stats = [
        float(np.median(sel_lab[:, 0]) / 255.0),
        float((np.median(sel_lab[:, 1]) - 128.0) / 128.0),
        float((np.median(sel_lab[:, 2]) - 128.0) / 128.0),
        float(sel_lab[:, 1].std() / 128.0),
        float(sel_lab[:, 2].std() / 128.0),
        float(np.median(sel_hsv[:, 1]) / 255.0),
        float(np.percentile(sel_hsv[:, 1], 80) / 255.0),
        fg_ratio,
        fg_box_ar,
        fg_box_fill,
        core_b,
        ring_b,
        ring_b - core_b,
        core_a,
        ring_a,
        core_s,
        ring_s,
        float(mag.mean() / 255.0),
        float(mag.std() / 255.0),
        float(np.percentile(mag, 90) / 255.0),
    ]
    return np.concatenate(
        [np.array(grid_feats + c_stats, dtype=np.float32), hue_hist.astype(np.float32), ori_hist.astype(np.float32)]
    )


def _synth_bill_scene_image(denom: int, rng: np.random.Generator) -> np.ndarray:
    """Create a realistic single-bill scene image for denominations missing from external photos (₱500, ₱100)."""
    bg_choices = [(220, 222, 224), (48, 44, 40), (72, 62, 54), (195, 190, 185)]
    bg = np.array(bg_choices[int(rng.integers(0, len(bg_choices)))], dtype=np.float32)
    canvas = np.full((192, 192, 3), bg, dtype=np.float32)
    canvas += rng.normal(0.0, 5.0, size=canvas.shape)

    base_bgr = np.array(BILL_SPECS[denom]["bgr"], dtype=np.float32)
    bw = int(rng.integers(126, 168))
    bh = int(rng.integers(74, 114))
    bill = np.full((bh, bw, 3), base_bgr, dtype=np.float32)
    cv2.ellipse(bill, (bw // 3, bh // 2), (bw // 7, bh // 3), 0, 0, 360, np.clip(base_bgr + 34, 0, 255).tolist(), -1)
    cv2.putText(bill, f"{denom}", (int(bw * 0.52), int(bh * 0.65)), cv2.FONT_HERSHEY_DUPLEX, 0.9, (245, 245, 245), 2)
    gain = float(rng.uniform(0.80, 1.18))
    bill = np.clip(bill * gain + rng.normal(0.0, 7.0, size=bill.shape), 0, 255)

    x1 = (192 - bw) // 2 + int(rng.integers(-10, 11))
    y1 = (192 - bh) // 2 + int(rng.integers(-10, 11))
    x1 = max(2, min(190 - bw, x1))
    y1 = max(2, min(190 - bh, y1))
    canvas[y1 : y1 + bh, x1 : x1 + bw] = bill
    return np.clip(canvas, 0, 255).astype(np.uint8)


def train_money_vision_model(model_path: Path = DEFAULT_VISION_MODEL_PATH) -> Path:
    """Train both:
    1. Bill denomination classifier (multi-item tabletop + real banknote crops, 6 classes: ₱20..₱1000)
    2. Unified real-photo currency classifier trained on the 80% split of RichMiguel/Banknotes-Coins + ₱500/₱100 anchors
    """
    rng = np.random.default_rng(42)

    palette_anchors: dict[int, list[tuple[int, int, int]]] = {
        1000: [(195, 130, 52), (210, 145, 70), (175, 120, 62), (205, 155, 92)],
        500: [(55, 195, 228), (75, 205, 232), (82, 185, 212), (96, 196, 222)],
        200: [(88, 175, 82), (105, 185, 98), (78, 155, 80), (112, 175, 108)],
        100: [(182, 108, 152), (192, 124, 165), (165, 98, 142), (186, 136, 172)],
        50: [(72, 68, 208), (85, 82, 214), (70, 60, 182), (96, 94, 196)],
        20: [(52, 128, 224), (66, 142, 228), (58, 118, 198), (78, 148, 215)],
    }

    X_bill: list[np.ndarray] = []
    y_bill: list[int] = []

    for denom, anchors in palette_anchors.items():
        for i in range(220):
            base = np.array(anchors[int(rng.integers(0, len(anchors)))], dtype=np.float32)
            patch = np.zeros((46, 110, 3), dtype=np.float32)
            patch[:, :] = base
            if i % 4 != 0:
                text_mask = rng.random((46, 110)) < rng.uniform(0.10, 0.30)
                patch[text_mask] = np.clip(base * rng.uniform(0.48, 0.78), 0, 255)
                light_mask = rng.random((46, 110)) < rng.uniform(0.05, 0.16)
                patch[light_mask] = np.clip(base * 0.45 + 130.0, 0, 255)
            gain = rng.uniform(0.70, 1.24)
            wb_shift = rng.uniform(0.93, 1.07, size=(1, 1, 3))
            noise = rng.normal(0.0, 5.0 if i % 4 == 0 else 8.0, size=patch.shape)
            patch = np.clip(patch * gain * wb_shift + noise, 0, 255).astype(np.uint8)
            X_bill.append(_extract_patch_features(patch))
            y_bill.append(denom)

    # Integrate real dataset training split (strictly index % 50 < 40 so 20% stays unseen test)
    X_roi: list[np.ndarray] = []
    y_roi: list[str] = []
    real_train_count = 0

    meta_csv = REAL_DATASET_DIR / "metadata.csv"
    if meta_csv.exists():
        with open(meta_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if int(r["index"]) % 50 >= 40:
                    continue
                img_path = REAL_DATASET_DIR / r["path"]
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                real_train_count += 1
                cls_name = _parse_dataset_class(r["label"])
                X_roi.append(_extract_roi_features(img))
                y_roi.append(cls_name)

                # Also feed real center crops of banknotes into bill_classifier
                if cls_name.startswith("bill_"):
                    denom_int = int(cls_name.split("_")[1])
                    h, w = img.shape[:2]
                    crop = img[int(h * 0.25) : int(h * 0.75), int(w * 0.22) : int(w * 0.78)]
                    X_bill.append(_extract_patch_features(crop))
                    y_bill.append(denom_int)

    # Supplement real ROI classifier with ₱500 and ₱100 synthetic scenes
    for extra_denom in (500, 100):
        for _ in range(90):
            s_img = _synth_bill_scene_image(extra_denom, rng)
            X_roi.append(_extract_roi_features(s_img))
            y_roi.append(f"bill_{extra_denom}")

    bill_clf = VotingClassifier(
        estimators=[
            ("lr", Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=8.0, max_iter=1000, random_state=42))])),
            ("rf", RandomForestClassifier(n_estimators=160, max_depth=14, random_state=42)),
        ],
        voting="soft",
    )
    bill_clf.fit(np.vstack(X_bill), np.array(y_bill, dtype=np.int32))

    roi_clf = None
    if X_roi:
        roi_clf = VotingClassifier(
            estimators=[
                ("lr", Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=6.0, max_iter=1500, random_state=42))])),
                ("et", ExtraTreesClassifier(n_estimators=240, random_state=42)),
                ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ],
            voting="soft",
        )
        roi_clf.fit(np.vstack(X_roi), np.array(y_roi))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": bill_clf,
            "roi_classifier": roi_clf,
            "denominations": BILL_DENOMINATIONS,
            "real_train_images": real_train_count,
        },
        model_path,
    )
    return model_path


def white_balance_gray_world(img: np.ndarray) -> np.ndarray:
    """Illumination correction: anchors to light neutral borders when available, else clamped Gray-World."""
    img_f = img.astype(np.float32)
    border_bgr = np.concatenate(
        [
            img_f[:10, :, :].reshape(-1, 3),
            img_f[-10:, :, :].reshape(-1, 3),
            img_f[:, :10, :].reshape(-1, 3),
            img_f[:, -10:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    med_b, med_g, med_r = np.median(border_bgr, axis=0)
    border_spread = max(med_b, med_g, med_r) - min(med_b, med_g, med_r)
    border_lum = (med_b + med_g + med_r) / 3.0

    if border_lum > 150.0 and border_spread < 65.0:
        gains = np.array(
            [
                np.clip(border_lum / max(med_b, 1.0), 0.72, 1.38),
                np.clip(border_lum / max(med_g, 1.0), 0.72, 1.38),
                np.clip(border_lum / max(med_r, 1.0), 0.72, 1.38),
            ],
            dtype=np.float32,
        )
        return np.clip(img_f * gains, 0, 255).astype(np.uint8)

    b_avg, g_avg, r_avg = img_f[:, :, 0].mean(), img_f[:, :, 1].mean(), img_f[:, :, 2].mean()
    gray_avg = (b_avg + g_avg + r_avg) / 3.0
    if gray_avg < 10:
        return img
    gains = np.array(
        [
            np.clip(gray_avg / max(b_avg, 1.0), 0.80, 1.25),
            np.clip(gray_avg / max(g_avg, 1.0), 0.80, 1.25),
            np.clip(gray_avg / max(r_avg, 1.0), 0.80, 1.25),
        ],
        dtype=np.float32,
    )
    return np.clip(img_f * gains, 0, 255).astype(np.uint8)


def _warp_bill_patch(img: np.ndarray, rect: tuple[tuple[float, float], tuple[float, float], float]) -> np.ndarray:
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    h, w = img.shape[:2]
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    x1 = max(0, int(round(cx - rw * 0.36)))
    x2 = min(w, int(round(cx + rw * 0.36)))
    y1 = max(0, int(round(cy - rh * 0.34)))
    y2 = min(h, int(round(cy + rh * 0.34)))
    return rotated[y1:y2, x1:x2]


def _analyze_coin(
    img_wb: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    px_per_mm: float | None,
) -> dict[str, Any]:
    h, w = img_wb.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    core_mask = dist <= (0.50 * radius)
    ring_mask = (dist >= 0.62 * radius) & (dist <= 0.90 * radius)

    lab = cv2.cvtColor(img_wb, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_wb, cv2.COLOR_BGR2HSV).astype(np.float32)

    core_lab = lab[core_mask] if np.any(core_mask) else lab[dist <= radius]
    ring_lab = lab[ring_mask] if np.any(ring_mask) else core_lab
    full_hsv = hsv[dist <= 0.90 * radius]

    core_b = float(np.median(core_lab[:, 2]) - 128.0) if len(core_lab) else 0.0
    ring_b = float(np.median(ring_lab[:, 2]) - 128.0) if len(ring_lab) else 0.0
    ring_a = float(np.median(ring_lab[:, 1]) - 128.0) if len(ring_lab) else 0.0
    mean_sat = float(np.median(full_hsv[:, 1])) if len(full_hsv) else 0.0
    ring_sat = float(np.median(hsv[ring_mask][:, 1])) if np.any(ring_mask) else 0.0
    core_sat = float(np.median(hsv[core_mask][:, 1])) if np.any(core_mask) else 0.0

    diameter_px = 2.0 * radius
    est_mm = diameter_px / px_per_mm if px_per_mm and px_per_mm > 0 else None

    if (ring_b - core_b > 5.5 and ring_sat > 32) or (ring_a > 4.0 and ring_b > 9.0 and ring_sat > 40 and core_sat < ring_sat - 10):
        if est_mm is None or est_mm >= 26.0:
            return {"denomination": 20.0, "label": "₱20 Coin (Bimetallic)", "confidence": 0.88, "is_bimetallic_20": True, "est_mm": est_mm}

    if core_b - ring_b > 6.5 and core_sat > 35:
        return {"denomination": 10.0, "label": "₱10 Coin (Bimetallic)", "confidence": 0.82, "is_bimetallic_20": False, "est_mm": est_mm}

    if mean_sat > 48 and core_b > 12.0 and ring_b > 12.0:
        return {"denomination": 5.0, "label": "₱5 Coin (Brass)", "confidence": 0.78, "is_bimetallic_20": False, "est_mm": est_mm}

    if est_mm is not None:
        silver_specs = [
            (10.0, 27.0, "₱10 Coin"),
            (5.0, 25.0, "₱5 Coin"),
            (1.0, 23.0, "₱1 Coin"),
            (0.25, 20.0, "25¢ Coin"),
            (0.05, 16.0, "5¢ Coin"),
        ]
        diffs = [(abs(est_mm - mm), val, lbl) for val, mm, lbl in silver_specs]
        diffs.sort(key=lambda x: x[0])
        best_diff, best_val, best_lbl = diffs[0]
        conf = float(np.clip(0.88 - best_diff * 0.08, 0.48, 0.90))
        return {"denomination": best_val, "label": best_lbl, "confidence": round(conf, 3), "is_bimetallic_20": False, "est_mm": round(est_mm, 1)}

    return {"denomination": None, "label": "Silver Coin (Uncalibrated)", "confidence": 0.45, "is_bimetallic_20": False, "est_mm": None}


class PhilippineMoneyCounter:
    """Computer-vision + ML detector & value counter for Philippine Peso bills and coins."""

    def __init__(self, model_path: Path = DEFAULT_VISION_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            train_money_vision_model(self.model_path)
        bundle = joblib.load(self.model_path)
        self.classifier = bundle["classifier"]
        self.roi_classifier = bundle.get("roi_classifier")
        if self.roi_classifier is None and (REAL_DATASET_DIR / "metadata.csv").exists():
            train_money_vision_model(self.model_path)
            bundle = joblib.load(self.model_path)
            self.classifier = bundle["classifier"]
            self.roi_classifier = bundle.get("roi_classifier")
        self.classes = list(self.classifier.classes_)
        self.real_train_images = int(bundle.get("real_train_images", 0))

    def count_from_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image. Please upload a valid PNG, JPEG, or WebP photo.")
        return self.count_image(img)

    def _classify_single_currency_roi(self, img_bgr: np.ndarray, bbox: list[int] | None = None) -> dict[str, Any] | None:
        if self.roi_classifier is None:
            return None
        feat = _extract_roi_features(img_bgr).reshape(1, -1)
        probas = self.roi_classifier.predict_proba(feat)[0]
        best_idx = int(np.argmax(probas))
        cls_key = str(self.roi_classifier.classes_[best_idx])
        conf = float(probas[best_idx])
        meta = REAL_CLASS_META[cls_key]
        h, w = img_bgr.shape[:2]
        use_bbox = bbox if bbox is not None else [int(w * 0.12), int(h * 0.14), int(w * 0.76), int(h * 0.72)]
        return {
            "type": meta["type"],
            "denomination": float(meta["denomination"]),
            "label": meta["label"],
            "confidence": round(conf, 3),
            "bbox": use_bbox,
            "box_bgr": meta["box_bgr"],
            "cls_key": cls_key,
        }

    def count_image(self, img_bgr: np.ndarray) -> dict[str, Any]:
        t0 = time.perf_counter()
        orig_h, orig_w = img_bgr.shape[:2]
        scale = min(1.0, 1024.0 / max(orig_h, orig_w))
        if scale < 1.0:
            img = cv2.resize(img_bgr, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img = img_bgr.copy()

        H, W = img.shape[:2]
        img_area = float(H * W)
        img_wb = white_balance_gray_world(img)
        annotated = img.copy()

        lab = cv2.cvtColor(img_wb, cv2.COLOR_BGR2LAB).astype(np.float32)
        border_pixels = np.concatenate(
            [
                lab[:10, :, :].reshape(-1, 3),
                lab[-10:, :, :].reshape(-1, 3),
                lab[:, :10, :].reshape(-1, 3),
                lab[:, -10:, :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg_lab = np.median(border_pixels, axis=0)
        delta_e = np.sqrt(np.sum((lab - bg_lab) ** 2, axis=2))
        delta_u8 = np.clip(delta_e * 2.2, 0, 255).astype(np.uint8)

        otsu_thresh, _ = cv2.threshold(delta_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        eff_thresh = max(24, min(72, int(otsu_thresh)))
        fg_mask = (delta_u8 >= eff_thresh).astype(np.uint8) * 255

        gray = cv2.cvtColor(img_wb, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 35, 110)
        combined = cv2.bitwise_or(fg_mask, edges)

        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close, iterations=1)
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bill_items: list[dict[str, Any]] = []
        coin_candidates: list[dict[str, Any]] = []
        suppression_boxes: list[tuple[int, int, int, int]] = []
        px_per_mm_samples: list[float] = []
        large_contours: list[tuple[float, list[int], float]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.0012 * img_area or area > 0.88 * img_area:
                continue

            perim = cv2.arcLength(cnt, True)
            if perim <= 0:
                continue

            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), _angle = rect
            long_side = max(rw, rh)
            short_side = min(rw, rh)
            if short_side < 14:
                continue

            aspect = long_side / short_side
            rect_area = rw * rh
            rectangularity = area / max(rect_area, 1.0)
            circularity = (4.0 * math.pi * area) / (perim * perim)
            x, y, bw, bh = cv2.boundingRect(cnt)
            large_contours.append((area / img_area, [int(x), int(y), int(bw), int(bh)], aspect))

            if 1.50 <= aspect <= 3.45 and rectangularity >= 0.62 and long_side >= 0.075 * min(H, W):
                patch = _warp_bill_patch(img_wb, rect)
                feat = _extract_patch_features(patch).reshape(1, -1)
                probas = self.classifier.predict_proba(feat)[0]
                best_idx = int(np.argmax(probas))
                denom = int(self.classes[best_idx])
                conf = float(probas[best_idx])

                spec = BILL_SPECS[denom]
                bill_items.append(
                    {
                        "type": "bill",
                        "denomination": float(denom),
                        "label": spec["name"],
                        "confidence": round(conf, 3),
                        "bbox": [int(x), int(y), int(bw), int(bh)],
                        "center": [round(cx, 1), round(cy, 1)],
                        "rect": rect,
                        "aspect": round(aspect, 3),
                    }
                )
                pad = 16
                suppression_boxes.append((max(0, x - pad), max(0, y - pad), min(W, x + bw + pad), min(H, y + bh + pad)))
                px_per_mm_samples.append(short_side / 66.0)
                px_per_mm_samples.append(long_side / 160.0)
            elif 0.72 <= aspect <= 1.38 and circularity >= 0.52 and short_side >= 16:
                # Large square-ish contour (> 18% of image) is more likely a squished/folded bill than a coin
                if area / img_area > 0.18:
                    pad = 12
                    suppression_boxes.append((max(0, x - pad), max(0, y - pad), min(W, x + bw + pad), min(H, y + bh + pad)))
                else:
                    coin_candidates.append(
                        {
                            "cx": float(cx),
                            "cy": float(cy),
                            "radius": float((long_side + short_side) * 0.25),
                            "bbox": [int(x), int(y), int(bw), int(bh)],
                        }
                    )

        # Supplementary HoughCircles scan to catch shiny silver coins on bright surfaces
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(24, int(min(H, W) * 0.045)),
            param1=85,
            param2=34,
            minRadius=max(10, int(min(H, W) * 0.016)),
            maxRadius=max(28, int(min(H, W) * 0.11)),
        )
        if circles is not None:
            for c in np.round(circles[0, :]).astype(int):
                hcx, hcy, hr = int(c[0]), int(c[1]), int(c[2])
                inside_bill = any(bx1 <= hcx <= bx2 and by1 <= hcy <= by2 for bx1, by1, bx2, by2 in suppression_boxes)
                dup_coin = any(math.hypot(hcx - cc["cx"], hcy - cc["cy"]) < max(hr, cc["radius"]) * 1.15 for cc in coin_candidates)
                if not inside_bill and not dup_coin:
                    coin_candidates.append(
                        {
                            "cx": float(hcx),
                            "cy": float(hcy),
                            "radius": float(hr),
                            "bbox": [int(hcx - hr), int(hcy - hr), int(2 * hr), int(2 * hr)],
                        }
                    )

        filtered_coins = [
            cc
            for cc in coin_candidates
            if not any(bx1 <= cc["cx"] <= bx2 and by1 <= cc["cy"] <= by2 for bx1, by1, bx2, by2 in suppression_boxes)
        ]

        # Detect single-currency close-up photo mode (e.g., close-up bill or close-up coin smartphone shot)
        # where square crop, perspective tilt, or hand/fold distorts aspect ratio or spawns uncalibrated coin ambiguity.
        has_multi_tabletop = (
            max(H, W) > 360
            and (
                len(bill_items) >= 2
                or (len(bill_items) == 1 and len(filtered_coins) >= 1 and bill_items[0]["aspect"] >= 1.68)
            )
        )
        max_contour_frac = max((frac for frac, _, _ in large_contours), default=0.0)
        single_object_closeup = (
            self.roi_classifier is not None
            and not has_multi_tabletop
            and (
                max(H, W) <= 360
                or (len(bill_items) + len(filtered_coins) <= 1)
                or (len(bill_items) == 1 and max_contour_frac >= 0.18)
            )
        )

        if single_object_closeup:
            # Prefer tightest prominent contour bbox if reasonable, else central crop box
            best_bbox = None
            if large_contours:
                large_contours.sort(key=lambda t: t[0], reverse=True)
                if 0.04 <= large_contours[0][0] <= 0.85:
                    best_bbox = large_contours[0][1]

            roi_pred = self._classify_single_currency_roi(img, bbox=best_bbox)
            min_roi_conf = 0.20 if (len(bill_items) + len(filtered_coins) == 0) else 0.34
            if roi_pred is not None and roi_pred["confidence"] >= min_roi_conf:
                bx, by, bw, bh = roi_pred["bbox"]
                box_color = roi_pred["box_bgr"]
                cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), box_color, 3)
                lbl_short = f"P{int(roi_pred['denomination']) if roi_pred['denomination'] >= 1 else roi_pred['denomination']:.2f}"
                tag = f"#1 {lbl_short} ({int(roi_pred['confidence'] * 100)}%)"
                ty = max(22, by - 8)
                cv2.rectangle(annotated, (bx, ty - 20), (bx + len(tag) * 9 + 10, ty + 4), box_color, -1)
                cv2.putText(annotated, tag, (bx + 4, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (15, 23, 42), 2, cv2.LINE_AA)

                item_out = {
                    "id": 1,
                    "type": roi_pred["type"],
                    "denomination": roi_pred["denomination"],
                    "label": roi_pred["label"],
                    "confidence": roi_pred["confidence"],
                    "bbox": roi_pred["bbox"],
                    "center": [round(bx + bw / 2.0, 1), round(by + bh / 2.0, 1)],
                }
                total_php = round(float(item_out["denomination"]), 2)
                ok, enc = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii") if ok else ""
                return {
                    "total_php": total_php,
                    "formatted_total": f"₱{total_php:,.2f}",
                    "bill_count": 1 if item_out["type"] == "bill" else 0,
                    "coin_count": 1 if item_out["type"] == "coin" else 0,
                    "item_count": 1,
                    "items": [item_out],
                    "breakdown": {item_out["label"]: 1},
                    "scale_source": "real_currency_roi_classifier (trained on PH Banknotes-Coins)",
                    "warnings": [],
                    "annotated_image_base64": annotated_b64,
                    "processing_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

        px_per_mm: float | None = float(np.median(px_per_mm_samples)) if px_per_mm_samples else None
        scale_source = "banknote_reference (160x66mm)" if px_per_mm else "none"

        if px_per_mm is None and filtered_coins:
            for cc in filtered_coins:
                probe = _analyze_coin(img_wb, cc["cx"], cc["cy"], cc["radius"], None)
                if probe.get("is_bimetallic_20"):
                    px_per_mm = (2.0 * cc["radius"]) / 30.0
                    scale_source = "p20_bimetallic_coin (30mm)"
                    break

        warnings: list[str] = []
        coin_items: list[dict[str, Any]] = []
        uncalibrated_silver = False

        rel_px_per_mm = px_per_mm
        if rel_px_per_mm is None and filtered_coins:
            max_diam = max(2.0 * cc["radius"] for cc in filtered_coins)
            rel_px_per_mm = max_diam / 27.0
            scale_source = "relative_coin_estimate (no bill/₱20 scale reference)"
            uncalibrated_silver = True

        for cc in filtered_coins:
            c_res = _analyze_coin(img_wb, cc["cx"], cc["cy"], cc["radius"], rel_px_per_mm)
            denom_val = float(c_res["denomination"]) if c_res["denomination"] is not None else 1.0
            conf = float(c_res["confidence"])
            if uncalibrated_silver and not c_res.get("is_bimetallic_20"):
                conf = min(conf, 0.62)

            coin_items.append(
                {
                    "type": "coin",
                    "denomination": denom_val,
                    "label": c_res["label"],
                    "confidence": round(conf, 3),
                    "bbox": cc["bbox"],
                    "center": [round(cc["cx"], 1), round(cc["cy"], 1)],
                    "radius": round(cc["radius"], 1),
                    "est_mm": c_res.get("est_mm"),
                }
            )

        if uncalibrated_silver:
            warnings.append(
                "Only silver coins were detected with no banknote or ₱20 bimetallic coin for size calibration. "
                "Silver coin values (₱10/₱5/₱1/25¢) were estimated from relative diameter—please verify below."
            )

        all_items: list[dict[str, Any]] = []
        idx_counter = 1

        for item in bill_items:
            denom_int = int(item["denomination"])
            box_color = BILL_SPECS[denom_int]["box_bgr"]
            box = cv2.boxPoints(item["rect"])
            box_pts = np.int32(box)
            cv2.drawContours(annotated, [box_pts], 0, box_color, 3)

            tag = f"#{idx_counter} P{denom_int} ({int(item['confidence'] * 100)}%)"
            tx, ty = int(item["bbox"][0]), max(22, int(item["bbox"][1]) - 8)
            cv2.rectangle(annotated, (tx, ty - 20), (tx + len(tag) * 10 + 10, ty + 4), box_color, -1)
            cv2.putText(annotated, tag, (tx + 5, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2, cv2.LINE_AA)

            clean_item = {k: v for k, v in item.items() if k not in ("rect", "aspect")}
            clean_item["id"] = idx_counter
            all_items.append(clean_item)
            idx_counter += 1

        for item in coin_items:
            cx, cy = int(round(item["center"][0])), int(round(item["center"][1]))
            rad = int(round(item["radius"]))
            cv2.circle(annotated, (cx, cy), rad, (56, 189, 248), 3)
            val_lbl = f"{int(item['denomination'])}" if item["denomination"] >= 1 else f"{item['denomination']:.2f}"
            tag = f"#{idx_counter} P{val_lbl} ({int(item['confidence'] * 100)}%)"
            tx, ty = max(4, cx - rad), max(22, cy - rad - 8)
            cv2.rectangle(annotated, (tx, ty - 18), (tx + len(tag) * 9 + 8, ty + 4), (56, 189, 248), -1)
            cv2.putText(annotated, tag, (tx + 4, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 23, 42), 2, cv2.LINE_AA)

            clean_item = dict(item)
            clean_item["id"] = idx_counter
            all_items.append(clean_item)
            idx_counter += 1

        if not all_items:
            warnings.append(
                "No bills or coins stood out clearly from the background. "
                "Lay Philippine bills and coins flat and separated on a contrasting plain surface with even lighting."
            )

        total_php = round(sum(float(it["denomination"]) for it in all_items), 2)
        breakdown: dict[str, int] = {}
        for it in all_items:
            key = it["label"]
            breakdown[key] = breakdown.get(key, 0) + 1

        ok, enc = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        annotated_b64 = ""
        if ok:
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")

        processing_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "total_php": total_php,
            "formatted_total": f"₱{total_php:,.2f}",
            "bill_count": len(bill_items),
            "coin_count": len(coin_items),
            "item_count": len(all_items),
            "items": all_items,
            "breakdown": breakdown,
            "scale_source": scale_source,
            "warnings": warnings,
            "annotated_image_base64": annotated_b64,
            "processing_ms": processing_ms,
        }


def generate_sample_ph_money_image() -> bytes:
    """Generate a realistic sample tabletop image of Philippine bills & coins for testing/demo."""
    H, W = 740, 1080
    rng = np.random.default_rng(123)
    canvas = np.full((H, W, 3), (46, 42, 38), dtype=np.int16)
    noise = rng.integers(-6, 7, size=(H, W, 3), dtype=np.int16)
    canvas = np.clip(canvas + noise, 0, 255).astype(np.uint8)

    px_per_mm = 2.25

    def draw_bill(denom: int, cx: int, cy: int, angle_deg: float) -> None:
        bw, bh = int(160 * px_per_mm), int(66 * px_per_mm)
        bill = np.zeros((bh, bw, 3), dtype=np.uint8)
        base_bgr = BILL_SPECS[denom]["bgr"]
        bill[:, :] = base_bgr
        cv2.rectangle(bill, (6, 6), (bw - 7, bh - 7), tuple(max(0, c - 30) for c in base_bgr), 2)
        cv2.ellipse(bill, (int(bw * 0.32), int(bh * 0.52)), (36, 48), 0, 0, 360, tuple(min(255, c + 38) for c in base_bgr), -1)
        cv2.putText(bill, f"{denom}", (int(bw * 0.58), int(bh * 0.65)), cv2.FONT_HERSHEY_DUPLEX, 1.5, (250, 250, 250), 3, cv2.LINE_AA)
        cv2.putText(bill, "REPUBLIKA NG PILIPINAS", (22, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

        mask = np.full((bh, bw), 255, dtype=np.uint8)
        pad = 220
        big_b = cv2.copyMakeBorder(bill, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        big_m = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        mc = (big_b.shape[1] / 2.0, big_b.shape[0] / 2.0)
        M = cv2.getRotationMatrix2D(mc, angle_deg, 1.0)
        rot_b = cv2.warpAffine(big_b, M, (big_b.shape[1], big_b.shape[0]))
        rot_m = cv2.warpAffine(big_m, M, (big_m.shape[1], big_m.shape[0]))

        rh, rw = rot_b.shape[:2]
        x1, y1 = cx - rw // 2, cy - rh // 2
        x2, y2 = x1 + rw, y1 + rh
        roi = canvas[max(0, y1) : min(H, y2), max(0, x1) : min(W, x2)]
        rx1, ry1 = max(0, -x1), max(0, -y1)
        rx2, ry2 = rx1 + roi.shape[1], ry1 + roi.shape[0]
        m_crop = rot_m[ry1:ry2, rx1:rx2] > 128
        roi[m_crop] = rot_b[ry1:ry2, rx1:rx2][m_crop]

    def draw_coin(denom: float, mm: float, cx: int, cy: int, bimetallic: bool = False) -> None:
        r = int(round((mm * px_per_mm) / 2.0))
        if bimetallic:
            cv2.circle(canvas, (cx, cy), r, (95, 148, 205), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), int(r * 0.64), (196, 198, 202), -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (cx, cy), r, (192, 195, 200), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), int(r * 0.85), (175, 178, 184), 2, cv2.LINE_AA)
        lbl = f"{int(denom)}" if denom >= 1 else ".25"
        cv2.putText(canvas, lbl, (cx - r // 2 + 4, cy + 7), cv2.FONT_HERSHEY_DUPLEX, 0.65, (85, 88, 92), 2, cv2.LINE_AA)

    draw_bill(1000, 275, 180, -6.0)
    draw_bill(500, 720, 195, 7.0)
    draw_bill(100, 290, 445, 4.0)
    draw_bill(50, 745, 455, -5.0)
    draw_coin(20, 30.0, 230, 640, bimetallic=True)
    draw_coin(10, 27.0, 415, 640, bimetallic=False)
    draw_coin(5, 25.0, 590, 640, bimetallic=False)
    draw_coin(1, 23.0, 755, 640, bimetallic=False)

    ok, enc = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return enc.tobytes() if ok else b""
