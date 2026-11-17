"""Turns the light that reached the sensor into the photo a phone would save.

The render is clean and linear. A phone's photo is not: the sensor adds noise that grows as the light falls, the
hand shakes during a long exposure, and the phone's processing smooths the noise, lifts the shadows, sharpens the
edges and compresses the file. Nothing here moves anything, so the truth's positions stay right.
README.md, "The camera".
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

# Light levels: how many electrons a white page fills a pixel with, and how often the level is drawn. A bright room
# fills the well; a dark one leaves a few hundred, and the photo is mostly noise.
LEVELS = {"bright": ((12000, 40000), 40), "indoor": ((1500, 9000), 35), "dim": ((300, 1500), 17), "dark": ((50, 300), 8)}


def draw_settings(rng: random.Random) -> dict:
    """How the photo is taken and processed, at random."""
    level = rng.choices(list(LEVELS), weights=[weight for _, weight in LEVELS.values()])[0]
    low, high = LEVELS[level][0]
    electrons = math.exp(rng.uniform(math.log(low), math.log(high)))
    shaky = {"bright": 0.03, "indoor": 0.1, "dim": 0.3, "dark": 0.5}[level]
    return {
        "light": level,
        "electrons": round(electrons),
        "exposure_ev": round(rng.gauss(0, 0.35), 2),                       # how far the phone's exposure misses
        "white_balance": round(rng.uniform(0.55, 1.0), 2),                  # how much of the light's colour the phone removes
        "tint": [round(rng.gauss(0, 0.02), 3) for _ in range(2)],
        "vignette": round(rng.uniform(0.0, 0.3), 2),
        "shake_px": round(rng.uniform(0.8, 5.0), 1) if rng.random() < shaky else 0.0,   # at a picture 2048 px high
        "shake_deg": round(rng.uniform(0, 180), 1),
        "defocus_px": round(rng.uniform(0.8, 3.0), 1) if rng.random() < 0.08 else 0.0,
        "smoothing": round(rng.uniform(0.5, 1.0), 2),                       # how hard the phone cleans noise away
        "shadow_lift": round(rng.uniform(0.0, 0.45), 2),
        "sharpen": round(rng.uniform(0.2, 1.3), 2),
        "saturation": round(rng.uniform(0.9, 1.25), 2),
        "jpeg_quality": rng.choice([60, 70, 80, 85, 90, 90, 95]),
    }


def develop(light: np.ndarray, settings: dict, seed: int) -> np.ndarray:
    """The photo: height by width by 3, 8 bits, as a screen shows it."""
    rng = np.random.default_rng(seed)
    height, width = light.shape[:2]
    scale = height / 2048
    picture = np.maximum(light.astype(np.float32), 0)

    # Exposure: the phone aims to put the brightest paper just under white, and misses a little. Judged, like the
    # white balance, from one pixel in 64: enough to measure by, and a twelfth of the time at 12 megapixels.
    sample = picture[::8, ::8]
    levels = sample.mean(axis=2)
    white = float(np.percentile(levels, 97)) or 1.0
    picture *= 0.8 * 2 ** settings["exposure_ev"] / white

    # White balance: the phone takes out most of the light's colour, judged from the bright parts of the picture.
    bright = sample[levels >= np.percentile(levels, 70)] * (0.8 * 2 ** settings["exposure_ev"] / white)
    cast = bright.mean(axis=0) / max(float(bright.mean()), 1e-6)
    picture /= (cast ** settings["white_balance"]).astype(np.float32)
    picture *= np.array([1 + settings["tint"][0], 1.0, 1 + settings["tint"][1]], np.float32)

    if settings["vignette"]:
        across = ((np.arange(width, dtype=np.float32) - width / 2) ** 2)[None, :]
        down = ((np.arange(height, dtype=np.float32) - height / 2) ** 2)[:, None]
        picture *= (1 - settings["vignette"] * (across + down) / ((width / 2) ** 2 + (height / 2) ** 2))[..., None]
    if settings["defocus_px"]:
        picture = cv2.GaussianBlur(picture, (0, 0), settings["defocus_px"] * scale)
    if settings["shake_px"]:
        # The hand moves during the exposure: every point is smeared along a short line.
        length = max(2, round(settings["shake_px"] * scale))
        kernel = np.zeros((length * 2 + 1, length * 2 + 1), np.float32)
        angle = math.radians(settings["shake_deg"])
        cv2.line(kernel, (round(length - length * math.cos(angle)), round(length - length * math.sin(angle))), (round(length + length * math.cos(angle)), round(length + length * math.sin(angle))), 1.0, 1)
        picture = cv2.filter2D(picture, -1, kernel / kernel.sum())

    # The sensor: light arrives as counted electrons, and counting is noisy. Fewer electrons, more noise.
    electrons = settings["electrons"]
    noise = rng.standard_normal(picture.shape, dtype=np.float32) * cv2.sqrt(np.maximum(picture, 0) / electrons + (3.0 / electrons) ** 2)
    noise[..., 0] *= 1.25   # red and blue are amplified more than green
    noise[..., 2] *= 1.45
    picture = picture + cv2.GaussianBlur(noise, (0, 0), 0.55)   # the sensor's colour mosaic spreads each pixel's noise to its neighbours

    # The phone's processing. In poor light it smooths hard, and fine detail goes with the noise.
    grain = math.sqrt(0.5 / electrons)
    if grain > 0.006:
        strength = settings["smoothing"] * min(1.0, grain / 0.03)
        # Smoothed on a perceptual scale. The scale keeps the sign of noise that dipped below black, so smoothing
        # averages it back to black and does not turn dark ink grey.
        encoded = np.sign(picture) * cv2.pow(np.abs(np.clip(picture, -4, 4)), 1 / 2.2)
        smoothed = cv2.bilateralFilter(encoded, d=9, sigmaColor=0.04 + 0.35 * strength * min(1.0, grain * 12), sigmaSpace=1.0 + 2.5 * strength * scale)
        mixed = encoded * (1 - strength) + smoothed * strength
        picture = np.sign(mixed) * cv2.pow(np.abs(mixed), 2.2)
    if settings["shadow_lift"]:
        # Shadows are lifted toward the rest of the picture, as a phone's HDR does.
        # The brightness of each neighbourhood, found on a picture an eighth the size: it is smooth, so nothing is lost.
        small = cv2.resize(picture, (width // 8, height // 8), interpolation=cv2.INTER_AREA).mean(axis=2)
        brightness = cv2.resize(cv2.GaussianBlur(small, (0, 0), 5 * scale), (width, height), interpolation=cv2.INTER_LINEAR)
        picture /= cv2.pow(np.maximum(brightness, 0.02) / 0.5, settings["shadow_lift"])[..., None]
    picture = np.clip(picture, 0, 1)
    encoded = np.where(picture <= 0.0031308, picture * 12.92, 1.055 * cv2.pow(picture, 1 / 2.4) - 0.055)
    grey = encoded.mean(axis=2, keepdims=True)
    encoded = grey + (encoded - grey) * settings["saturation"]
    if settings["sharpen"]:
        encoded = encoded + settings["sharpen"] * (encoded - cv2.GaussianBlur(encoded, (0, 0), 1.2 * max(scale, 0.5)))
    return (np.clip(encoded, 0, 1) * 255 + 0.5).astype(np.uint8)


def jpeg(photo: np.ndarray, quality: int) -> bytes:
    done, data = cv2.imencode(".jpg", cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert done
    return data.tobytes()
