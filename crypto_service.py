import numpy as np
import cv2
import hashlib
import time
from pathlib import Path
from pymongo import MongoClient
from datetime import datetime
from math import log2
import matplotlib.pyplot as plt
from ga_service import optimize_hyperchaotic_params

# ===============================
# DATABASE CONNECTION
# ===============================
client = MongoClient("mongodb://localhost:27017/")
db = client["medical_security"]
collection = db["encrypted_images"]

# ===============================
# HYPERCHAOTIC MODEL
# ===============================
def hyper_iter(x, y, z, w, params):
    a, b, c, d = params

    dx = a*(y - x) + b*z
    dy = (c - a)*x - x*z + d*y
    dz = x*y - b*z + np.sin(w)
    dw = -c*w + 0.1*np.cos(x)

    x += dx * 0.01
    y += dy * 0.01
    z += dz * 0.01
    w += dw * 0.01
    x, y, z, w = np.clip([x, y, z, w], -50, 50)

    return x, y, z, w

def generate_keystream(params, init_state, length):
    x, y, z, w = init_state
    ks = np.zeros(length, dtype=np.uint8)

    for i in range(length):
        x, y, z, w = hyper_iter(x, y, z, w, params)

        val = abs(x) + 1.7*abs(y) + 0.9*abs(z) + 0.5*abs(w)

        if not np.isfinite(val):
            val = 0.5

        val = (np.tanh(val) + 1) / 2
        ks[i] = int(val * 255) & 0xFF

    return ks

# ===============================
# DIFFUSION
# ===============================
def strong_diffusion(image, key_stream):
    h, w = image.shape
    flat = image.flatten()
    key = key_stream.flatten()

    # Forward
    for i in range(1, flat.size):
        flat[i] ^= key[i] ^ flat[i-1]

    # Backward
    for i in range(flat.size - 2, -1, -1):
        flat[i] ^= key[i] ^ flat[i+1]

    # Bit rotation
    for i in range(flat.size):
        r = key[i] % 8
        flat[i] = ((flat[i] << r) | (flat[i] >> (8 - r))) & 0xFF

    return flat.reshape(h, w)

def encrypt_pipeline(img, params, init_state, key):

    h, w = img.shape
    total = h * w

    # ---- plaintext dependency ----
    init_state2 = init_state

    # ---- keystream ----
    ks = generate_keystream(params, init_state2, total)

    # ---- permutation ----
    seed = int.from_bytes(
        hashlib.sha256(key + img.tobytes()).digest()[:4],
        "big"
    )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)

    shuffled = img.flatten()[perm]

    key_stream = np.frombuffer(key, np.uint8)
    key_stream = np.resize(key_stream, total)

    # ---- XOR ----
    cipher = np.bitwise_xor(shuffled, ks)
    cipher = np.bitwise_xor(cipher, key_stream)

    enc = cipher.reshape(h, w)

    # ---- diffusion ----
    enc = strong_diffusion(enc, key_stream.reshape(h, w))

    return enc

def inverse_diffusion(image, key_stream):
    h, w = image.shape
    flat = image.flatten()
    key = key_stream.flatten()

    # Reverse rotation
    for i in range(flat.size):
        r = key[i] % 8
        flat[i] = ((flat[i] >> r) | (flat[i] << (8 - r))) & 0xFF

    # Reverse backward
    for i in range(flat.size - 1):
        flat[i] ^= key[i] ^ flat[i+1]

    # Reverse forward
    for i in range(flat.size - 1, 0, -1):
        flat[i] ^= key[i] ^ flat[i-1]

    return flat.reshape(h, w)

# ===============================
# 🔒 ENCRYPTION
# ===============================
def heavy_encryption_task(fingerprint_bytes, image_bytes, img_type):

    print("ENCRYPTION START")
    start_time = time.time()

    # ---- fingerprint → key ----
    f = np.frombuffer(fingerprint_bytes, np.uint8)
    f = cv2.imdecode(f, 0)
    f = cv2.resize(f, (512, 512))

    flat = f.flatten() / 255.0
    key = hashlib.sha512(flat.tobytes()).digest()

    # ---- GA ----
    # Generate seed ONLY from fingerprint key
    seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    ga = optimize_hyperchaotic_params(seed)
    params = ga["params"]
    init_state = ga["init_state"]

    # ---- image ----
    img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img, 0)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)

    h, w = img.shape
    total = h * w

    # ---- plaintext dependent ----
    plain_hash = hashlib.sha256(img.tobytes()).digest()

    x0, y0, z0, w0 = init_state

    x0 = (x0 + plain_hash[0]/255) % 1
    y0 = (y0 + plain_hash[1]/255) % 1
    z0 = (z0 + plain_hash[2]/255) % 1
    w0 = (w0 + plain_hash[3]/255) % 1

    init_state = (x0, y0, z0, w0)

    # ---- keystream ----
    ks = generate_keystream(params, init_state, total)

    # ---- permutation seed from keystream ----
    seed = int(np.sum(ks) % (2**32))

    # ---- permutation ----
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    shuffled = img.flatten()[perm]

    # ---- fingerprint key ----
    key_stream = np.frombuffer(key, np.uint8)
    key_stream = np.resize(key_stream, total)

    # ---- XOR ----
    cipher = np.bitwise_xor(shuffled, ks)
    cipher = np.bitwise_xor(cipher, key_stream)

    enc = cipher.reshape(h, w)

    # ---- diffusion ----
    enc = strong_diffusion(enc, key_stream.reshape(h, w))
    # ---- metrics ----
    ent = entropy(enc)
    npcr, uaci = npcr_uaci(img, enc)
    combined_hist_path = f"output/combined_hist_{int(time.time())}.png"
    # Plain image correlation
    plain_corr = correlation_analysis(img)
    # Encrypted image correlation
    enc_corr = correlation_analysis(enc)

    # ---- Histogram Figure ----
    plt.figure(figsize=(12, 5))

    # Plain Histogram
    plt.subplot(1, 2, 1)
    plt.hist(img.flatten(), bins=256)
    plt.title("Plain Image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    # Encrypted Histogram
    plt.subplot(1, 2, 2)
    plt.hist(enc.flatten(), bins=256)
    plt.title("Encrypted Image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    # Proper spacing
    plt.tight_layout()

    # Save single image
    plt.savefig(combined_hist_path)
    plt.close()

    # ---- save ----
    out = Path("output")
    out.mkdir(exist_ok=True)
    name = f"{img_type}_{int(time.time())}.png"
    path = out / name

    cv2.imwrite(str(path), enc)

    # ---- DB ----
    collection.insert_one({
        "img_name": name,
        "fingerprint_hash": key.hex(),
        "encrypted_image": enc.tobytes(),
        "original_image": img.tobytes(),
        "shape": enc.shape,
        "params": params.tolist(),
        "init_state": list(init_state),
        "plain_hash": hashlib.sha256(img.tobytes()).hexdigest(),
        "perm_seed": int(seed),
        "timestamp": datetime.now()
    })
    print("Encrypted image stored into MongoDB successfully")
    end_time = time.time()
    encryption_time = end_time - start_time
    print(f"Encryption Time: {encryption_time:.4f} seconds")
    print("PERFORMANCE METRICS")
    print("🔹 Correlation Coefficients")
    print("Plain Image:", plain_corr)
    print("Encrypted Image:", enc_corr)
    print(f"🔹Information Entropy: {ent:.4f}")
    print("🔹Differential Analysis")
    print(f"NPCR: {npcr:.2f}%")
    print(f"UACI: {uaci:.2f}%")

    return {
        "file_name": name,
        "path": str(path),
        "encryption_time": round(encryption_time, 4),
        "entropy": round(ent, 4),
        "npcr": round(npcr, 2),
        "uaci": round(uaci, 2),
        "correlation": {
        "plain": plain_corr,
        "encrypted": enc_corr
        },
        "combined_histogram": combined_hist_path
    }
# ===============================
# 🔓 DECRYPTION
# ===============================
def decrypt_image(enc_img, params, key, init_state, seed):
    print("DECRYPTION START")

    h, w = enc_img.shape
    total = h * w

    ks = generate_keystream(params, init_state, total)

    # use SAME permutation logic as encryption
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    key_stream = np.frombuffer(key, np.uint8)
    key_stream = np.resize(key_stream, total)

    # inverse diffusion
    enc_img = inverse_diffusion(enc_img, key_stream.reshape(h, w))

    flat = enc_img.flatten()

    # reverse XOR
    unxor = np.bitwise_xor(flat, key_stream)
    unxor = np.bitwise_xor(unxor, ks)

    # reverse permutation
    original = np.zeros_like(unxor)
    original[perm] = unxor

    return original.reshape(h, w)

# ===============================
# 🔓 DECRYPT
# ===============================
def decrypt_from_db(fingerprint_bytes, img_name):
    start_time = time.time()

    f = np.frombuffer(fingerprint_bytes, np.uint8)
    f = cv2.imdecode(f, 0)
    f = cv2.resize(f, (512, 512))

    flat = f.flatten() / 255.0
    key = hashlib.sha512(flat.tobytes()).digest()

    record = collection.find_one({
        "fingerprint_hash": key.hex(),
        "img_name": img_name
        
    })
    if record is None:
        return None
    
    seed = record["perm_seed"]
    params = np.array(record["params"])
    init_state = tuple(record["init_state"])
    shape = record["shape"]

    enc_img = np.frombuffer(record["encrypted_image"], dtype=np.uint8).reshape(shape)

    dec = decrypt_image(enc_img, params, key, init_state, seed)

    orig = np.frombuffer(record["original_image"], dtype=np.uint8).reshape(shape)

    mse_val = mse(orig, dec)
    psnr_val = psnr(orig, dec)
    
    diff = np.sum(orig != dec)

    print("MSE:", mse_val)
    print("PSNR:", psnr_val)
    
    out_path = Path("output/decrypted.png")
    cv2.imwrite(str(out_path), dec)
    
    end_time = time.time()
    decryption_time = end_time - start_time
    print(f"Decryption Time: {decryption_time:.4f} seconds")

    return {
        "path": str(out_path),
        "decryption_time": round(decryption_time, 4),
        "mse": round(mse_val, 6),
        "psnr": "inf" if np.isinf(psnr_val) else round(float(psnr_val), 4)
    }

# ---------- PERFORMANCE METRICS ----------
def mse(a, b):
    return np.mean((a.astype(float) - b.astype(float)) ** 2)

def psnr(a, b):
    m = mse(a, b)
    if m == 0:
        return float('inf')   # keep true math
    return 20 * np.log10(255.0 / np.sqrt(m))
import numpy as np

def correlation_analysis(img):
    img = img.astype(np.float64)

    # Horizontal
    x_h = img[:, :-1].flatten()
    y_h = img[:, 1:].flatten()
    h_corr = np.corrcoef(x_h, y_h)[0, 1]

    # Vertical
    x_v = img[:-1, :].flatten()
    y_v = img[1:, :].flatten()
    v_corr = np.corrcoef(x_v, y_v)[0, 1]

    # Diagonal
    x_d = img[:-1, :-1].flatten()
    y_d = img[1:, 1:].flatten()
    d_corr = np.corrcoef(x_d, y_d)[0, 1]

    return {
        "Horizontal": float(h_corr),
        "Vertical": float(v_corr),
        "Diagonal": float(d_corr)
    }
def entropy(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    hist = hist / np.sum(hist)
    return -np.sum([p * log2(p) for p in hist if p != 0])
def npcr_uaci(a, b):
    diff = a != b
    npcr = np.sum(diff) * 100 / diff.size
    uaci = np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)) / 255.0) * 100
    return npcr, uaci

def histogram_data(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    return hist.tolist()
    

