import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.templating import Jinja2Templates
from crypto_service import heavy_encryption_task, decrypt_from_db
from fastapi.responses import FileResponse

app = FastAPI(title="Medical Image Encryption System")

templates = Jinja2Templates(directory="templates")

# =========================
# 🏠 HOME PAGE
# =========================
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# =========================
# 🔒 ENCRYPT API
# =========================
@app.post("/encrypt")
async def encrypt_image_api(
    fingerprint: UploadFile = File(...),
    image: UploadFile = File(...),
    img_type: str = Form(...)
):
    fingerprint_bytes = await fingerprint.read()
    image_bytes = await image.read()

    result = heavy_encryption_task(
        fingerprint_bytes,
        image_bytes,
        img_type
    )

    return {
        "status": "success",
        "file_name": result["file_name"],
        "encryption_time": result["encryption_time"],
        "entropy": result["entropy"],
        "npcr": result["npcr"],
        "uaci": result["uaci"],
        "correlation": result["correlation"],
        "combined_histogram": result["combined_histogram"]
    }
# =========================
# 🔓 DECRYPT API
# =========================
@app.post("/decrypt")
async def decrypt_api(
    fingerprint: UploadFile = File(...),
    img_name: str = Form(...)
):
    fingerprint_bytes = await fingerprint.read()

    result = decrypt_from_db(fingerprint_bytes, img_name)

    if result is None:
        return {"status": "fail", "message": "Unauthorized"}

    return {
        "status": "success",
        "decryption_time": result["decryption_time"],
        "decrypted_path": result["path"],
        "mse": result["mse"],
        "psnr": result["psnr"]
}