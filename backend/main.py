"""
Mamba NIDS — FastAPI Backend

Endpoints:
    POST /predict           — Classify a traffic sequence
    GET  /blocked-ips       — List all blocked IPs
    DELETE /blocked-ips/{ip} — Unblock an IP
    WS   /ws/live-traffic   — WebSocket for live traffic simulation
"""
import os
import sys
import json
import random
import asyncio
from datetime import datetime

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent directory to path so we can import the model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.mamba_nids import MambaNIDS

# ──────────────────────────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mamba NIDS API",
    description="Real-Time Cyber Attack Detection using Mamba (State Space Models)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend")

# ──────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Attempt to load trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "model", "checkpoints", "mamba_nids_complete.pth")

ATTACK_NAMES = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
                "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms"]

model = None
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    hp = checkpoint["hyperparams"]
    model = MambaNIDS(
        input_dim=hp["input_dim"], d_model=hp["d_model"], d_state=hp["d_state"],
        n_layers=hp["n_layers"], num_classes=hp["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    ATTACK_NAMES = checkpoint.get("attack_encoder_classes", ATTACK_NAMES)
    print(f"Model loaded from {MODEL_PATH} on {device}")
else:
    print(f"WARNING: No model checkpoint found at {MODEL_PATH}")
    print("Running in demo mode with random predictions.")

# ──────────────────────────────────────────────────────────────────
# Blocked IPs
# ──────────────────────────────────────────────────────────────────
blocked_ips: set[str] = set()

# ──────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────
class TrafficSequence(BaseModel):
    features: list[list[float]]
    source_ip: str

# ──────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
    return {"message": "Mamba NIDS API is running", "docs": "/docs"}


@app.post("/predict")
async def predict(data: TrafficSequence):
    """Classify a traffic sequence as normal or a specific attack type."""
    if model is None:
        # Demo mode: random prediction
        pred_class = random.randint(0, len(ATTACK_NAMES) - 1)
        confidence = random.uniform(0.5, 0.99)
    else:
        x = torch.tensor([data.features], dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            pred_class = probs.argmax(dim=-1).item()
            confidence = probs[0, pred_class].item()

    attack_name = ATTACK_NAMES[pred_class]
    is_attack = attack_name != "Normal"

    if is_attack and confidence > 0.8:
        blocked_ips.add(data.source_ip)

    return {
        "prediction": attack_name,
        "is_attack": is_attack,
        "confidence": round(confidence, 4),
        "source_ip": data.source_ip,
        "blocked": data.source_ip in blocked_ips,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/blocked-ips")
async def get_blocked_ips():
    return {"blocked_ips": sorted(blocked_ips), "count": len(blocked_ips)}


@app.delete("/blocked-ips/{ip}")
async def unblock_ip(ip: str):
    blocked_ips.discard(ip)
    return {"message": f"Unblocked {ip}", "blocked_ips": sorted(blocked_ips)}


@app.websocket("/ws/live-traffic")
async def live_traffic_websocket(websocket: WebSocket):
    """WebSocket endpoint that simulates live network traffic."""
    await websocket.accept()
    print("WebSocket client connected")

    try:
        while True:
            is_attack = random.random() < 0.15
            if is_attack:
                attack_type = random.choice(["DoS", "Backdoor", "Exploits", "Generic", "Fuzzers"])
                source_ip = f"192.168.{random.randint(1,255)}.{random.randint(1,255)}"
            else:
                attack_type = "Normal"
                source_ip = f"10.0.{random.randint(1,10)}.{random.randint(1,255)}"

            features = np.random.randn(32, 42).astype(np.float32).tolist()

            if model is not None:
                x = torch.tensor([features], dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=-1)
                    pred_class = probs.argmax(dim=-1).item()
                    confidence = probs[0, pred_class].item()
                prediction = ATTACK_NAMES[pred_class]
            else:
                prediction = attack_type
                confidence = random.uniform(0.6, 0.99)

            is_pred_attack = prediction != "Normal"
            if is_pred_attack and confidence > 0.8:
                blocked_ips.add(source_ip)

            event = {
                "timestamp": datetime.now().isoformat(),
                "source_ip": source_ip,
                "prediction": prediction,
                "is_attack": is_pred_attack,
                "confidence": round(confidence, 4),
                "blocked": source_ip in blocked_ips,
                "blocked_ips_count": len(blocked_ips),
            }

            await websocket.send_json(event)
            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        print("WebSocket client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
