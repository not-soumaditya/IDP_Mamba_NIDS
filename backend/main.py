"""
FastAPI backend for NIDS-Mamba.
Provides endpoints for health checks and real-time traffic prediction.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="NIDS-Mamba API",
    description="Real-Time Cyber Attack Detection using Mamba SSM",
    version="0.1.0",
)

class TrafficSequence(BaseModel):
    """Schema for a sequence of network traffic features."""
    features: List[List[float]]  # Expected shape: [seq_len, num_features]

@app.get("/health")
def health_check():
    """Verify the API is running."""
    return {"status": "ok", "message": "NIDS-Mamba API is running."}

@app.post("/predict")
def predict_traffic(sequence: TrafficSequence):
    """
    Accepts a sequence of network features and returns a prediction.
    Note: Currently a stub for Review 2. Will connect to trained Mamba model in Phase 3.
    """
    # Placeholder response to demonstrate API contract
    return {
        "status": "success",
        "prediction": "Normal",
        "confidence": 0.98,
        "action": "allow"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
