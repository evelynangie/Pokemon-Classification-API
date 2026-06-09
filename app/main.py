import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from app.predictor import PokemonPredictor

app = FastAPI(title="Pokemon Classification API")

model_path = os.path.join(base_dir, "models", "model.pkl")
predictor = PokemonPredictor(model_path=model_path)

class PredictRequest(BaseModel):
    image_base64: str

@app.get("/")
def read_root():
    return {
        "status": "success",
        "message": "Pokemon Classification API beroperasi. Gunakan POST /predict."
    }

@app.post("/predict")
def predict_endpoint(payload: PredictRequest):
    try:
        result = predictor.predict(payload.image_base64)
        return {
            "status": "success",
            "data": result
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)