# file: api_main.py

from fastapi import FastAPI
from core import run_pipeline

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI()

class TickersRequest(BaseModel):
    symbols: List[str]      # E.g., ["AAPL", "MSFT", "TSLA"]
    config_file: str = "config.yaml"

@app.post("/inference")
def inference(req: TickersRequest):
    """
    POST JSON like:
    {
      "watchlists": ["temp2"],
      "config_file": "config.yaml"  # or omit for default
    }
    """
    # We call the pipeline with run_local=False 
    # so we skip printing/plotting in a server context
    final_json = run_pipeline(
        config_file=req.config_file,
        symbols_override=req.symbols,
        run_local=False
    )
    return final_json

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)