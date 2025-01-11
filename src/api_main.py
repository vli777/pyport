# file: api_main.py

from pathlib import Path
from fastapi import FastAPI
from core import run_pipeline

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from config import Config

app = FastAPI()


class PipelineRequest(BaseModel):
    symbols: Optional[List[str]] = None  # E.g., ["AAPL", "MSFT", "TSLA"]
    config_file: Optional[str] = None


@app.post("/inference")
def inference(req: PipelineRequest):
    # Use default local config.yaml if config_file is not provided
    config_path = req.config_file or "config.yaml"

    # Load configuration from the file
    if not Path(config_path).exists():
        return {"error": f"Config file '{config_path}' not found"}

    config_obj = Config.from_yaml(config_path)

    # Override symbols if provided
    symbols_override = req.symbols if req.symbols else None

    # Run the pipeline with the configuration
    return run_pipeline(config_obj, symbols_override=symbols_override, run_local=False)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
