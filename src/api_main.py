# file: api_main.py

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from config import Config
from core import run_pipeline


app = FastAPI(
    title="Pipeline API",
    description="API to run pipelines with JSON configurations",
    version="1.0.0",
)


class PipelineRequest(BaseModel):
    symbols: List[str]  # List of symbols is required

    class Config:
        json_schema_extra = {
            "example": {"symbols": ["AAPL", "MSFT", "TSLA", "SPY", "TLT", "GLD"]}
        }


@app.get("/")
def redirect_to_docs():
    """Redirect root to Swagger UI."""
    return RedirectResponse(url="/docs")


@app.post("/inference")
def inference(req: PipelineRequest):
    """
    Run the pipeline without changing the loaded config object.
    Only pass `symbols_override` to the pipeline.
    """
    # Load the default configuration
    default_config_path = "config.yaml"
    try:
        config_obj = Config.from_yaml(default_config_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Default configuration file '{default_config_path}' not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading default configuration: {str(e)}"
        )

    try:
        result = run_pipeline(
            config=config_obj, symbols_override=req.symbols, run_local=False
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Pipeline execution failed: {str(e)}"
        )

    return {"status": "success", "result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
