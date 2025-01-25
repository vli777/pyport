# file: api_main.py

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from config import Config
from iterative_pipeline import iterative_pipeline_runner


app = FastAPI(
    title="Pipeline API",
    description="API to run pipelines with JSON configurations",
    version="1.0.0",
)


class PipelineRequest(BaseModel):
    symbols: Optional[List[str]] = None
    max_epochs: Optional[int] = None
    min_weight: Optional[float] = None
    portfolio_max_size: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "symbols": ["AAPL", "MSFT", "TSLA", "SPY", "TLT", "GLD"],
                "max_epochs": 15,
                "min_weight": 0.02,
                "portfolio_max_size": 20,
            }
        }


@app.get("/")
def redirect_to_docs():
    """Redirect root to Swagger UI."""
    return RedirectResponse(url="/docs")


@app.post("/inference")
def inference(req: PipelineRequest):
    """
    Run the pipeline, optionally overriding config parameters.
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
        result = iterative_pipeline_runner(
            config=config_obj,
            initial_symbols=req.symbols,
            max_epochs=req.max_epochs,
            min_weight=req.min_weight,
            portfolio_max_size=req.portfolio_max_size,
            run_local=False,
        )
    except TypeError as te:
        raise HTTPException(
            status_code=400, detail=f"Invalid parameter type: {str(te)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Pipeline execution failed: {str(e)}"
        )

    return {"status": "success", "result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)