# PyPort
Modern Portfolio Optimization 

# Latest Changes (January 2025)
- Reworked implementation of Nested Clustering with vector ops => much faster!
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3469961
- WIP: additional models to feature deep learning methods i.e. RL, transformers

Daily and Cumulative Returns Statistics
![image](https://github.com/user-attachments/assets/10fd55c7-728f-454c-a408-95a8fd17f6d7)

Anomaly Detection
![image](https://github.com/user-attachments/assets/0cc9c310-fbef-42b6-834f-38a17b8edc2b)

Hierarchical Clustering Dendrogram
![image](https://github.com/user-attachments/assets/c3136d02-b4af-4072-a05b-a04667e2f49e)

Dynamic Z-score Thresholds for Mean Reversion
![image](https://github.com/user-attachments/assets/58b510e8-ccf6-4cba-8cf6-c4790f8c4aab)

## Instructions

### Setting Up Configuration
- Create a `config.yaml` file based on the provided `testconfig.yaml` template.
- In the `config.yaml` file, you can specify:
  - Input files containing ticker symbols.
  - Models and time periods to optimize against.

### Preparing Input Files
- Each input file (CSV format) should have a single column containing the ticker symbols to include in the optimization.
- Any commented lines (starting with `#`) will be ignored during processing.

### Configuring Input Files in `config.yaml`
- Under the `input_files` section of the config, list multiple files if needed. For example:
  ```yaml
  input_files:
    - file1.csv
    - file2.csv

You can specify multiple models under models. The output will contain a simple average of all selected models.

# Local Usage

This project is now compatible with LTS Python version as of January 2025 (3.12.8) and can be installed simply with 
```
pip install -r requirements.txt
```
There is a CLI main runner, API main available, as well as a new iterative_pipeline for entrypoints.

# Running the API

The API allows you to execute the pipeline dynamically by providing configurations and symbol overrides.

## Starting the API

Run the following command to start the API server:

```
python api_main.py
```

The server will start on `http://localhost:8000`.

## API Endpoint

**POST** `/inference`

- **Description**: Runs the portfolio optimization pipeline.
- **Payload**: json { "symbols": ["AAPL", "MSFT", "TSLA", "SPY", "TLT", "GLD"] }

- **Example Request**:

```
curl -X POST "http://localhost:8000/inference" \
-H "Content-Type: application/json" \
-d '{
  "symbols": ["AAPL", "MSFT", "TSLA", "SPY", "TLT", "GLD"],
}'
```

- **Response**: A JSON object containing the results of the pipeline:

```json
{ "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "models": "model_name_1, model_name_2", "symbols": ["symbol1", "symbol2"], "normalized_avg": { "symbol1": 0.25, "symbol2": 0.75 } }
```

---

# Notes

- Ensure your `config.yaml` file is correctly configured for your use case.
- The API allows dynamic overrides for symbols and configurations without modifying the local `config.yaml`.
- The symbol selection process will auto-remove highly correlated pairs within a group and select representatives based on sharpe ratio vs the group
- SIM_PORT is the hypothetical portfolio with the allocation result over the same period (used for graphing daily and cumulative returns).

