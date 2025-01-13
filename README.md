# PyPort
Portfolio Optimization

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

![image](https://github.com/user-attachments/assets/5b0b3ef7-a1cd-4fb5-962c-48a7b8de5fc6)

---

# Notes

- Ensure your `config.yaml` file is correctly configured for your use case.
- The API allows dynamic overrides for symbols and configurations without modifying the local `config.yaml`.


