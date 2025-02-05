# PyPort
Modern Portfolio Optimization 

# WIP
- Statistical arbitrage on cointegrated baskets

# February 2025 Latest (2/5/25)
- Updated features with more modern techniques: (VAE for anomaly detection would be considered more SOTA, but due to our data sample size and the marginal benefit vs computational resources required, isolation forest was selected.
- Changed the data processing to use the full available history for various computations vs slicing to the latest data range available among newer assets first.
- Fixed mean reversion hyperparameters optimization study results and continuous vs discrete signal generation for weight adjustment vs binary exclusion/inclusion
- Balanced precision for computational speed by using clustering to apply group optimal params vs for each individual asset for fast results
![image](https://github.com/user-attachments/assets/7bcefd44-e20a-41f3-8ac8-80dd8fd3c16a)
- Overall the results are very similar to the previous version which is a good sign for consistency!

- [x] Anomaly detection with Isolation Forest
![image](https://github.com/user-attachments/assets/cbbbad01-8886-4516-ba50-f88bab486ba4)
![image](https://github.com/user-attachments/assets/8dbf7f0b-8408-469b-9e29-b41923e7f330)

- [x] DBSCAN clustering for de-correlation
![image](https://github.com/user-attachments/assets/77242ffc-f2e9-44da-9218-db7f8346897a)
![image](https://github.com/user-attachments/assets/f27be7ef-85c1-460f-9928-63a17f3ea36d)

- [x] Robust Z-score Mean Reversion
![image](https://github.com/user-attachments/assets/1ab4d752-e68f-4935-aaad-5965ffe39b13)


# January 2025
- Reworked implementation of Nested Clustering (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3469961) with vector ops => much faster!
- Added anomaly detection with Kalman Filter and mean reversion filters
- Integrated Optuna to dynamically find optimal thresholds for maximizing performance metrics and returns 

Daily and Cumulative Returns Statistics
![image](https://github.com/user-attachments/assets/10fd55c7-728f-454c-a408-95a8fd17f6d7)

Anomaly Detection for Automatic Filtering
![image](https://github.com/user-attachments/assets/f8048fe3-4d92-4424-8d40-68490d47374d)

Hierarchical Clustering for De-correlation
![image](https://github.com/user-attachments/assets/ec4b9b05-e32e-4012-a49c-4446ef0ce603)

Dynamic Z-score Thresholds for Mean Reversion
![image](https://github.com/user-attachments/assets/58b510e8-ccf6-4cba-8cf6-c4790f8c4aab)

All Automated Hyperparameter Tuning
![image](https://github.com/user-attachments/assets/da12462f-3f0c-4b84-beac-59b96a6702b2)

![image](https://github.com/user-attachments/assets/78663eff-19cb-42c6-9fe7-713e8e143a2a)

![image](https://github.com/user-attachments/assets/521a32ef-78a5-4c99-bb51-6db81664eed3)


## Instructions

### Setting Up Configuration
- Create a `config.yaml` file based on the provided `testconfig.yaml` template.
- In the `config.yaml` file, you can specify:
  - Input files containing ticker symbols.
  - Models and time periods to optimize against.
  - Whether to use various filters like de-correlation.

### Preparing Input Files
- Each input file (CSV format) should have a single column containing the ticker symbols to include in the optimization.
- Any commented lines (starting with `#`) will be ignored during processing.

### Configuring Input Files in `config.yaml`
- Under the `input_files` section of the config, list multiple files if needed. For example:
  ```yaml
  input_files:
    - file1.csv
    - file2.csv

You can specify multiple models under models. The output will contain a simple average of all selected models and time periods.

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
{
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "models": "model_name_1, model_name_2",
  "symbols": ["symbol1", "symbol2"],
  "normalized_avg": { "symbol1": 0.25, "symbol2": 0.75 },
  "daily_returns": pd.DataFrame,
  "cumulative_returns": pd.DataFrame
}
```

---

# Notes

- Ensure your `config.yaml` file is correctly configured for your use case.
- The API allows dynamic overrides for symbols and configurations without modifying the local `config.yaml`.
- The symbol selection process will optionally filter out inputs based on a number of configurable methods prior to optimization so the result set may not contain all assets in the input list. Enabled by default.
- SIM_PORT is the hypothetical portfolio with the allocation result over the same period (used for graphing daily and cumulative returns).


