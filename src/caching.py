import csv
from datetime import datetime, timedelta
from pathlib import Path


def save_model_results(model_name, time_period, input_filename, symbols, scaled):
    """Saves the model results to cache."""
    cache_dir = Path.cwd() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_file = cache_dir / f"{input_filename}-{model_name}-{time_period}.csv"

    print("Saving results to model cache")
    with output_file.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for key, val in scaled.items():
            writer.writerow([key, val])

        filtered_symbols = [sym for sym in symbols if sym not in scaled.keys()]
        for symbol in filtered_symbols:
            writer.writerow([symbol, 0])


def load_model_results_from_cache(model_name, time_period, input_filename, symbols):
    """Loads the model results from cache if available and valid."""
    cache_dir = Path.cwd() / "cache"
    cache_file = cache_dir / f"{input_filename}-{model_name}-{time_period}.csv"

    if cache_file.exists():
        print(
            f"Loading results from cache for {model_name} and time period {time_period}"
        )
        with cache_file.open("r") as csvfile:
            reader = csv.reader(csvfile)
            results = {row[0]: float(row[1]) for row in reader}
        # Check if all symbols in the cached results match the provided symbols list
        cached_symbols = set(results.keys())
        provided_symbols = set(symbols)

        if cached_symbols == provided_symbols:
            return results
        else:
            print("New symbols found. Recalculating.")
            return None
    else:
        return None


def cleanup_cache(cache_dir, max_age_hours=24):
    """
    Removes files in the cache directory that are older than max_age_hours.
    """
    now = datetime.now()
    max_age = timedelta(hours=max_age_hours)

    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        print(f"Cache directory {cache_dir} does not exist.")
        return

    for filepath in cache_dir.iterdir():
        if filepath.is_file():
            creation_time = datetime.fromtimestamp(filepath.stat().st_ctime)
            if now - creation_time > max_age:
                print(f"Removing {filepath} (created at {creation_time})")
                filepath.unlink()
