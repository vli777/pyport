from reversion.get_reversion_recommendations import apply_mean_reversion


def filter_with_reversion(returns_df):
    """
    Filter tickers using mean reversion.

    Args:
        returns_df (pd.DataFrame): DataFrame with symbol columns and returns indexed on date.

    Returns:
        list[str]: Final list of filtered tickers.
    """
    # Identify the set of all tickers from the returns_df columns
    all_tickers = set(returns_df.columns)

    # Generate mean reversion exclusions and inclusions
    mean_reversion_exclusions, mean_reversion_inclusions = apply_mean_reversion(
        returns_df=returns_df, plot=False  # Adjust plot as needed
    )

    # Convert exclusions and inclusions to sets for set operations
    mean_reversion_exclusions = set(mean_reversion_exclusions)
    mean_reversion_inclusions = set(mean_reversion_inclusions)

    # Apply mean reversion logic
    filtered_set = all_tickers - mean_reversion_exclusions  # Remove excluded tickers
    filtered_set |= mean_reversion_inclusions  # Add included tickers

    # Return final filtered tickers as a list
    return list(filtered_set)


