from collections import Counter, defaultdict


def normalize_weights(weights, min_weight):
    """
    Normalize the weights depending on the min_weight configuration. 
    If min_weight is positive, only positive weights are normalized.
    If min_weight includes negative values, both positive and negative weights are scaled.
    Weights below min_weight are filtered out.
    """
    # Separate positive and negative weights
    positive_weights = {k: v for k, v in weights.items() if v > min_weight}
    negative_weights = {k: v for k, v in weights.items() if v < -abs(min_weight)}  # Filter negatives based on absolute min_weight
    
    total_positive = sum(positive_weights.values())
    total_negative = abs(sum(negative_weights.values()))  # We want the absolute sum of negatives

    # Case 1: Only normalize positive weights if min_weight is strictly positive
    if min_weight >= 0:
        if total_positive > 0:
            # Scale positive weights to sum to 1
            normalized_positive = {k: v / total_positive for k, v in positive_weights.items()}
        else:
            normalized_positive = positive_weights  # If no positive weights, leave it as is
        
        # Return only positive weights above min_weight
        return {k: v for k, v in normalized_positive.items() if v > min_weight}

    # Case 2: Normalize both positive and negative weights if min_weight includes negatives
    elif min_weight < 0:
        # Normalize positive weights
        normalized_positive = {k: v / total_positive for k, v in positive_weights.items()} if total_positive > 0 else positive_weights
        # Normalize negative weights
        normalized_negative = {k: v / total_negative for k, v in negative_weights.items()} if total_negative > 0 else negative_weights

        # Combine the two, filter out weights below min_weight for both positive and negative
        normalized_combined = {**normalized_positive, **normalized_negative}
        return {k: v for k, v in normalized_combined.items() if abs(v) > abs(min_weight)}


def stacked_output(stack_dict):
    """
    Return the arithmetic average of input model dicts.
    Each value in `stack_dict` should be a dictionary of weights (portfolio).
    """

    # Collect all unique symbols across all portfolios
    all_symbols = set()
    for value in stack_dict.values():
        if isinstance(value, dict):
            all_symbols.update(value.keys())
        else:
            print(f"Warning: Invalid entry format in stack_dict: {value}")

    if not all_symbols:
        raise ValueError("No valid portfolios found in stack_dict.")

    # Initialize a defaultdict to store the summed weights (starting from 0 for each symbol)
    total_weights = defaultdict(float)

    # Track the number of portfolios processed
    num_portfolios = len(stack_dict)

    # Sum all portfolios, ensuring missing symbols default to 0
    for portfolio in stack_dict.values():
        if isinstance(portfolio, dict):
            for symbol in all_symbols:
                total_weights[symbol] += portfolio.get(symbol, 0)  # Default to 0 if the symbol is missing

    # Calculate the arithmetic mean by dividing each symbol's total weight by the number of portfolios
    average_holdings = {symbol: total_weight / num_portfolios for symbol, total_weight in total_weights.items()}

    return average_holdings


def holdings_match(cached_model_dict, input_file_symbols, test_mode = False):
    """
    check if all selected symbols match between the input list and cache files
    """
    for symbol in cached_model_dict.keys():
        if symbol not in input_file_symbols:
            if test_mode:
                print(symbol, "not found in", input_file_symbols)
            return False
    for symbol in input_file_symbols:
        if symbol not in cached_model_dict.keys():
            if test_mode:
                print(symbol, "not found in", cached_model_dict.keys())
            return False
    return True
