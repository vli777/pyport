from collections import Counter

def scale_to_one(weights_dict):
    """
    scaling function to have filtered holding allocations sum to one
    """
    total_alloc = max(.001, sum(weights_dict.values()))
    scaled = {k: v / total_alloc for k, v in weights_dict.items()}
    return scaled


def custom_scaling(weights_dict, scaling):
    """
    returns a weights dict with custom scaled values to inc/dec the impact of an allocation result
    """
    return {k: v * scaling for k, v in weights_dict.items()}


def stacked_output(stack_dict):
    """
    Return a scaled arithmetic average of input model dicts.
    Each value in `stack_dict` should be a tuple or list where:
    - The first element is a dictionary of weights (portfolio).
    - The second element is a scaling factor (e.g., the length of the portfolio).
    """
    
    valid_values = []
    
    for value in stack_dict.values():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            portfolio, scaling_factor = value            
            if isinstance(portfolio, dict) and isinstance(scaling_factor, (int, float)):
                valid_values.append((portfolio, scaling_factor))
            else:
                print(f"Warning: Invalid types in stack_dict entry: {value}")
        else:
            print(f"Warning: Invalid entry format in stack_dict: {value}")
    
    if not valid_values:
        raise ValueError("No valid entries found in stack_dict.")
    
    # Find the maximum scaling factor
    maxlen = max([v[1] for v in valid_values])

    # Apply scaling to each portfolio
    for model in stack_dict:
        value = stack_dict[model]
        if isinstance(value, (list, tuple)) and len(value) == 2:
            portfolio, scaling_factor = value
            if isinstance(portfolio, dict) and isinstance(scaling_factor, (int, float)):
                stack_dict[model] = custom_scaling(weights_dict=portfolio,
                                                   scaling=scaling_factor / maxlen)
        else:
            print(f"Warning: Skipping invalid entry in stack_dict for model {model}")

    # Combine holdings from all models and calculate average holdings
    holding = [v for _, v in stack_dict.items()]
    total = sum(map(Counter, holding), Counter())
    average_holdings = {k: v / len(stack_dict) for k, v in total.items()}

    return average_holdings


def apply_weights(row, weights_dict):
    """
    apply scaling weights from a custom dict to a row
    """
    for i, _ in enumerate(row):
        row[i] *= weights_dict[i]
    return row


def clip_by_weight(weights_dict, mininum_weight):
    """
    filters any holdings below a min threshold
    """
    return {k: v for k, v in weights_dict.items() if v > mininum_weight}


def get_min_by_size(weights_dict, size, mininum_weight=0.01):
    """
    find the minimum allocation of a weight dict
    """
    if len(weights_dict) > size:
        sorted_weights = sorted(weights_dict.values(), reverse=True)
        mininum_weight = sorted_weights[size]
    return mininum_weight


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
