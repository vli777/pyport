import numpy as np
import pandas as pd

def convert_to_dict(weights, asset_names):
    """
    Convert optimizer weights to a dictionary.
    Handles ndarray, DataFrame, and dict types.
    """
    if isinstance(weights, np.ndarray):
        # If the weights array is 2D (like in your example), ensure it's flattened before mapping
        if weights.ndim > 1:
            # Assuming weights is a 2D array with a single row (like in your case), flatten it
            weights = weights.flatten()
        
        # Ensure that the length of weights matches the length of asset_names
        if len(weights) != len(asset_names):
            raise ValueError("The number of weights does not match the number of asset names.")
        
        # Map each asset name to the corresponding weight
        return {asset: weight for asset, weight in zip(asset_names, weights)}
    
    elif isinstance(weights, pd.DataFrame):
        # Convert DataFrame to dict (assuming single column for weights)
        return weights.squeeze().to_dict()
    
    elif isinstance(weights, dict):
        # Already a dictionary, no conversion needed
        return weights
    
    else:
        raise TypeError("Unsupported weight type: must be ndarray, DataFrame, or dict.")

