import numpy as np

def iAUSC(survival_function, time, event_time):
    """
    Calculate the integrated Area Under the Survival Curve (iAUSC) for a given survival function.
    
    Parameters:
    - survival_function: List or array of survival probabilities over time.
    - time: List or array of time points corresponding to the survival function.
    - event_time: Event time for the individual.
    
    Returns:
    - iAUSC score
    """
    # Convert inputs to numpy arrays if they are not already
    if not isinstance(survival_function, np.ndarray):
        survival_function = np.array(survival_function)
    if not isinstance(time, np.ndarray):
        time = np.array(time)
    
    # Mask for times before or at the event time
    mask = time <= event_time
    P = np.max(time)
    
    # Denominator: Weighted sum across all times
    weighted_denom = np.sum(np.exp(-1/P * np.abs(event_time - time)))
    
    # Weighted area until the event time
    weighted_area_until_event = np.sum(
        survival_function[mask] * np.exp(-1/P * np.abs(event_time - time[mask]))
    )
    
    # Weighted area after the event time
    weighted_area_after_event = np.sum(
        (1 - survival_function[~mask]) * np.exp(-1/P * np.abs(event_time - time[~mask]))
    )
    
    # iAUSC score calculation
    score = (weighted_area_until_event + weighted_area_after_event) / weighted_denom
    
    return score
