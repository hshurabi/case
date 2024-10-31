import numpy as np
import pandas as pd

from .data_processing import detect_event_time_fields


def i_ausc(survival_function, periods, event_time):
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
    if not isinstance(periods, np.ndarray):
        periods = np.array(periods)
    
    # Mask for times before or at the event time
    mask = periods <= event_time
    P = np.max(periods)
    
    # Denominator: Weighted sum across all times
    weighted_denom = np.sum(np.exp(-1/P * np.abs(event_time - periods)))
    
    # Weighted area until the event time
    weighted_area_until_event = np.sum(
        survival_function[mask] * np.exp(-1/P * np.abs(event_time - periods[mask]))
    )
    
    # Weighted area after the event time
    weighted_area_after_event = np.sum(
        (1 - survival_function[~mask]) * np.exp(-1/P * np.abs(event_time - periods[~mask]))
    )
    
    # iAUSC score calculation
    score = (weighted_area_until_event + weighted_area_after_event) / weighted_denom
    
    return score


def m_ausc(survivals_tr, y_tr, periods):
    """
    Calculate the mean Area Under the Survival Curve (mAUSC) for a dataset.
    
    Parameters:
    - survival_functions: 2D list or array-like of survival probabilities for each individual over time.
    - periods: List or array of time points corresponding to survival probabilities.
    - event_times: List or array of event times for each individual.
    
    Returns:
    - mAUSC score
    """
    # Convert inputs to numpy arrays if not already
    periods = np.array(periods)
    event_filed, time_feild = detect_event_time_fields(y_tr)
    event_times = pd.DataFrame(np.array(y_tr[time_feild]),
                               index = survivals_tr.keys())
    
    # Ensure the shape of survival_functions matches number of individuals and time points
    # assert n_times == len(periods), "Mismatch between times and survival function time points"
    
    # Initialize list to accumulate iAUSC for each individual
    iausc_scores = []
    
    for record_idx,survival in survivals_tr.items():
        
        iausc = i_ausc(survival, periods, event_times.loc[record_idx,0])
        iausc_scores.append(iausc)
    
    # Calculate mean AUC
    mAUSC_score = np.mean(iausc_scores)
    
    return mAUSC_score
