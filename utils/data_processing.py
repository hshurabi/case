import numpy as np



def detect_event_time_fields(y):
    """
    Detects the names of the event and time fields in a structured array.

    Parameters:
    - y: Structured array with fields representing event and time.

    Returns:
    - event_field: Name of the field representing the event (boolean or integer type).
    - time_field: Name of the field representing time (floating-point type).
    
    Raises:
    - ValueError: If event or time field cannot be detected.
    """
    event_field = None
    time_field = None

    # Loop through fields in the structured array
    for name, dtype in y.dtype.fields.items():
        # Identify event field by boolean or integer type
        if np.issubdtype(dtype[0], np.bool_) or np.issubdtype(dtype[0], np.integer):
            event_field = name
        # Identify time field by floating-point type
        elif np.issubdtype(dtype[0], np.floating):
            time_field = name
    
    # Raise an error if either field is missing
    if event_field is None or time_field is None:
        raise ValueError("Unable to detect the event or time field in y.")
    
    return event_field, time_field









def detect_uncensored_records(X, y, study_period, period_length=1):
    """
    Detects indices of uncensored records in the dataset based on event time and study period.

    Parameters:
    - X: DataFrame, feature matrix.
    - y: Structured array with event and time fields.
    - study_period: int, the study period cutoff for uncensored records.
    - period_length: int, optional, length of each period (default is 1).

    Returns:
    - Array of indices for uncensored records in X.
    """
    # Detect event and time fields in structured array y
    event_field, time_field = detect_event_time_fields(y)
    
    # Calculate time periods
    time_in_periods = y[time_field] / period_length
    
    # Define the mask for uncensored records
    is_uncensored = ((time_in_periods < study_period) & (y[event_field])) | \
                    (time_in_periods >= study_period)
    
    # Get indices of uncensored records
    uncensored_indices = X.index[is_uncensored].values

    return uncensored_indices
