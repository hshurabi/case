# CASE Pipeline

## Overview
The **CASE (Classification-Augmented Survival Estimation)** pipeline is a machine learning framework designed to transform survival data into classification tasks, predict survival probabilities, and de-augment results to generate survival curves. The pipeline supports flexible period-based survival analysis, robust handling of incomplete probability lists, and de-augmentation of results.

## Table of Contents
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
  - [Initialization](#initialization)
  - [Fitting the Model](#fitting-the-model)
  - [Handling Incomplete Probabilities](#handling-incomplete-probabilities)
  - [De-Augmentation](#de-augmentation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites
- Python 3.7 or higher
- Required packages: 
  - `numpy`
  - `pandas`
  - `scikit-learn`

### Install the Required Packages
You can install the required packages using pip:

```bash
pip install numpy pandas scikit-learn
```

Clone the repository:

```bash
git clone https://github.com/your-username/case-pipeline.git
cd case-pipeline
```

## Features
- **Transform survival data to classification tasks**: Converts survival data into classification tasks based on specified periods (year, month, etc.).
- **Flexible de-augmentation**: Reconstructs individual survival curves from predicted probabilities.
- **Handles incomplete probability lists**: Identifies missing periods and queries the model for missing probabilities.
- **Supports various survival periods**: Allows transformation based on custom time periods (e.g., year, 6-month, 3-month).
- **Built-in augmentation tracking**: Keeps track of the transformation mapping for accurate de-augmentation.

## Usage

### Initialization
To initialize the **CASE** class, specify the maximum number of periods and the type of period:

```python
from case import CASE

# Initialize the CASE model
case_model = CASE(max_periods=5, period_type='year')
```

### Fitting the Model
You can fit the model with your data by calling the `fit()` method. The input data should include:
- `X`: Feature matrix (DataFrame).
- `y`: Structured array containing event and time fields.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Example data
X = pd.DataFrame({'Feature1': [1, 3, 5], 'Feature2': [2, 4, 6]})
y = np.array([(1, 500), (0, 200), (1, 800)], dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

# Transform the data
x_case, y_case = case_model.transform(X, y)

# Fit model
case_model.fit(x_case,y_case, RandomForestClassifier())

# Get survival scores
pred_scores = case_model.fitted_classifier.predict_proba(x_case)[:,1]

case_model.inverse_transform(pred_scores)
```

### Handling Incomplete Probabilities
If a training record is censored, use the `predict_survival_function_for_censored_training_data()` method to obtain a predicted survival function:

```python
# Example record probabilities with incomplete lists
record_probs = {0: [0.9, 0.8], 1: [0.85, 0.7, 0.6], 2: [0.75]}

# Handle incomplete probabilities
updated_record_probs = case_model.predict_survival_function_for_censored_training_data(record_probs, X, study_period=3)

print(updated_record_probs)
```

### De-Augmentation
To de-augment the predicted probabilities and generate individual survival curves, use the `deaugment()` method:

```python
# Example predicted probabilities
preds = np.random.rand(len(case_model.augmentation_map))

# De-augment to get survival curves
survival_curves = case_model.deaugment(preds)

print(survival_curves)
```

## Examples
Check the `examples/` directory for more detailed Jupyter Notebook examples, including:
- Data transformation and augmentation.
- Model training and evaluation.
- Handling incomplete probabilities.
- De-augmentation and survival curve generation.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to your branch (`git push origin feature-name`).
5. Open a Pull Request.

Please ensure your code follows Python coding standards and includes relevant tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```
