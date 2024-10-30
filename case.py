import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CASE(BaseEstimator, TransformerMixin):
    def __init__(self, study_period, period_type='year'):
        """
        Initialize the CASE Transformer with period settings.
        
        :param study_period: Maximum number of periods to consider for oversampling.
        :param period_type: Type of period to use ('year', 'month', '6-month', '3-month').
        """
        self.study_period = study_period
        self.period_type = period_type
        self.period_lengths = {
            'year': 365,
            'month': 30,
            '6-month': 182,
            '3-month': 91
        }
        if period_type not in self.period_lengths:
            raise ValueError("Invalid period_type. Choose from 'year', 'month', '6-month', '3-month'.")
        self.period_length = self.period_lengths[period_type]
        self.event_field = None
        self.time_field = None
        self.transformation_map = None
        self.classifier = None
        self.regressor = None
        

    def _detect_fields(self, y):
        """
        Detect the event and time fields in the structured array y.
        
        :param y: Structured array with event and time fields.
        """
        for name, dtype in y.dtype.fields.items():
            if np.issubdtype(dtype[0], np.bool_) or np.issubdtype(dtype[0], np.integer):
                self.event_field = name
            elif np.issubdtype(dtype[0], np.floating):
                self.time_field = name
        
        if not self.event_field or not self.time_field:
            raise ValueError("Unable to detect the event or time field in y.")

    def transform(self, X, y, is_test_data = False):
        """
        Transform survival data into classification data by oversampling for each period.
        
        :param X: Feature matrix (array-like, shape = (n_samples, n_features)).
        :param y: Structured array with event and time fields.
        :return: Augmented feature matrix (X_aug), augmented target vector (y_aug).
        """
        # Detect event and time fields if not already set
        if self.event_field is None or self.time_field is None:
            self._detect_fields(y)
        
        if self.transformation_map is None:
            self.transformation_map = {}
            self.transformation_map['test'] = {}
            self.transformation_map['train'] = {}

        # Convert X to a NumPy array if it's a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a Pandas dataframe.")

        # Initialize lists to store augmented data and targets
        X_aug, y_aug = [], []

        # Iterate over the records
        for idx, (row,row_y) in enumerate(zip(X.iterrows(), y)):
            
            # Extract event and survival time for the current record
            event = row_y[self.event_field]
            
            surv_period = int(np.floor(row_y[self.time_field] / self.period_length))
            
            # Determine the range of periods for oversampling
            if event or is_test_data:
                periods = np.arange(0, self.study_period + 1)
            else:
                periods = np.arange(0, surv_period + 2)
            
            # Track the indices of the augmented data for each original record
            aug_indices = []

            # Duplicate the record for each period
            
            for period in periods:
                row_copy = row[1].copy()
                row_copy['period'] = period

                # Add to augmented data
                X_aug.append(row_copy.values)
                target = 1 if period <= surv_period else 0
                y_aug.append(target)

                # Track the index for mapping
                aug_indices.append(len(X_aug) - 1)

            # Update the augmentation map
            if is_test_data:
                self.transformation_map['test'][row[0]] = aug_indices
            else:
                self.transformation_map['train'][row[0]] = aug_indices

        # Convert the augmented data to Pandas
        X_aug_df = pd.DataFrame(X_aug, columns=X.columns.tolist()+['period'])
        
        # Return Augmented data
        return X_aug_df, y_aug


    def inverse_transform(self, preds, is_test_data= False):
        """
        De-augment the predicted survival probabilities to reconstruct individual survival curves.

        :param preds: Predicted probabilities for the augmented data.
        :return: De-augmented survival curves for original records.
        """
        if self.transformation_map is None:
            raise ValueError("No augmentation map found. Ensure that the model has been fitted.")

        survival_curves = {}

        # Iterate over the original record indices
        current_map = self.transformation_map['test'] if is_test_data else self.transformation_map['train']
        for idx, aug_indices in current_map.items():
            # Extract the predicted probabilities for the current original record
            pred_probs = preds[aug_indices]

            # Calculate the cumulative survival probability for each period
            survival_curve = [np.prod(pred_probs[:period]) for period in range(1, len(pred_probs) + 1)]

            # Store the survival curve for the original record
            survival_curves[idx] = survival_curve
            
        return survival_curves
    
    def fit_classifier(self, X_aug, y_aug, classifier):
        """
        Fit the CASE model by augmenting data and training a classification model.
        
        :param X: Feature matrix.
        :param T: Time-to-event vector.
        :param E: Event indicator vector (1 if event occurred, 0 if censored).
        """
        self.classifier = classifier
        self.fitted_classifier = self.classifier.fit(X_aug, y_aug)
        return self
    
    # def predict_survival_function(self, X):
    #     """
    #     Predict survival probabilities for new data.

    #     :param X: Feature matrix.
    #     :return: Predicted survival probabilities for each time point.
    #     """
    #     survival_probs = []

    #     for x in X:
    #         probs = []
    #         for tau in range(1, self.study_period + 1):
    #             x_aug = np.append(x, tau).reshape(1, -1)
    #             prob = self.fitted_classifier.predict_proba(x_aug)[0, 1]  # Probability of survival
    #             probs.append(prob)
    #         survival_probs.append(probs)

    #     return np.array(survival_probs)

    def construct_case_regression_data(self,X, survivals, **kwargs):
        # Filter data to include event times > study_period
        # Call inverse_transform to predict survival
        # 
        y = kwargs.get('y',None)
        
        if y is None:
            regression_indices = X.index.values
        else:
            regression_mask = ((y[self.time_field]/self.period_length<(self.study_period)) & \
                                (y[self.event_field]==True) ) | \
                                (y[self.time_field]/self.period_length>=(self.study_period))
            self.regression_indices = X.reset_index()[regression_mask].set_index('index',drop=True).index.values
            regression_indices = self.regression_indices 

        probs_df = pd.DataFrame([survivals[record] for record in survivals.keys() if \
                                record in regression_indices], 
                                columns = [str(i) for i in np.arange(0,self.study_period+1)],
                                index=regression_indices)

        case_reg_df = pd.concat([X.loc[regression_indices,:],probs_df], axis=1)
        if y is None:
            return case_reg_df
        else:
            case_reg_y = y[self.time_field][regression_indices]
            return case_reg_df, case_reg_y

    def fit_regression(self, X_reg, y_reg, regressor):
        """
        Fit a regression model on the combined dataset to predict survival times.
        
        :param X: Original feature matrix.
        :param survival_probs: Predicted survival probabilities.
        :return: Fitted regression model.
        """
        
        self.fitted_regressor = regressor.fit(X_reg, y_reg)
        return self

    def predict_survival_times(self, X):
        """
        Predict exact survival times using the fitted regression model.

        :param X: Feature matrix.
        :return: Predicted survival times.
        """
        if self.regressor is None:
            raise ValueError("Regression model is not fitted. Call fit_regression() first.")

        return self.regressor.predict(X)

    def predict_survival_function_for_censored_training_data(self, X_tr, record_probs):
        """
        Handle records with incomplete probability lists by constructing new augmented samples and predicting.
        
        :param record_probs: Dictionary with records as keys and probability lists as values.
        :param X_test: Original test set (DataFrame).
        :param study_period: Total number of periods to be considered.
        :return: Updated record probabilities.
        """
        # Iterate over the records in the probability dictionary
        for record, probs in record_probs.items():
            current_len = len(probs)

            # Check if the current list size is less than the study period
            if current_len < self.study_period + 1:
                # Get the original test record
                # Construct new augmented samples for the missing periods
                new_aug_samples = []
                for period in range(current_len, self.study_period + 1):
                    test_record = X_tr.iloc[record].copy()
                    
                    test_record['period'] = period
                    new_aug_samples.append(test_record.values)
                
                # Convert new augmented samples to a NumPy array
                new_aug_samples_df = pd.DataFrame(new_aug_samples,columns=self.fitted_classifier.feature_names_in_)
                
                # Query the model to get probabilities for the missing periods
                new_probs = self.classifier.predict_proba(new_aug_samples_df)[:, 1]

                # Update the record's probability list with the new probabilities
                record_probs[record].extend(new_probs.tolist())

        return record_probs