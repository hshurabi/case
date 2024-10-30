# import numpy as 

# def get_surv_score_case(patient_tx_id, version = 1):
#     surv = aosa_result_test_masked_6[aosa_result_test_masked_6['TX_ID']==patient_tx_id]['y_pred_prob'].values
#     surv = np.concatenate(([1.0], surv))
#     case_time_years = aosa_result_test_masked_6[aosa_result_test_masked_6['TX_ID']==patient_tx_id]['each_year_int'].values
#     case_time_years = np.concatenate(( case_time_years, [33.0]))
#     event_time = Mydatasets_Cox['outcome_test'][Mydatasets_Cox['outcome_test']['TX_ID']==patient_tx_id]['Surv_GS_Y'].values[0]    
#     P = len(surv)
#     return survival_function_score(surv,case_time_years,event_time,P=P,version = version)


# def survival_function_score(surv_probs,surv_preiod,event_time,P, version=1):
#     """
#     Calculate the ideal score for a survival curve.
    
#     Parameters:
#     - times: Array-like, representing the time points.
#     - events: Array-like, representing the event indicators (1 for event, 0 for censored).
    
#     Returns:
#     - score: The ideal score for the survival curve.
#     """
#     # Get surv function 

#     # Find the time of event
#     # event_time = 
    
#     # Until survival mask
#     # mask = < event_time
#     if version != 's-score':
#         mask = surv_preiod <= event_time
    
#         # Calculate the area under the survival curve until time of event
#         area_until_event = np.sum(surv_probs[mask])/(np.sum(surv_preiod[mask]))
        
#         # Calculate the area under the survival curve after time of event
#         area_after_event = np.sum(1-surv_probs[~mask])/(np.sum(surv_preiod[~mask]))

#         # Calculate weighted AUSC
#         weighted_area_until_event =  np.sum(np.array(surv_probs[mask])*np.array(surv_preiod[mask]))

#         # weighted_area_after_event
#         weighted_area_after_event =  np.sum(np.array(1-surv_probs[~mask])*np.array(surv_preiod[~mask]))


#     # Denominator
#     weighted_denom = np.sum( np.exp(-1/P*np.abs(event_time-np.array(surv_preiod))) )
#     if version ==8:
#             # Calculate weighted AUSC
#         weighted_area_until_event_1 =  np.sum(np.array(surv_probs[mask])*np.exp(-1/P*np.abs(event_time-np.array(surv_preiod[mask]))))

#         # weighted_area_after_event
#         weighted_area_after_event_1 =  np.sum(np.array(1-surv_probs[~mask])*np.exp(-1/P*np.abs(event_time-np.array(surv_preiod[~mask]))))

#         score = (weighted_area_until_event_1+weighted_area_after_event_1 )/weighted_denom

#     elif version =="s-score":
#         score = (np.sum(np.array(surv_probs)*np.exp(-event_time/P)) )/weighted_denom
#     return score