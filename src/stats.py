from scipy.stats import wilcoxon
import numpy as np

def run_significance_test(true_labels, preds_model_a, preds_model_b):
    errors_a = np.abs(true_labels - preds_model_a)
    errors_b = np.abs(true_labels - preds_model_b)
    
    if np.array_equal(errors_a, errors_b):
        return 1.0 
        
    stat, p_value = wilcoxon(errors_a, errors_b)
    return p_value
