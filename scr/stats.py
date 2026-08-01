from scipy.stats import wilcoxon
import numpy as np

def run_significance_test(true_labels, preds_model_a, preds_model_b):
    """
    Compares the absolute errors of two models using Wilcoxon Signed-Rank Test.
    Returns the p-value.
    """
    errors_a = np.abs(true_labels - preds_model_a)
    errors_b = np.abs(true_labels - preds_model_b)
    
    # Check if errors are exactly the same (to avoid ValueError in wilcoxon)
    if np.array_equal(errors_a, errors_b):
        return 1.0 
        
    stat, p_value = wilcoxon(errors_a, errors_b)
    return p_value
