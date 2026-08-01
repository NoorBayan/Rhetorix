import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

def generate_ordinal_reports(y_true, y_pred, model_name):
    target_names = ["Low", "Medium", "High"]

    df_results = pd.DataFrame({'Actual': [target_names[int(i)] for i in y_true], 
                               'Predicted': [target_names[int(i)] for i in y_pred]})
    
    print(f"\n{'='*60}\n📊 Raw Data for OOF Predictions ({model_name})\n{'='*60}")
    cross_tab = pd.crosstab(df_results['Actual'], df_results['Predicted'])
    cross_tab = cross_tab.reindex(index=target_names, columns=target_names, fill_value=0)
    print(cross_tab.to_markdown())

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=target_names, yticklabels=target_names)
    plt.plot([0, 3], [0, 3], color='black', lw=2, linestyle='--')
    plt.title(f"Ordinal Confusion Matrix ({model_name})", pad=15)
    plt.ylabel('Actual Processing Effort')
    plt.xlabel('Predicted Processing Effort')
    plt.tight_layout()
    plt.savefig(f'ordinal_cm_{model_name.replace("/", "_")}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    errors = np.abs(y_true - y_pred)
    error_counts = pd.Series(errors).value_counts().sort_index()
    print(f"\n📉 Error Severity Breakdown:")
    for severity, count in error_counts.items():
        print(f"  - Severity {severity}: {count} instances ({(count/len(y_true))*100:.1f}%)")
