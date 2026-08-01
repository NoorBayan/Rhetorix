import numpy as np
import time
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, set_seed
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from .custom_trainer import OrdinalTrainer
from .metrics import compute_metrics

def run_5fold_cv(model_name, df, num_labels=3, use_ordinal=True, k_folds=5, seed=42):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["clean_text"], truncation=True, padding="max_length", max_length=128)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    oof_predictions = np.zeros(len(df)) # Out-Of-Fold predictions
    oof_true = np.zeros(len(df))
    
    total_train_time = 0
    total_inf_time = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['clean_text'], df['label'])):
        print(f"\n--- Training Fold {fold+1}/{k_folds} ---")
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        train_ds = Dataset.from_pandas(train_df[['clean_text', 'label']]).map(tokenize_function, batched=True)
        val_ds = Dataset.from_pandas(val_df[['clean_text', 'label']]).map(tokenize_function, batched=True)

        classes = np.unique(train_df['label'])
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df['label'])

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type="single_label_classification"
        )

        training_args = TrainingArguments(
            output_dir=f"./results_fold_{fold}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=5, # Reduced epochs for CV speed
            load_best_model_at_end=True,
            metric_for_best_model="qwk",
            save_total_limit=1,
            report_to="none"
        )

        trainer = OrdinalTrainer(
            class_weights=weights,
            use_ordinal_loss=use_ordinal,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

        t0 = time.time()
        trainer.train()
        total_train_time += (time.time() - t0)

        t1 = time.time()
        eval_res = trainer.evaluate()
        total_inf_time += (time.time() - t1)
        
        fold_results.append(eval_res)
        
        # Save Out-Of-Fold predictions for analysis
        preds = trainer.predict(val_ds)
        oof_predictions[val_idx] = np.argmax(preds.predictions, axis=-1)
        oof_true[val_idx] = val_df['label'].values

    # Aggregate Metrics across folds
    qwk_scores = [r['eval_qwk'] for r in fold_results]
    mae_scores = [r['eval_mae'] for r in fold_results]
    f1_scores = [r['eval_macro_f1'] for r in fold_results]
    acc_scores = [r['eval_accuracy'] for r in fold_results]

    metrics = {
        'QWK': f"{np.mean(qwk_scores)*100:.2f} ±{np.std(qwk_scores)*100:.2f}",
        'MAE': f"{np.mean(mae_scores):.3f} ±{np.std(mae_scores):.3f}",
        'Macro_F1': f"{np.mean(f1_scores)*100:.2f} ±{np.std(f1_scores)*100:.2f}",
        'Accuracy': f"{np.mean(acc_scores)*100:.2f} ±{np.std(acc_scores)*100:.2f}",
        'Train_Time(s)': f"{total_train_time:.1f}",
        'Inf_Time(s)': f"{total_inf_time:.2f}"
    }

    return metrics, oof_predictions, oof_true
