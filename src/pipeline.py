import numpy as np
import pandas as pd  
import time
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, set_seed
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from .custom_trainer import OrdinalTrainer
from .metrics import compute_metrics

def run_5fold_cv(model_name, df, num_labels=3, use_ordinal=True, k_folds=5, seed=42):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["clean_text"], truncation=True, padding="max_length", max_length=128)

    # التقسيم الأساسي للـ Out-of-fold (هذا سيكون الـ Test النهائي)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    oof_predictions = np.zeros(len(df))
    oof_true = np.zeros(len(df))
    oof_probs = np.zeros((len(df), num_labels)) # Added to track probabilities
    
    total_train_time = 0
    total_inf_time = 0

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(df['clean_text'], df['label'])):
        print(f"\n--- Training Fold {fold+1}/{k_folds} ---")
        
        # استخراج بيانات التدريب/التقييم الداخلي وبيانات الاختبار النهائي
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
        # الجديد هنا: اقتطاع 15% من بيانات التدريب لتكون Validation حقيقي لاختيار الـ Checkpoint
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_df['clean_text'], train_val_df['label'], 
            test_size=0.15, stratify=train_val_df['label'], random_state=seed
        )
        
        # تحويلها إلى DataFrames
        train_df = pd.DataFrame({'clean_text': train_texts, 'label': train_labels}).reset_index(drop=True)
        val_df = pd.DataFrame({'clean_text': val_texts, 'label': val_labels}).reset_index(drop=True)
        
        # تحويل البيانات إلى Dataset Format الخاصة بـ HuggingFace
        train_ds = Dataset.from_pandas(train_df[['clean_text', 'label']]).map(tokenize_function, batched=True)
        val_ds = Dataset.from_pandas(val_df[['clean_text', 'label']]).map(tokenize_function, batched=True)
        test_ds = Dataset.from_pandas(test_df[['clean_text', 'label']]).map(tokenize_function, batched=True)

        # حساب أوزان الفئات بناءً على بيانات التدريب فقط
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
            num_train_epochs=5,
            load_best_model_at_end=True, # سيختار الأفضل بناءً على val_ds
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
            eval_dataset=val_ds, # التقييم لاختيار الـ Checkpoint يتم هنا
            compute_metrics=compute_metrics,
        )

        # التدريب
        t0 = time.time()
        trainer.train()
        total_train_time += (time.time() - t0)

        # التقييم النهائي والأهم (على test_ds غير المرئية تماماً)
        t1 = time.time()
        test_preds = trainer.predict(test_ds) # نستخدم predict للتقييم الخارجي
        
        eval_res = test_preds.metrics
        
        cleaned_eval_res = {
            'eval_qwk': eval_res['test_qwk'],
            'eval_mae': eval_res['test_mae'],
            'eval_macro_f1': eval_res['test_macro_f1'],
            'eval_accuracy': eval_res['test_accuracy']
        }
        
        total_inf_time += (time.time() - t1)
        
        fold_results.append(cleaned_eval_res)
        
        # حفظ التوقعات والاحتمالات للـ OOF
        oof_predictions[test_idx] = np.argmax(test_preds.predictions, axis=-1)
        oof_true[test_idx] = test_df['label'].values
        oof_probs[test_idx] = torch.nn.functional.softmax(torch.tensor(test_preds.predictions), dim=-1).numpy()

    # Aggregate Metrics
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

    return metrics, fold_results, oof_predictions, oof_true, oof_probs
