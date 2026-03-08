import os
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error
)

from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.train_models import train_models
from models.tuning import tuning

def compute_average_metrics(use_tuned=False, output_dir="performance_results"):
    if use_tuned:
    
        trained_models, X_train, X_test, y_train, y_test = tuning()
    
        suffix = "After Tuning"
    
    else:
        trained_models, X_train, X_test, y_train, y_test, scaler, feature_cols = train_models()
        suffix = "Before Tuning"

    results = []
    
    print(f"\n========== {suffix} METRICS ==========\n")

    for model_name, model in trained_models.items():
    
        y_pred = model.predict(X_test)
        y_pred = np.ravel(y_pred) 

        acc = accuracy_score(y_test, y_pred)
    
        prec = precision_score(y_test, y_pred)
    
        rec = recall_score(y_test, y_pred)
    
        f1 = f1_score(y_test, y_pred)


        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_prob)
        except:
            roc = np.nan

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
        mae = mean_absolute_error(y_test, y_pred)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
    
        cv_std = cv_scores.std()

        train_acc = accuracy_score(y_train, model.predict(X_train))
        overfit_gap = train_acc - acc

        results.append({
    
            "Model": model_name,
    
            "Accuracy": acc,
    
            "Precision": prec,
    
            "Recall": rec,
    
            "F1-Score": f1,
    
            "ROC-AUC": roc,
    
            "RMSE": rmse,
    
            "MAE": mae,
    
            "CV Mean Accuracy": cv_mean,
    
            "CV Std Dev": cv_std,
    
            "Train Accuracy": train_acc,
    
            "Overfitting Gap": overfit_gap
        })
    
        print(f"{model_name} evaluated.")

    df = pd.DataFrame(results).set_index("Model")

    avg = pd.DataFrame(df.mean()).T

    avg.index = [f"Average ({suffix})"]


    final_df = pd.concat([df, avg])

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"average_metrics_{suffix.replace(' ','_').lower()}.csv")
    final_df.to_csv(save_path)


    print(f"\nSaved at: {save_path}")
    
    print("\n", final_df.round(4))
    
    return final_df

def compare_before_after(output_dir="performance_results"):
    
    before = compute_average_metrics(False, output_dir)
    
    after = compute_average_metrics(True, output_dir)

    comparison = pd.concat(
        [before.loc[f"Average (Before Tuning)"], after.loc[f"Average (After Tuning)"]],
        axis=1
    )
    
    comparison.columns = ["Before Tuning", "After Tuning"]

    save_path = os.path.join(output_dir, "before_after_comparison.csv")
    comparison.to_csv(save_path)

    print("\nComparison saved at:", save_path)
    
    print("\n", comparison.round(4))

if __name__ == "__main__":
    
    compute_average_metrics(False)
    
    compute_average_metrics(True)
    
    compare_before_after()
