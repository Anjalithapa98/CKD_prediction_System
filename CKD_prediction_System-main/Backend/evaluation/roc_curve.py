import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def load_models(use_tuned=False):
    
    if use_tuned:
    
        from models.tuning import tuning
    
        trained_models, X_train, X_test, y_train, y_test = tuning()
    
        label = "after_tuning"
    
    else:
    
        from models.train_models import train_models
        trained_models, X_train, X_test, y_train, y_test, scaler, feature_cols = train_models()
        label = "before_tuning"
    
    return trained_models, X_test, y_test, label

def plot_individual_roc(output_dir="roc_results", use_tuned=False):
    
    os.makedirs(output_dir, exist_ok=True)
    
    trained_models, X_test, y_test, label = load_models(use_tuned)
    
    roc_data = []

    
    for model_name, model in trained_models.items():
    
        if hasattr(model, "predict_proba"):
    
            y_scores = model.predict_proba(X_test)[:, 1]
    
        else:
    
            y_scores = model.decision_function(X_test)

    
        fpr, tpr, _ = roc_curve(y_test, y_scores)
    
        roc_auc = auc(fpr, tpr)
        roc_data.append({"Model": model_name, "ROC-AUC": roc_auc})

    
        plt.figure()
    
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],'--')
        plt.title(f"ROC Curve - {model_name} ({label})")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.savefig(os.path.join(output_dir, f"{model_name.replace(' ','_')}_{label}_roc.png"))
        plt.close()

    
    df = pd.DataFrame(roc_data)
    df.to_csv(os.path.join(output_dir, f"roc_auc_table_{label}.csv"), index=False)
    print(f"Saved ROC curves and table for {label}")
    return df

def plot_combined_roc(output_dir="roc_results", use_tuned=False):
    
    os.makedirs(output_dir, exist_ok=True)
    
    trained_models, X_test, y_test, label = load_models(use_tuned)
    
    plt.figure()
    
    for model_name, model in trained_models.items():
    
        if hasattr(model, "predict_proba"):
    
            y_scores = model.predict_proba(X_test)[:, 1]
    
        else:
    
            y_scores = model.decision_function(X_test)
    
        fpr, tpr, _ = roc_curve(y_test, y_scores)
    
        roc_auc = auc(fpr, tpr)
    
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'--')
    plt.title(f"Combined ROC ({label})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"combined_roc_{label}.png"))
    plt.close()
    print(f"Saved combined ROC for {label}")

def compare_before_after(output_dir="roc_results"):
    
    before = plot_individual_roc(output_dir, False)
    
    after = plot_individual_roc(output_dir, True)
    comparison = before.merge(after, on="Model", suffixes=("_Before","_After"))
    comparison.to_csv(os.path.join(output_dir, "roc_auc_comparison.csv"), index=False)
    print("ROC-AUC comparison saved\n", comparison.round(4))

if __name__ == "__main__":
    
    plot_individual_roc("roc_results", False)
    
    plot_individual_roc("roc_results", True)
    
    plot_combined_roc("roc_results", False)
    
    plot_combined_roc("roc_results", True)
    
    compare_before_after("roc_results")
