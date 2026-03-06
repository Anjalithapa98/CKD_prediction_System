import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from Backend.preprocessing.preprocessing import load_and_preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from models.tuning import tuning

def get_default_models():
    X_train, X_test, y_train, y_test, scaler, _, _ = load_and_preprocess_data()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42),
        "SVM": SVC(C=0.5, kernel='rbf', probability=True, random_state=42),
        "Naive Bayes": GaussianNB()
    }
    for m in models.values(): m.fit(X_train, y_train)
    return models, X_train, X_test, y_train, y_test

def get_tuned_models():
  
    best_models, X_train, X_test, y_train, y_test = tuning()
  
    return best_models, X_train, X_test, y_train, y_test

def evaluate(models, X_train, X_test, y_train, y_test, output_dir, prefix):
  
    os.makedirs(output_dir, exist_ok=True)
  
    metrics_list = []

    plt.figure(figsize=(8,6), dpi=300)
  
    for model_name, model in models.items():
        y_pred = np.ravel(model.predict(X_test))
        y_train_pred = np.ravel(model.predict(X_train))

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_pred)
  
        gap = train_acc - test_acc
  
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        try:
            y_prob = model.predict_proba(X_test)[:,1]
            roc_auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
        except:
            roc_auc = np.nan

  
        metrics_list.append({
  
            "Model": model_name,
  
            "Train Accuracy": train_acc,
  
            "Test Accuracy": test_acc,
            "Overfit Gap": gap,
            "Precision": precision,
  
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc
        })

    plt.plot([0,1],[0,1],'--')
    plt.title(f"ROC Curves - {prefix}")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"roc_curves_{prefix}.png"))
    plt.close()

    df = pd.DataFrame(metrics_list)
    df.to_csv(os.path.join(output_dir, f"full_metrics_{prefix}.csv"), index=False)
    print(f"Saved metrics and ROC curves for {prefix}")
    return df

if __name__ == "__main__":
    output_dir = "full_evaluation_results"

    print("Evaluating BEFORE tuning...")
    default_models, X_train, X_test, y_train, y_test = get_default_models()
    evaluate(default_models, X_train, X_test, y_train, y_test, output_dir, "before_tuning")

    print("Evaluating AFTER tuning...")
    tuned_models, X_train, X_test, y_train, y_test = get_tuned_models()
    evaluate(tuned_models, X_train, X_test, y_train, y_test, output_dir, "after_tuning")