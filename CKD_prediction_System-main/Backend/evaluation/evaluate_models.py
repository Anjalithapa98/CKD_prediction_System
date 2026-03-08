import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error
)

from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def evaluate_models(use_tuned=False, output_dir="evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)

    if use_tuned:
    
        from models.tuning import tuning
    
        trained_models, X_train, X_test, y_train, y_test = tuning()
        
        suffix = "after_tuning"
    
    else:
        
        from models.train_models import train_models
        trained_models, X_train, X_test, y_train, y_test, scaler, feature_cols = train_models()
        suffix = "before_tuning"

    results_file = os.path.join(output_dir, f"evaluation_{suffix}.txt")

    with open(results_file, "w") as f:
        
        for model_name, model in trained_models.items():
            y_pred = np.ravel(model.predict(X_test))

            accuracy = accuracy_score(y_test, y_pred)
            
            precision = precision_score(y_test, y_pred)
            
            recall = recall_score(y_test, y_pred)
            
            f1 = f1_score(y_test, y_pred)

            try:
            
                y_prob = model.predict_proba(X_test)[:, 1]
            
                roc_auc = roc_auc_score(y_test, y_prob)
            
                fpr, tpr, _ = roc_curve(y_test, y_prob)
            
                plt.figure()
            
                plt.plot(fpr, tpr)
                plt.plot([0,1],[0,1])
                plt.title(f"ROC Curve - {model_name} ({suffix})")
                plt.xlabel("FPR"); plt.ylabel("TPR")
                plt.savefig(os.path.join(output_dir, f"roc_{model_name}_{suffix}.png"))
                plt.close()
            
            except:
            
                roc_auc = np.nan

            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure()
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title(f"Confusion Matrix - {model_name} ({suffix})")
            plt.savefig(os.path.join(output_dir, f"cm_{model_name}_{suffix}.png"))
            plt.close()

            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

            train_acc = accuracy_score(y_train, np.ravel(model.predict(X_train)))
            overfit_gap = train_acc - accuracy


            print(f"\nModel: {model_name}")
            print(f"Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, CV Mean={cv_mean:.4f}, Overfit Gap={overfit_gap:.4f}")
            f.write(f"Model: {model_name}\nAccuracy={accuracy:.4f}\nPrecision={precision:.4f}\nRecall={recall:.4f}\nF1={f1:.4f}\nROC-AUC={roc_auc:.4f}\nCV Mean={cv_mean:.4f}\nOverfit Gap={overfit_gap:.4f}\nConfusion Matrix:\n{cm}\n{'-'*40}\n")

    print(f"\nEvaluation saved at: {results_file}")

if __name__ == "__main__":
    evaluate_models(False)
    evaluate_models(True)