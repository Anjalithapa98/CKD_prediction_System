import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


def load_models(use_tuned):

    if use_tuned:

        from models.tuning import tuning

        trained_models, X_train, X_test, y_train, y_test = tuning()

        label = "after_tuning"

    else:

        from models.train_models import train_models
        trained_models, X_train, X_test, y_train, y_test, scaler, feature_cols = train_models()
        label = "before_tuning"

    return trained_models, X_train, X_test, y_train, y_test, label

def plot_performance_measures(output_dir="performance_results", use_tuned=False):
    os.makedirs(output_dir, exist_ok=True)

    trained_models, X_train, X_test, y_train, y_test, label = load_models(use_tuned)

    names, accs, precs, recs, f1s, rocs, gaps = [], [], [], [], [], [], []

    for model_name, model in trained_models.items():
        y_pred = np.ravel(model.predict(X_test))
        names.append(model_name)
        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred))
        recs.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))

        try:
            y_prob = model.predict_proba(X_test)[:,1]
            roc = roc_auc_score(y_test, y_prob)

        except:

            roc = 0
        rocs.append(roc)

        train_acc = accuracy_score(y_train, np.ravel(model.predict(X_train)))
        gaps.append(train_acc - accs[-1])

    # Accuracy
    plt.figure()
    plt.bar(names, accs)
    plt.title(f"Accuracy ({label})")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30)
    plt.savefig(os.path.join(output_dir, f"accuracy_{label}.png"))
    plt.close()

    # Statistical measures
    x = np.arange(len(names))
    width = 0.25

    plt.figure()
    plt.bar(x - width, precs, width, label="Precision")
    plt.bar(x, recs, width, label="Recall")
    plt.bar(x + width, f1s, width, label="F1")
    plt.xticks(x, names, rotation=30)
    plt.title(f"Statistical Measures ({label})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"statistical_{label}.png"))
    plt.close()

    # ROC-AUC
    plt.figure()
    plt.bar(names, rocs)
    plt.title(f"ROC-AUC ({label})")
    plt.xticks(rotation=30)
    plt.savefig(os.path.join(output_dir, f"roc_auc_{label}.png"))
    plt.close()

    # Overfitting gap
    plt.figure()
    plt.bar(names, gaps)
    plt.title(f"Overfitting Gap ({label})")
    plt.ylabel("Train Accuracy - Test Accuracy")
    plt.xticks(rotation=30)
    plt.savefig(os.path.join(output_dir, f"overfitting_gap_{label}.png"))
    plt.close()

    # Combined ROC curve
    plt.figure()

    for model_name, model in trained_models.items():

        try:
            y_prob = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{model_name}")
        except:
            continue
    plt.plot([0,1],[0,1],'--')
    plt.title(f"Combined ROC ({label})")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"roc_curve_{label}.png"))
    plt.close()


    print(f"All plots saved in: {output_dir}")

def compare_before_after(output_dir="performance_results"):

    os.makedirs(output_dir, exist_ok=True)

    before_models, _, X_test_b, _, y_test_b, _ = load_models(False)

    after_models, _, X_test_a, _, y_test_a, _ = load_models(True)

    names = list(before_models.keys())
    before_acc = [accuracy_score(y_test_b, np.ravel(before_models[m].predict(X_test_b))) for m in names]
    after_acc = [accuracy_score(y_test_a, np.ravel(after_models[m].predict(X_test_a))) for m in names]

    x = np.arange(len(names))

    width = 0.35

    plt.figure()

    plt.bar(x - width/2, before_acc, width)

    plt.bar(x + width/2, after_acc, width)

    plt.xticks(x, names, rotation=30)
    plt.title("Before vs After Accuracy")
    plt.savefig(os.path.join(output_dir, "before_after_accuracy.png"))
    plt.close()

    print("Before vs After comparison saved.")

if __name__ == "__main__":
    plot_performance_measures("performance_results/before_tuning", False)
    plot_performance_measures("performance_results/after_tuning", True)
    compare_before_after()
    