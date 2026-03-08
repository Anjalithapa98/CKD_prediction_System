import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from Backend.preprocessing.preprocessing import load_and_preprocess_data

def train_models():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, target_encoder, feature_cols = load_and_preprocess_data()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver='lbfgs',
            C=0.5,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            bootstrap=True,
            random_state=42
        ),
        "SVM": SVC(
            kernel="rbf",
            C=0.5,
            gamma='scale',
            probability=True,
            random_state=42
        ),
        "Naive Bayes": GaussianNB(var_smoothing=1e-8)
    }

    trained_models = {}

    best_model = None
    best_score = 0  
    best_model_name = ""

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
        mean_score = cv_scores.mean()

        print(f"{name} CV F1-score: {mean_score:.4f}")

        trained_models[name] = model


        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            best_model_name = name

    with open("Backend/models/ckd_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"Best model: {best_model_name}")
    print("Best model saved as 'Backend/models/ckd_model.pkl'")

    return trained_models, X_train, X_test, y_train, y_test, scaler, feature_cols



if __name__ == "__main__":
    train_models()