from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from Backend.preprocessing.preprocessing import load_and_preprocess_data

def tuning():
   
    print("Running Hyperparameter Tuning (Anti-overfitting)...")
    X_train, X_test, y_train, y_test, scaler, _, _ = load_and_preprocess_data()

    # Logistic Regression
    lr_grid = {
   
        "C": [0.01, 0.1, 0.5, 1.0],
   
        "penalty": ["l2"],
   
        "solver": ["lbfgs"],
   
        "max_iter": [2000]
   
    }
    lr = GridSearchCV(LogisticRegression(random_state=42), lr_grid, cv=5, scoring="accuracy", n_jobs=-1)
    lr.fit(X_train, y_train)

    # SVM
    svm_grid = {
   
        "C": [0.1, 0.5, 1.0],
   
        "gamma": ["scale", 0.01, 0.1],
   
        "kernel": ["rbf"]
   
    }
    svm = GridSearchCV(SVC(probability=True, random_state=42), svm_grid, cv=5, scoring="accuracy", n_jobs=-1)
    svm.fit(X_train, y_train)

    # Random Forest
    rf_grid = {
       
        "n_estimators": [100, 150, 200],
       
        "max_depth": [4, 6, 8],
    
        "min_samples_split": [5, 8, 10],
   
     "min_samples_leaf": [2, 4, 5]
    }
    rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_grid, cv=5, scoring="accuracy", n_jobs=-1)
    
    rf.fit(X_train, y_train)
    # Naive Bayes (no hyperparams)
    nb = GaussianNB(var_smoothing=1e-8)
    nb.fit(X_train, y_train)

    best_models = {
        
        "Logistic Regression": lr.best_estimator_,
        
        "SVM": svm.best_estimator_,
        
        "Random Forest": rf.best_estimator_,
        
        "Naive Bayes": nb
    }

    print("Best LR:", lr.best_params_)
    print("Best SVM:", svm.best_params_)
    print("Best RF:", rf.best_params_)


    return best_models, X_train, X_test, y_train, y_test
