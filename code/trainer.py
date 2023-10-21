import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, f1_score

def train_evaluate_model(X_train, y_train, X_test, y_test, algorithm, hyperparams):
    # Select the model based on the algorithm choice
    if algorithm == "RF":
        model = RandomForestClassifier(n_estimators=hyperparams.get("n_estimators", 100),
                                       max_depth=hyperparams.get("max_depth", None),
                                       n_jobs=-1)
    elif algorithm == "XGBoost":
        model = XGBClassifier(learning_rate=hyperparams.get("learning_rate", 0.1),
                              max_depth=hyperparams.get("max_depth", 3),
                              n_estimators=hyperparams.get("n_estimators", 100),
                              n_jobs=-1)
    elif algorithm == "Logistic":
        model = LogisticRegression(C=hyperparams.get("C", 1.0),
                                   max_iter=hyperparams.get("max_iter", 100),
                                   n_jobs=-1)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Train the model
    model.fit(X_train, y_train)
    # Binarize y_test for multiclass roc_auc_score and average_precision_score
    y_bin_test = label_binarize(y_test, classes=np.unique(y_train))

    n_classes = len(np.unique(y_train))
    y_score = model.predict_proba(X_test)
    average_precision = [average_precision_score(y_bin_test[:, i], y_score[:, i]) for i in range(n_classes)]

    # Evaluate the model
    y_pred = model.predict(X_test)
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),  # Convert numpy array to list for serialization
        "roc_auc": roc_auc_score(y_bin_test, model.predict_proba(X_test), multi_class="ovr", average="macro"),
        "pr_auc": np.mean(average_precision),
        "f1": f1_score(y_test, y_pred, average="macro")
    }

    return model, results

def zero_out_wavelet_features(df):
    """Set wavelet-based feature values to zero."""
    wavelet_columns = [col for col in df.columns if any(comp in col for comp in ['LL', 'LH', 'HL', 'HH'])]
    df[wavelet_columns] = 0
    return df
