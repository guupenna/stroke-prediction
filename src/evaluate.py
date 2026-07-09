from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_validate

def evaluate(model, X_train, y_train, X_test, y_test):
    y_pred = model.predict(X_test)

    print("--- Confusion matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(cm)
    print(f"True negative: {tn}")
    print(f"False positive: {fp}")
    print(f"False negative: {fn}")
    print(f"True positive: {tp}")

    print("\n--- Classification report ---")
    print(classification_report(y_test, y_pred))

    metrics = ['accuracy', 'precision', 'recall']
    cv = cross_validate(
        model, X_train, y_train, cv=5, scoring=metrics)
    print("--- Cross validation ---")
    print(f"Accuracy: {cv['test_accuracy'].mean():.2f}")
    print(f"Precision: {cv['test_precision'].mean():.2f}")
    print(f"Recall: {cv['test_recall'].mean():.2f}")