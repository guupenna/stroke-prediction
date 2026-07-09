from sklearn.metrics import confusion_matrix, classification_report

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))