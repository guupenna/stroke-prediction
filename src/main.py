from preprocessing import load_data, create_preprocessor
from models import create_knn
from evaluate import evaluate

def main():
    X_train, X_test, y_train, y_test = load_data('../data/healthcare-dataset-stroke-data.csv')

    preprocessor = create_preprocessor()

    model = create_knn(preprocessor)

    model.fit(X_train, y_train)

    evaluate(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()