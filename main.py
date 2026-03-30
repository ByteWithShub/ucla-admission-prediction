# Main execution script for UCLA Admission Prediction project
import joblib
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import clean_data, encode_data, split_features_target, scale_data
from src.train_model import train_mlp_classifier, save_model
from src.evaluate import evaluate_model, save_loss_curve


def main():
    df = load_data("Admission.csv")
    df = clean_data(df)
    df = encode_data(df)

    X, y = split_features_target(df)

    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y,
        test_size=0.2,
        random_state=123,
        stratify=y
    )

    xtrain_scaled, xtest_scaled, scaler = scale_data(xtrain, xtest)

    model = train_mlp_classifier(xtrain_scaled, ytrain)

    accuracy, cm, ypred = evaluate_model(model, xtest_scaled, ytest)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(cm)

    save_model(model, "models/ucla_mlp_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(X.columns.tolist(), "models/columns.pkl")

    save_loss_curve(model)

    print("Saved model, scaler, columns, and loss curve successfully.")


if __name__ == "__main__":
    main()