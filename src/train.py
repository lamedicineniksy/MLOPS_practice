from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_evaluate_model():
    iris = load_iris(as_frame=True)
    x = iris.data
    y = iris.target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    

    joblib.dump(model, "models/model.joblib")
    return accuracy

if __name__ == "__main__":
    acc = train_evaluate_model()
    print(f"Model accuracy: {acc}")
