from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from forestforge.tree.decision_tree import DecisionTreeClassifier


# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = DecisionTreeClassifier(
    max_depth=5,
    criterion="gini"
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = model.score(X_test, y_test)

print("=" * 50)
print("ForestForge Decision Tree")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")

print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y_test)

print("\nTree Structure:\n")
model.print_tree()