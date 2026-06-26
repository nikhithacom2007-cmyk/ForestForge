"""
Test ForestForge Gradient Boosting Classifier
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from forestforge import GradientBoostingClassifier


def main():

    ####################################################
    # LOAD DATASET
    ####################################################

    iris = load_iris()

    X = iris.data
    y = iris.target

    ####################################################
    # TRAIN TEST SPLIT
    ####################################################

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )

    ####################################################
    # CREATE MODEL
    ####################################################

    model = GradientBoostingClassifier(

        n_estimators=25,

        learning_rate=0.1,

        max_depth=3,

        random_state=42

    )

    ####################################################
    # TRAIN
    ####################################################

    model.fit(

        X_train,

        y_train

    )

    ####################################################
    # PREDICT
    ####################################################

    predictions = model.predict(

        X_test

    )

    ####################################################
    # SCORE
    ####################################################

    accuracy = model.score(

        X_test,

        y_test

    )

    ####################################################
    # RESULTS
    ####################################################

    print("=" * 60)

    print("ForestForge Gradient Boosting")

    print("=" * 60)

    print()

    print(model)

    print()

    print("Weak Learners :", len(model))

    print()

    print("Accuracy :", round(accuracy, 4))

    print()

    print("First 10 Predictions")

    print(predictions[:10])

    print()

    print("First 10 Actual Labels")

    print(y_test[:10])

    print()

    print("=" * 60)


if __name__ == "__main__":

    main()