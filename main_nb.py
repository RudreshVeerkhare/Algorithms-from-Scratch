# C:\Users\Acer\Anaconda3\Scripts\activate.bat

from utils.nb import NaiveBayes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    df = pd.read_csv("Iris.csv")
    X = df.drop(['Id', 'Species'], axis=1)
    Y = df["Species"]
    # train-test split
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=6)

    # training model
    model = NaiveBayes()
    model.fit(X_train.values, Y_train.values)
    
    # printing test data
    print("\n\nTest Data >> ")
    print(x_test)

    # predicting results
    print("\n\nPredictions >> ")
    res_df = pd.DataFrame()
    res_df['Predicted'] = model.predict(x_test.values)
    res_df['Actual'] = y_test.values
    print(res_df)

    print(f"\nAccuracy score: {accuracy_score(res_df['Predicted'], res_df['Actual'])}")

