import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def get_predictions(df):
    x_train, x_test, y_train, y_test = train_test_split(df["X"], df["Y"], test_size = 0.3)
    print(x_test)
    print(y_test)
    print(type(x_test))
    print(type(y_test))

    knn = KNeighborsClassifier().fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)
    print(y_pred_knn)


    #bayes = GaussianNB().fit(x_train, y_train)
    #dt = DecisionTreeClassifier().fit(x_train, y_train)
    #rf = RandomForestClassifier().fit(x_train, y_train)
    #svc = make_pipeline(StandardScaler(), SVC()).fit(x_train, y_train)
    #mpl = MLPClassifier().fit(x_train, y_train)

    return

def main():
    #get_predictions(pd.read_csv("./input/dataset_1.csv", sep = ";"))
    #get_predictions(pd.read_csv("./input/dataset_2.csv", sep = ";"))
    get_predictions(pd.read_csv("./input/dataset_3.csv", sep = ";"))

if __name__ == "__main__":
    main()
