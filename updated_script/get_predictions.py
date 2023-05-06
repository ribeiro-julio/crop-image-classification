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
from sklearn import metrics

def get_predictions(dataset):
    df = pd.read_csv(f"./input/{dataset}", sep = ";")
    for i, row in df.iterrows():
        if dataset == "dataset_1.csv":
            df.at[i, "X"] = [int(x) for x in row["X"].strip("[]").split(",")]
        else:
            df.at[i, "X"] = [float(x) for x in row["X"].strip("[]").split(",")]
    df = pd.concat([df.pop("X").apply(pd.Series), df["Y"]], axis = 1)

    x_train, x_test, y_train, y_test = train_test_split(df.drop("Y", axis = 1), df["Y"], test_size = 0.3)

    knn = KNeighborsClassifier().fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)
    print("knn", metrics.accuracy_score(y_test, y_pred_knn))

    bayes = GaussianNB().fit(x_train, y_train)
    y_pred_bayes = bayes.predict(x_test)
    print("bayes", metrics.accuracy_score(y_test, y_pred_bayes))

    dt = DecisionTreeClassifier().fit(x_train, y_train)
    y_pred_dt = dt.predict(x_test)
    print("dt", metrics.accuracy_score(y_test, y_pred_dt))
    
    rf = RandomForestClassifier().fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    print("rf", metrics.accuracy_score(y_test, y_pred_rf))

    svc = make_pipeline(StandardScaler(), SVC()).fit(x_train, y_train)
    y_pred_svc = svc.predict(x_test)
    print("svc", metrics.accuracy_score(y_test, y_pred_svc))

    mpl = MLPClassifier().fit(x_train, y_train)
    y_pred_mpl = mpl.predict(x_test)
    print("mpl", metrics.accuracy_score(y_test, y_pred_mpl))

    out_df = pd.DataFrame({
        "df_index": x_test.index,
        "Y": y_test,
        "Y_pred_knn": y_pred_knn,
        "Y_pred_bayes": y_pred_bayes,
        "Y_pred_dt": y_pred_dt,
        "Y_pred_rf": y_pred_rf,
        "Y_pred_svc": y_pred_svc,
        "Y_pred_mpl": y_pred_mpl
    })
    out_df.to_csv(f"./output/predictions_{dataset}", index = False)

    return

def main():
    for dataset in ["dataset_1.csv", "dataset_2.csv", "dataset_3.csv"]:
        print(dataset.replace(".csv", ""))
        get_predictions(dataset)
        print("-------------------------------------------")

if __name__ == "__main__":
    main()
