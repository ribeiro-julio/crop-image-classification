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

def get_accuracies(dataset):
    df = pd.read_csv(f"./input/{dataset}", sep = ";")
    for i, row in df.iterrows():
        if dataset == "dataset_1.csv":
            df.at[i, "X"] = [int(x) for x in row["X"].strip("[]").split(",")]
        else:
            df.at[i, "X"] = [float(x) for x in row["X"].strip("[]").split(",")]
    df = pd.concat([df.pop("X").apply(pd.Series), df["Y"]], axis = 1)
    
    knn_acc = []
    bayes_acc = []
    dt_acc = []
    rf_acc = []
    svc_acc = []
    mpl_acc = []
    for i in range(0, 30):
        x_train, x_test, y_train, y_test = train_test_split(df.drop("Y", axis = 1), df["Y"], test_size = 0.3, shuffle = True)
        
        knn = KNeighborsClassifier().fit(x_train, y_train)
        y_pred_knn = knn.predict(x_test)
        knn_acc.append(metrics.accuracy_score(y_test, y_pred_knn))

        bayes = GaussianNB().fit(x_train, y_train)
        y_pred_bayes = bayes.predict(x_test)
        bayes_acc.append(metrics.accuracy_score(y_test, y_pred_bayes))

        dt = DecisionTreeClassifier().fit(x_train, y_train)
        y_pred_dt = dt.predict(x_test)
        dt_acc.append(metrics.accuracy_score(y_test, y_pred_dt))
        
        rf = RandomForestClassifier().fit(x_train, y_train)
        y_pred_rf = rf.predict(x_test)
        rf_acc.append(metrics.accuracy_score(y_test, y_pred_rf))

        svc = make_pipeline(StandardScaler(), SVC()).fit(x_train, y_train)
        y_pred_svc = svc.predict(x_test)
        svc_acc.append(metrics.accuracy_score(y_test, y_pred_svc))

        mpl = MLPClassifier().fit(x_train, y_train)
        y_pred_mpl = mpl.predict(x_test)
        mpl_acc.append(metrics.accuracy_score(y_test, y_pred_mpl))

        print("{:.2f}".format(i/30*100), " %\r", end = "")
    
    print()

    out_df = pd.DataFrame({
        "knn": knn_acc,
        "nb": bayes_acc,
        "dt": dt_acc,
        "rf": rf_acc,
        "svc": svc_acc,
        "mpl": mpl_acc,
    })
    out_df.to_csv(f"./output/accuracies_{dataset}", index = False)
    
    return

def main():
    for dataset in ["dataset_1.csv", "dataset_2.csv", "dataset_3.csv"]:
        print(dataset.replace(".csv", ""))
        get_accuracies(dataset)
        print("-------------------------------------------")

if __name__ == "__main__":
    main()
