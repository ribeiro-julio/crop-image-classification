# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# handling data frames
import pandas as pd

# training and testing splits
from sklearn.model_selection import train_test_split

# ML algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# evaluation metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

def get_predictions(dataset):
    
    # ---------------------------------------------------------
    # TODO: Running just if it was not executed before
    # ---------------------------------------------------------



    # ---------------------------------------------------------
    # Creating dataset using pandas
    # ---------------------------------------------------------
    
    df = pd.read_csv(f"./../data/{dataset}", sep = ";")
    for i, row in df.iterrows():
        if dataset == "dataset_1.csv":
            df.at[i, "X"] = [int(x) for x in row["X"].strip("[]").split(",")]
        else:
            df.at[i, "X"] = [float(x) for x in row["X"].strip("[]").split(",")]
    df = pd.concat([df.pop("X").apply(pd.Series), df["Y"]], axis = 1)

    # ---------------------------------------------------------
    # Defining all the ML classifier that we are going to test
    # ---------------------------------------------------------

    dict_classifiers = {
        "KNN": KNeighborsClassifier(),
        "NB": GaussianNB(),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(),
        "SVM": make_pipeline(StandardScaler(), SVC()),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier()),
    }
    print(dict_classifiers)

    # ---------------------------------------------------------
    # Repeating 30 times for each dataset
    # ---------------------------------------------------------
    
    perf_reps = []
    pred_reps = []

    # TODO: paralellize into different threads


    for i in range(0, 30):
        print("############################")
        print(" * Running for seed = ", i)
        print("############################")
      
        # Performing a stratified holdout with 70/30, using i as the seed
        x_train, x_test, y_train, y_test = train_test_split(df.drop("Y", axis = 1), df["Y"], 
            test_size = 0.3, shuffle = True, stratify =  df["Y"], random_state=i)
      
        performances = []
        all_predictions = []
      
        # Training algorithms
        for model, model_instantiation in dict_classifiers.items():
            print(" - Training: ", model)
            true_model = model_instantiation.fit(x_train, y_train)
            preditctions = true_model.predict(x_test)
            all_predictions.append(preditctions)
            acc = accuracy_score(y_test, preditctions)
            bac = balanced_accuracy_score(y_test, preditctions)
            f1s = f1_score(y_test, preditctions)
            print("acc = ", acc)
            print("bac = ", bac)
            print("f1c = ", f1s)
            print("----------------------------")
            performances.append([acc, bac, f1s, i])
      
        # Binding results of the repetition (seed)

        df_perf = pd.DataFrame(performances, columns=["acc", "bac", "f1s", "seed"])
        df_perf['Algo'] =  dict_classifiers.keys()
        perf_reps.append(df_perf)
        df_predictions = pd.DataFrame(all_predictions)
        df_predictions['Seed'] =  i
        df_predictions['Algo'] =  dict_classifiers.keys()
        pred_reps.append(df_predictions)
    
    # ---------------------------------------------------------
    #Binding all predictions
    # ---------------------------------------------------------

    pred_results = pd.concat(pred_reps, axis = 0)
    pred_results.to_csv(f"./../results/predictions_{dataset}", index = False)

    # ---------------------------------------------------------
    # Binding all peformances
    # ---------------------------------------------------------
    perf_results = pd.concat(perf_reps, axis = 0)
    perf_results.to_csv(f"./../results/performances_{dataset}", index = False)

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

def main():
    for dataset in ["dataset_1.csv", "dataset_2.csv", "dataset_3.csv"]:
        print(dataset.replace(".csv", ""))
        get_predictions(dataset)
        print("-------------------------------------------")

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------