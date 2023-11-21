# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
import os

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
    if os.path.exists(f"./../results/performances_{dataset}"):
        print("Results already exists. Skipping this dataset.")
        return # nothing returned
    else:
        print("Running for the first time.")

    # ---------------------------------------------------------
    # Creating dataset using pandas
    # ---------------------------------------------------------
    
    df = pd.read_csv(f"./../data/{dataset}", sep = ";")
    for i, row in df.iterrows():
        if dataset == "dataset_1.csv":
            df.at[i, "X"] = [int(x) for x in row["X"].strip("[]").split(",")]
        else:
            df.at[i, "X"] = [float(x) for x in row["X"].strip("[]").split(",")]
    df2 = pd.concat([df.pop("X").apply(pd.Series), df["Y"]], axis = 1)
    df_full = pd.concat([df[['im_path']], df2], axis = 1)

    # ---------------------------------------------------------
    # Defining all the ML classifier that we are going to test
    # ---------------------------------------------------------

    dict_classifiers = {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
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
    # for i in range(0, 30):
    for i in range(0,2): #for debug    
        print("############################")
        print(" * Running for seed = ", i)
        print("############################")
      
        # Performing a stratified holdout with 70/30, using i as the seed
        x_train, x_test, y_train, y_test = train_test_split(df_full.drop("Y", axis = 1), 
            df_full["Y"], test_size = 0.3, shuffle = True, stratify =  df_full["Y"], 
            random_state=i)
      
        performances = []
        all_predictions = []
      
        # Training algorithms
        for model, model_instantiation in dict_classifiers.items():
            print(" - Training: ", model)
            true_model = model_instantiation.fit(x_train.drop("im_path", axis = 1), y_train)
            predictions = true_model.predict(x_test.drop("im_path", axis = 1))
            preds = pd.DataFrame(predictions, index = x_test.index)
            preds = preds.rename(columns={0: 'predictions'})

            # creating a data frame with [img_path, seed, Y, prediction, algo]
            df_predictions = pd.concat([x_test["im_path"], y_test, preds], axis = 1)
            df_predictions['algo'] = model

            all_predictions.append(df_predictions)
            acc = accuracy_score(y_test, predictions)
            bac = balanced_accuracy_score(y_test, predictions)
            f1s = f1_score(y_test, predictions)
            print("acc = ", acc)
            print("bac = ", bac)
            print("f1c = ", f1s)
            print("----------------------------")
            performances.append([acc, bac, f1s, i])
      
        # Binding results of the repetition (seed)
        df_perf = pd.DataFrame(performances, columns=["acc", "bac", "f1s", "seed"])
        df_perf['Algo'] =  dict_classifiers.keys()
        perf_reps.append(df_perf)

        # combining all the algorithm's df (with the predictions) - a seed data frame
        complete_predictions = pd.concat(all_predictions, axis = 0)
        complete_predictions['seed'] = i
        pred_reps.append(complete_predictions)
    
    # ---------------------------------------------------------
    #Binding all predictions
    # ---------------------------------------------------------

    # combine all seeds predictions
    pred_results = pd.concat(pred_reps, axis = 0)
    pred_results.to_csv(f"./../results/predictions_{dataset}", index = False)

    # ---------------------------------------------------------
    # Binding all peformances
    # ---------------------------------------------------------
    perf_results = pd.concat(perf_reps, axis = 0)
    perf_results.to_csv(f"./../results/performances_{dataset}", index = False)

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    #for dataset in ["dataset_1.csv", "dataset_2.csv", "dataset_3.csv"]:
    for dataset in ["dataset_3.csv"]:
        print(dataset.replace(".csv", ""))
        get_predictions(dataset)
        print("-------------------------------------------")

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------