# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

import pandas as pd
import os
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

def get_folds(dataset):
   
    # keeping just image paths (im_path) and labels (Y)
    print(" - Loading dataset")
    df = pd.read_csv(f"./../data/{dataset}", sep = ";")
    df = df.drop('X', axis = 1)
    
    folds = []
    # splitting training and test folds 30 times (seeds from 0 to 29)
    for i in range(0, 30):
        print(" - Running for seed:", i)
        x_train, x_test, y_train, y_test = train_test_split(df.drop("Y", axis = 1), df["Y"], test_size = 0.3, 
            shuffle = True, stratify = df["Y"], random_state=i)
     
        df_training = pd.concat([x_train, y_train], axis = 1)
        df_training["Fold"] = "Training"

        df_test    = pd.concat([x_test, y_test], axis = 1)
        df_test["Fold"] = "Test"

        df_repetition = pd.concat([df_training, df_test], axis = 0)
        df_repetition["Seed"] = i
        folds.append(df_repetition)
    
    # merging all the repetitions in a single data frame
    print(" - Generating a single dataset")
    folds = pd.concat(folds, axis = 0)
    return(folds)

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    dataset = "dataset_1.csv"
    print(dataset.replace(".csv", ""))
    folds = get_folds(dataset)
    print(" - Exporting a file")
    folds.to_csv("./../data/folds_coffee_dataset.csv", index = False)

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
