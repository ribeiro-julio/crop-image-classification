# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

import os 
import pandas as pd
import numpy as np
import keras

from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, BatchNormalization, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# TODO: see this
# https://www.kaggle.com/code/raulcsimpetru/vgg16-binary-classification

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

def get_VGG16_model_Keras(input_shape=(64,64,3)) :

	VGGmodel = Sequential() 
	baseModel = VGG16(
		input_shape=input_shape,
		weights='imagenet',
		include_top=False,
	)
	baseModel.trainable = False
	VGGmodel.add(baseModel)
	VGGmodel.add(Flatten()) 
	VGGmodel.add(Dense(4096,activation = 'relu'))
	VGGmodel.add(Dense(1,activation = 'sigmoid'))

	return(VGGmodel)


# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

def readImagesFromDF(df, seed):

    # Selecting training and testing based on the seed value
    df_seed = df.loc[df['Seed'] == seed]
    df_training = df_seed[df_seed["Fold"] == "Training"]
    df_testing  = df_seed[df_seed["Fold"] == "Test"]
	    
    # ----------------------------
    # Loading training images
    # ----------------------------
    print(" - loading training images")
    train_images = []
    for i in range(0, len(df_training)):
    	example = df_training.iloc[i]
    	img = Image.open(example["im_path"]) #.getdata()
    	train_images.append(np.array(img))
    train_images = np.array(train_images)
    # >>> train_images .shape
    # (2013, 64, 64, 3)
	    
    # ----------------------------
    # Loading testing images
    # ----------------------------
    print(" - loading testing images")
    test_images = []
    for i in range(0, len(df_testing)):
    	example = df_testing.iloc[i]
    	img = Image.open(example["im_path"]) #.getdata()
    	test_images.append(np.array(img))
    test_images = np.array(test_images)
    # >>> test_images.shape
    # (863, 64, 64, 3)

    # ----------------------------
    # Normalize pixel values to be between 0 and 1
    # ----------------------------
    train_images, test_images = train_images / 255.0, test_images / 255.0
	    
    # ----------------------------
    # Labels from training and testing folds
    # ----------------------------
    train_labels = np.array(df_training["Y"])
    test_labels  = np.array(df_testing["Y"])
    return train_images, train_labels, test_images, test_labels 

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

def trainVGG16(df):

	all_performances = []
	all_predictions  = []

	for seed in range(0, 30):#2):

	    print("############################")
	    print(" * Running for seed = ", seed)
	    print("############################")
	    # ----------------------------
	    # Set the seed using keras.utils.set_random_seed. This will set:
	    # 1) `numpy` seed
	    # 2) `tensorflow` random seed
	    # 3) `python` random seed
	    # ----------------------------
	    keras.utils.set_random_seed(seed)
	    train_images, train_labels, test_images, test_labels = readImagesFromDF(df=df, seed=seed)

	    # ----------------------------
	    # Defining the prediction model (CNN)
	    # ----------------------------

	    print(" - defining VGG16 model")
	    model = get_VGG16_model_Keras(input_shape=(64,64,3))
	    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['binary_accuracy', 'accuracy'])

	    # ----------------------------
	    # Traninig the algorithm
	    # ----------------------------

	    # Callbacks
	    early_stopper = EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=1)
	    csv_logger    = CSVLogger(f"./../results/vgg16Logs/log_history_vgg16_seed_{seed}.csv", separator=",", append=False)

	    print(" - training VGG16")
	    history  = model.fit(train_images, train_labels, epochs=100, validation_split=0.3,
	    	batch_size=16, callbacks=[early_stopper, csv_logger]) 

	    # ----------------------------
	    # Evaluating predictions
	    # ----------------------------
	    print(" - Evaluating VGG16")
	    predictions = model.predict(test_images)
	    rounded_predictions = np.round(predictions)

		# ----------------------------
	    # adding predictions to a data frame
	    # ----------------------------
	    preds   = pd.DataFrame(rounded_predictions, index = df_testing.index)
	    preds   = preds.rename(columns={0: 'predictions'})
	    df_pred = pd.concat([df_testing, preds], axis = 1) # by column

	    # ----------------------------
	    # evaluating with scikit learn metrics
	    # ----------------------------
	    acc = accuracy_score(test_labels, rounded_predictions)
	    bac = balanced_accuracy_score(test_labels, rounded_predictions)
	    f1s = f1_score(test_labels, rounded_predictions)
	    print("acc = ", acc)
	    print("bac = ", bac)
	    print("f1c = ", f1s)
	    print("----------------------------")
	    all_performances.append([acc, bac, f1s, seed])
	    all_predictions.append(df_pred)

	# ---------------------------------------------------------
	# Binding all predictions and performances
	# ---------------------------------------------------------
	pred_results = pd.concat(all_predictions, axis = 0) # by row
	pred_results[['algo']] = "VGG16"

	perf_results = pd.DataFrame(all_performances, columns=["acc", "bac", "f1s", "seed"])
	return (pred_results, perf_results)

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

	path = "./../results/vgg16Logs/"
	if not os.path.exists(path):
		os.makedirs(path)
		print("The new directory is created!")
	else:
		print("vgg16 folder already exists!")

	df = pd.read_csv(f"./../data/folds_coffee_dataset.csv", sep = ",")

	(pred_results, perf_results) = trainVGG16(df=df)
	
	perf_results.to_csv("./../results/performances_vgg16.csv", index = False)
	pred_results.to_csv("./../results/predictions_vgg16.csv",  index = False)
	print("Done!")

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
