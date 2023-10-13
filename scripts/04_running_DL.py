# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


# handling data frames
import pandas as pd

import numpy as np

from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# Reading image pathspod
df = pd.read_csv(f"./../data/folds_coffee_dataset.csv", sep = ",")

# Selecting training and testing for seed = 0
seed = 0
df_seed = df.loc[df['Seed'] == seed]

df_training = df_seed[df_seed["Fold"] == "Training"]
df_testing  = df_seed[df_seed["Fold"] == "Test"]

# ----------------------------
# Loading training images
# ----------------------------
train_images = []
for i in range(0, len(df_training)):
	example = df_training.iloc[i]
	# Loading one image 
	img = Image.open(example["im_path"]) #.getdata()
	# converting image to numpy array
	train_images.append(np.array(img))
train_images = np.array(train_images)
# >>> train_images .shape
# (2013, 64, 64, 3)

# ----------------------------
# Loading testing images
# ----------------------------
test_images = []
for i in range(0, len(df_testing)):
	example = df_testing.iloc[i]
	# Loading one image 
	img = Image.open(example["im_path"]) #.getdata()
	# converting image to numpy array
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

train_labels = np.array( df_training["Y"])
test_labels  = np.array( df_testing["Y"])

# ----------------------------
# Defining the prediction model (CNN)
# ----------------------------

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="sigmoid"))


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['binary_accuracy', 'accuracy'])

# ----------------------------
# Traninig the algorithm
# ----------------------------

history = model.fit(train_images, train_labels, epochs=100, 
                    validation_data=(test_images, test_labels))


# ----------------------------
# Evaluating predictions
# ----------------------------

test = model.evaluate(test_images, test_labels, verbose=2)

predictions = model.predict(test_images)
print(predictions)


# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
