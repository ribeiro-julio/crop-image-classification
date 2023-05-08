# ---------------------------------------------
# ---------------------------------------------

print("Doing imports")

from PIL import Image
import random
from statistics import mean
import pandas as pd

dataset_2 = []
classes = []

print(" - Loading images")
desc_file = open("./dataset-brazilian_coffee_scenes/desc.txt")
desc = desc_file.read().split("\n")
desc_file.close()

print(" - Extracting values and classes")
for image in desc:
    print(image)
    if "noncoffee" in image:
        classes.append(0)
        im = Image.open("./dataset-brazilian_coffee_scenes/images/" + image.replace("noncoffee.", "") + ".jpg")
    elif "coffee" in image:
        classes.append(1)
        im = Image.open("./dataset-brazilian_coffee_scenes/images/" + image.replace("coffee.", "") + ".jpg")

    im_data = []
    for pixel_data in list(im.getdata()):
        im_data.append(pixel_data[0]) # green 
        im_data.append(pixel_data[1]) # red values
        im_data.append(pixel_data[2])

        #ndvi values        
        try:
            im_data.append(((pixel_data[2]-pixel_data[1])/(pixel_data[2]+pixel_data[1])) + 1)
        except:
            im_data.append(0)

    dataset_2.append(im_data)
 
print(" - Creating dataframe")
df = pd.DataFrame(data = dataset_2)
df['Target'] = classes
print(df.head())
df.to_csv('./data/dataset2.csv')