from PIL import Image
import pandas as pd
from statistics import mean

def main():
    print("generating datasets...")

    desc_file = open("../dataset-brazilian_coffee_scenes/desc.txt")
    desc = desc_file.read().split("\n")
    desc_file.close()

    ds_1 = []
    ds_2 = []
    ds_3 = []
    for i, im_name in enumerate(desc):
        im_path = "../dataset-brazilian_coffee_scenes/images/"
        Y = 0 # noncoffee = 0; coffee = 1
        if "noncoffee" in im_name: im_path += im_name.replace("noncoffee.", "") + ".jpg"
        elif "coffee" in im_name:
            im_path += im_name.replace("coffee.", "") + ".jpg"
            Y = 1
        
        pixel_values = [[], [], [], []]
        ds_1_X = []
        ds_2_X = []
        for pixel in list(Image.open(im_path).getdata()):
            pixel = list(pixel)

            pixel_values[0].append(pixel[0]) # green
            pixel_values[1].append(pixel[1]) # red
            pixel_values[2].append(pixel[2]) # nir

            ds_1_X.extend(pixel)
            
            ndvi = 0
            try: ndvi = ((pixel[2] - pixel[1])/(pixel[2] + pixel[1])) + 1
            except: pass
            pixel.append(ndvi)
            pixel_values[3].append(ndvi)
            
            ds_2_X.extend(pixel)
        ds_3_X = [mean(pixel_values[0]), mean(pixel_values[1]), mean(pixel_values[2]), mean(pixel_values[3])]

        ds_1.append({"im_path": im_path, "Y": Y, "X": ds_1_X})
        ds_2.append({"im_path": im_path, "Y": Y, "X": ds_2_X})
        ds_3.append({"im_path": im_path, "Y": Y, "X": ds_3_X})

        print("{:.2f}".format(i/len(desc)*100), " %\r", end = ""),

    print("\nnumber of rows:", len(ds_1), len(ds_2), len(ds_3))
    print("number of X elements of first row:", len(ds_1[0]["X"]), len(ds_2[0]["X"]), len(ds_3[0]["X"]))

    print("saving datasets...")

    pd.DataFrame.from_dict(ds_1).to_csv("./input/dataset_1.csv", sep = ";", index = False)
    pd.DataFrame.from_dict(ds_2).to_csv("./input/dataset_2.csv", sep = ";", index = False)
    pd.DataFrame.from_dict(ds_3).to_csv("./input/dataset_3.csv", sep = ";", index = False)

    print("done.")

if __name__ == "__main__":
    main()
