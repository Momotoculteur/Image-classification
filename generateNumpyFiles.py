import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def launchConversion(pathData, pathNumpy):
    for flowerClasse in os.listdir(pathData):
        pathFlower = pathData + '\\' + flowerClasse
        imgs = []

        for imgFlower in tqdm(os.listdir(pathFlower), "Conversion de la classe : '{}'".format(flowerClasse)):
            imgFlowerPath = pathFlower + '\\' + imgFlower
            img = Image.open(imgFlowerPath)
            img.load()
            data = np.asarray(img, dtype=np.float16)
            imgs.append(data)

        imgs = np.asarray(imgs) / 255.
        np.save(pathNumpy + '\\ ' + flowerClasse + '.npy', imgs)


## MAIN ##
if __name__ == '__main__':
    pathNumpy = '.\\numpy'
    pathData = '.\\dataset'
    launchConversion(pathData, pathNumpy)
