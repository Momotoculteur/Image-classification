# IMPORT
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

"""
# Classe permettant de convertir notre dataset d'images en tableaux Numpy
"""


def launchConversion(pathData, pathNumpy, resizeImg, imgSize):
    """
    # Permet de lancer la conversion des images en tableau numpy
    :param pathData: chemin ou sont les
    :param pathNumpy:
    :param resizeImg:
    :param imgSize:
    """

    #Pour chaque classe
    for flowerClasse in os.listdir(pathData):
        pathFlower = pathData + '\\' + flowerClasse
        imgs = []

        #Pour chaque image d'une classe, on la charge, resize et transforme en tableau
        for imgFlower in tqdm(os.listdir(pathFlower), "Conversion de la classe : '{}'".format(flowerClasse)):
            imgFlowerPath = pathFlower + '\\' + imgFlower
            img = Image.open(imgFlowerPath)
            img.load()
            if resizeImg == True:
                img = img.resize(size=imgSize)

            data = np.asarray(img, dtype=np.float32)
            imgs.append(data)

        #Converti les gradients de pixels (allant de 0 à 255) vers des gradients compris entre 0 et 1
        imgs = np.asarray(imgs) / 255.

        #Enregistre une classe entiere en un fichier numpy
        np.save(pathNumpy + '\\ ' + flowerClasse + '.npy', imgs)


def main():
    """
    # Fonction main
    """

    pathNumpy = '.\\numpy'
    pathData = '.\\dataset'
    resizeImg = True
    imgSize = (50, 50)
    launchConversion(pathData, pathNumpy, resizeImg, imgSize)


if __name__ == '__main__':
    """
    # MAIN
    """
    main()

