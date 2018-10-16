#IMPORT
from keras.models import load_model
from PIL import Image
import numpy as np
import time


"""
# Classe permettant de réaliser une prédiction sur une nouvelle donnée
"""


def main():
    """
    # On definit les chemins d'acces au différentes hyper parametre
    """

    modelPath = '.\\trainedModel\\moModel.hdf5'
    imagePath =  '.\\testImage\\rose.jpg'
    imageSize = (50,50)
    label = ['marguerite', 'pissenlit', 'rose', 'tournesol', 'tulipe']

    predict(modelPath, imagePath,imageSize, label)


def predict(modelPath,imagePath, imageSize, label):
    """
    # Fonction qui permet de convertir une image en array, de charger le modele et de lui injecter notre image pour une prediction
    :param modelPath: chemin du modèle au format hdf5
    :param imagePath: chemin de l'image pour realiser une prediction
    :param imageSize: défini la taille de l'image. IMPORTANT : doit être de la même taille que celle des images
    du dataset d'entrainements
    :param label: nom de nos 5 classes de sortie
    """

    start = time.time()

    # Chargement du modele
    print("Chargement du modèle :\n")
    model = load_model(modelPath)
    print("\nModel chargé.")

    #Chargement de notre image et traitement
    data = []
    img = Image.open(imagePath)
    img.load()
    img = img.resize(size=imageSize)
    img = np.asarray(img) / 255.
    data.append(img)
    data = np.asarray(data)

    #On reshape pour correspondre aux dimensions de notre modele
    # Arg1 : correspond au nombre d'image que on injecte
    # Arg2 : correspond a la largeur de l'image
    # Arg3 : correspond a la hauteur de l'image
    # Arg4 : correspond au nombre de canaux de l'image (1 grayscale, 3 couleurs)
    dimension = data[0].shape

    #Reshape pour passer de 3 à 4 dimension pour notre réseau
    data = data.astype(np.float32).reshape(data.shape[0], dimension[0], dimension[1], dimension[2])

    #On realise une prediction
    prediction = model.predict(data)


    #On recupere le numero de label qui a la plus haut prediction
    maxPredict = np.argmax(prediction)

    #On recupere le mot correspondant à l'indice precedent
    word = label[maxPredict]
    pred = prediction[0][maxPredict] * 100.
    end = time.time()


    #On affiche les prédictions
    print()
    print('----------')
    print(" Prediction :")
    for i in range(0, len(label)):
        print('     ' + label[i] + ' : ' + "{0:.2f}%".format(prediction[0][i] * 100.))

    print()
    print('RESULTAT : ' + word + ' : ' + "{0:.2f}%".format(pred))
    print('temps prediction : ' + "{0:.2f}secs".format(end-start))

    print('----------')


if __name__ == "__main__":
    """
    # MAIN
    """
    main()