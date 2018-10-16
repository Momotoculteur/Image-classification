# IMPORT
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import load_model


"""
# Classe permettant de génerer une matrice de confusion à partir d'un dataset de test et d'un modèle entrainé
# au préalable
"""


def generateMatrix(model, datasetTestPath, imageSize, destinationMatrix):
	"""
	# Fonction qui va construire notre matrice de confusion
	:param model: chemin du modèle à charger pour realiser la prediction
	:param datasetTestPath: chemin du dataset contenant nos images de test
	:param imageSize: definit la taille de l'ensemble de nos images
	:param destinationMatrix: définit le chemin ou va être sauvegardé notre matrice sous format d'image
	:return:
	"""

	#Les tableaux contenanrt les predictions
	y_true = []
	y_pred = []

	total = 0
	success = 0
	index = 0

	print('\nEvaluation :')
	#On parcours notre dataset de test
	for root, dirs, files in os.walk(datasetTestPath):
		for mydir in dirs:
			for sample in tqdm(os.listdir(root + '\\' + mydir), "Prediction de la classe '{}'".format(mydir)):

				sample_path = root + '\\' + mydir + '\\' + sample
				#Chargement et traitement de l'image
				img = Image.open(sample_path)
				img.load()
				img = img.resize(size=imageSize)
				img = np.asarray(img) / 255.
				#On reshape pour etre de la forme (nbImage,hauteurImage,largeurImage,nbCanaux)
				img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
				#Prediction de notre modele
				pred = np.argmax(model.predict(img))

				total += 1
				if pred == index:
					success += 1

				y_true.append(index)
				y_pred.append(pred)

			index += 1

	#Precision de notre modele sur notre jeu de test en entier
	accuracy = (success / total) * 100.
	print('\nPrecision : {0:.3f}%'.format(accuracy))


	cnf_matrix = confusion_matrix(y_true, y_pred)
	np.set_printoptions(precision=2)

	# Plot normalized confusion matrix
	plt.figure()

	cmap = plt.cm.Blues
	classes = ['marguerite', 'pissenlit', 'rose', 'tournesol', 'tulipe']
	title = 'Confusion matrix'

	cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

	#Legende de notre matrice
	plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f'
	thresh = cnf_matrix.max() / 2.
	for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
		plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center", color="white" if cnf_matrix[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()

	#On sauvegarde notre matrice en image
	plt.savefig(destinationMatrix + '\\' + 'confusionMatrix')


def main():
	"""
	# Fonction main
	"""

	#On definit les chemins de nos divers ressources
	modelPath = '.\\trainedModel\\moModel.hdf5'
	datasetTestPath = '.\\datasetTest'
	destinationMatrix = '.\\graph'
	imageSize = (50, 50)
	model = load_model(modelPath)

	generateMatrix(model, datasetTestPath, imageSize, destinationMatrix)


if __name__ == "__main__":
	"""
	# MAIN
	"""
	main()
