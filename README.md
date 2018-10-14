#Infos
Code source du tutoriel de http://deeplylearning.fr
Permet de creer avec Tensorflow et Keras une reconnaissance d'image entre 5 types de fleurs différentes, avec des algorithmes de deep learning.

## Installer les pré-requis
`$ install.bat`

## Generer les tableaux numpy
Changer vos chemins du dataset d'image si besoin

| Attribut | Description                    |
| ------------- | ------------------------------ |
| `pathNumpy`      | Destination ou seront sauvegarder les tableaux      |
| `pathData`   | Chemin ou sont les images en format png    |
| `imgResize`   | On met l'ensemble du dataset à la même taille    |
Lancer la commande suivante :
`$ python generateNumpyFiles.py`

## Entrainer le model
| Attribut | Description                    |
| ------------- | ------------------------------ |
| `csv_logger`      | Chemin du callbak permettant l'enregistrement des metriques      |
| `check`   | Chemin du callback permettant d'enregistrer le modèle sous format hdf5    |
| `pathData`   | Chemin des tableaux numpy   |
| `trainRatio`   | Ratio définissant la taille du jeu d'entrainemnt et de validation   |
| `batch_size`   | Nombre d'item que on envoi sur une phase de feedforward/backpropagation   |
| `earlyStopPatience`   | Permet de définir l'arrêt de l'entrainement, lorsque les données de précision sur le jeu de validation n'évolu plus  |
Lancer la commande suivante :
`$ python trainModel.py`

## Generer les graphiques de suivi de métriques
Lancer la commande suivante :
`$ python generateMetrics.py`

## Generer la matrice de confusion
Lancer la commande suivante :
`$ python generateConfusionMatrix.py`

## Realiser une prédiction sur une nouvelle donnée
| Attribut | Description                    |
| ------------- | ------------------------------ |
| `modelPath`      | Chemin du model au format hdf5 pour le charger en memoire      |
| `imagePath`   | Chemin de l'image que l'on doit predire    |
| `imageResize`   | Doit être identique aux tailles d'image d'entrainement    |

Puis lancer la commande suivante :
`$ python autoPredict.py`