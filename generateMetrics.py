import matplotlib.pyplot as plt
import pandas as pd



#Fonction permettant de creer nos graph de suivi de metriques
def displayGraph(pathLog,pathSaveGraph):
    data = pd.read_csv(pathLog)
    # split into input (X) and output (Y) variables
    plot(data['epoch'], data['acc'], data['val_acc'], 'TRAIN_VAL_Accuracy', 'Epoch', 'Accuracy', 'upper left',pathSaveGraph)
    plot(data['epoch'], data['loss'], data['val_loss'], 'TRAIN_VAL_Loss', 'Epoch', 'Loss', 'upper left',pathSaveGraph)

#Fonction d'affichage de graph
def plot(X, Y, Y2, title, xLabel, yLabel, legendLoc, pathSaveGraph):
   #On trace nos differentes courbes
    plt.plot(Y)
    plt.plot(Y2)
   #titre du graph, legende...
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend(['train', 'val'], loc=legendLoc)
   #Pour avoir un courbe propre qui demarre à 0
    plt.xlim(xmin=0.0, xmax=max(X))
    plt.savefig(pathSaveGraph +'\\' + title)
    plt.figure()
    #plt.show()


def main():
    #Definition des chemins d'acces a notre fichier log
    pathLogs = '.\\logs\\log_moModel.csv'
    pathSaveGraph = '.\\graph'
    displayGraph(pathLogs,pathSaveGraph)



# MAIN
if __name__ == "__main__":
    main()