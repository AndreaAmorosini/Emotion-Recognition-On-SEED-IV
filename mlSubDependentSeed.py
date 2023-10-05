from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import featureMatrixSeed
import numpy as np
import pandas as pd

import MlAlgs

#Labels per ogni trial in ogni sessione
session1Labels = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2Labels = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3Labels = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
labels_dict = {0: "Neutral", 1: "Sad", 2: "Fear", 3: "Happy"}

#Sceglie le trial da usare per il testing per ogni sessione e crea una lista di labels corrispondenti per training e test set
trialTestSess1 = [0,1,2,3,4,5,7,15]
trialTestSess1Labels = []
trialTrainSess1Labels = session1Labels.copy()
for i in trialTestSess1:
    trialTestSess1Labels.append(session1Labels[i])
    trialTrainSess1Labels.remove(session1Labels[i])
trialTestSess2 = [0,1,2,3,4,5,8,14]
trialTestSess2Labels = []
trialTrainSess2Labels = session2Labels.copy()
for i in trialTestSess2:
    trialTestSess2Labels.append(session2Labels[i])
    trialTrainSess2Labels.remove(session2Labels[i])
trialTestSess3 = [0, 1, 2, 3, 4, 5, 11, 15]
trialTestSess3Labels = []
trialTrainSess3Labels = session3Labels.copy()
for i in trialTestSess3:
    trialTestSess3Labels.append(session3Labels[i])
    trialTrainSess3Labels.remove(session3Labels[i])

#Crea un unica lista di labels per training e test set
trainLabels = trialTrainSess1Labels * 15 + trialTrainSess2Labels * 15 + trialTrainSess3Labels * 15
testLabels = trialTestSess1Labels * 15 + trialTestSess2Labels * 15 + trialTestSess3Labels * 15

fm_all = []
fmt_all = []

#Per ogni sessione
for s in range(1, 4):
    #Sceglie le trial da utilizzare in base alla sessione
    if s == 1:
        trial_test_sess = trialTestSess1
    elif s == 2:
        trial_test_sess = trialTestSess2
    else:
        trial_test_sess = trialTestSess3

    #EEG Signals
    #Carica i path dei file .mat per la sessione
    #in caso non si stia usando il dataset, commentare la riga
    paths = featureMatrixSeed.load_trial_paths_by_session(s)

    #Crea la feature matrix di train e di test per la sessione
    # Per creare la feature matrix per segnali EEG da dati differenti dal dataset,
    # Basta passare alla funzione di creazione della feature matrix una lista contenente i path dei file .mat desiderati

    # feature_matrix_train, feature_matrix_test = featureMatrixSeed.create_feature_matrix_subDep_split(paths, trial_test_sess)


    #Eye Signals
    #Carica i path dei file .mat per la sessione
    #In caso non si stia usando il dataset, commentare la riga
    path_session_eye = featureMatrixSeed.load_trial_eye_data_paths_by_session(s)

    #Crea la feature matrix di train e di test per la sessione (solo segnali oculari)
    # Per creare la feature matrix per segnali EEG da dati differenti dal dataset,
    # Basta passare alla funzione di creazione della feature matrix una lista contenente i path dei file .mat desiderati

    # feature_matrix_train, feature_matrix_test = featureMatrixSeed.create_feature_matrix_subDep_eye_data_split(path_session_eye, trial_test_sess)


    #EEG+Eye
    #Unisce le feature matrix di segnali EEG e oculari
    feature_matrix_train, feature_matrix_test = featureMatrixSeed.create_feature_matrix_subDep_eeg_and_eye_split(paths, path_session_eye, trial_test_sess)
    print(len(feature_matrix_train))

    #Aggiunge le feature matrix di ogni sessione alla lista totale
    for i in range(0, len(feature_matrix_train)):
        fm_all.append(feature_matrix_train[i])
    for i in range(0, len(feature_matrix_test)):
        fmt_all.append(feature_matrix_test[i])


df_train = pd.DataFrame(fm_all)
df_test = pd.DataFrame(fmt_all)

#Serve a rimuovere i valori NaN dovuti a una differenza di tempistiche tra i vari trial
df_train = df_train.replace(np.nan, 0)
df_test = df_test.replace(np.nan, 0)

#Inizializza lo scaler per normalizzare i dati tra 0 e 1
scaler = MinMaxScaler(feature_range=(0, 1))
#Allena lo scaler e lo applica sui dati di train e di test
df_train = scaler.fit_transform(df_train)
df_test = scaler.fit_transform(df_test)


#Train Data
#inizializza la PCA per la riduzione delle feature matrix in 80 componenti
pca = PCA(n_components=80, whiten = True)
#Allena la PCA sui dati di train e applica la riduzione delle dimensioni
pca.fit(df_train)
df = pd.DataFrame(pca.transform(df_train))
# print("explained variance ratio (Train data): ", pca.explained_variance_ratio_)
# print("Preserved Variance (Train data): ", sum(pca.explained_variance_ratio_))

#Test Data
#Allena la PCA sui dati di test e applica la riduzione delle dimensioni
pca.fit(df_test)
dfTest = pd.DataFrame(pca.transform(df_test))
# print("explained variance ratio (Test Data): ", pca.explained_variance_ratio_)
# print("Preserved Variance (Test Data): ", sum(pca.explained_variance_ratio_))


print("Train Data")
print(df)

print("\nTest Data")
print(dfTest)

#Inizializza la lista di labels per il training e il test set
x_train = df
y_train = trainLabels
x_test = dfTest
y_test = testLabels

#Specifica la strategia di validazione per il log dei dati su Optuna
strategy = "Subject Dependent"

#Metodi per l'applicazione di uno dei modelli
#Basta decommentare quella desiderata e avviare il programma

# MlAlgs.tensor_flow_test_model(np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), strategy)
# MlAlgs.svm_test_model(np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), strategy)
# MlAlgs.gradient_boosting_early_stopping_test_model(np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), strategy)
# MlAlgs.adaboost_test_model(np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), strategy)
# MlAlgs.random_forest_test_model(np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), strategy)
# MlAlgs.tensorflow_cnn_model(np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), strategy)


