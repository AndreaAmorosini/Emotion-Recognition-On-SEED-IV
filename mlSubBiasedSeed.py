from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import preprocseed.featureMatrixSeed as featureMatrixSeed
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import MlAlgs

#Labels per ogni trial di ogni sessione
session1Labels = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2Labels = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3Labels = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
labels_dict = {0: "Neutral", 1: "Sad", 2: "Fear", 3: "Happy"}

#Crea le label corrispondenti alla feature matrix da passare ai vari modelli
trainLabels = session1Labels * 15 + session2Labels * 15 + session3Labels * 15

#EEG
#Carica i path dei file .mat
# in caso non si stia usando il dataset, commentare la riga
path_eeg = featureMatrixSeed.load_trial_paths()

#Per creare la feature matrix per segnali EEG da dati differenti dal dataset,
#Basta passare alla funzione di creazione della feature matrix una lista contenente i path dei file .mat desiderati

#Crea la feature matrix solo per segnali EEG
feature_matrix = featureMatrixSeed.create_feature_matrix_complete(path_eeg)

#EYE
#Carica i path dei file .mat
# in caso non si stia usando il dataset, commentare la riga
path_eye = featureMatrixSeed.load_trial_eye_data_paths()

#Per creare la feature matrix per dati sul movimento oculare da dati differenti dal dataset,
#Basta passare alla funzione di creazione della feature matrix una lista contenente i path dei file .mat desiderati

#Crea la feature matrix solo per dati sul movimento oculare
# feature_matrix = featureMatrixSeed.create_feature_matrix_eye_data_complete(path_eye)

#EEG + EYE
#Unisce le feature matrix di EEG e EYE
# feature_matrix = featureMatrixSeed.create_feature_matrix_eye_and_eeg_complete(path_eeg, path_eye)

#Trasforma la feature matrix in Dataframe
df = pd.DataFrame(feature_matrix)
#Sostituisce tutti i valori NaN con 0
df = df.replace(np.nan, 0)
#inizializza lo scaler per normalizzare i dati tra 0 e 1
scaler = MinMaxScaler(feature_range=(0, 1))
#Scala i valori nella feature matrix
df = scaler.fit_transform(df)

#Inizializza la PCA con 80 componenti
pca = PCA(n_components=80, whiten = True)
#Allena la PCA sulla feature matrix
pca.fit(df)
#Trasforma la feature matrix con la PCA
df = pd.DataFrame(pca.transform(df))

#Esegue lo split dei dati in training set e test set con una proporzione di 80% training set e 20% test set
X_train, X_test, Y_train, Y_test = train_test_split(df, trainLabels, test_size=0.2, random_state=42)

#Strategia di validazione (Serve a specificare il tag per la strategia utilizzata su Neptune)
strategy = "Subject Biased"

#Le varie funzioni per l'allenamento e la validazione sui vari modelli proposti
#Basta decommentare quella desiderata e avviare il programma
MlAlgs.tensor_flow_test_model(np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test), strategy)
# MlAlgs.svm_test_model(np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test), strategy)
# MlAlgs.gradient_boosting_early_stopping_test_model(np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test), strategy)
# MlAlgs.adaboost_test_model(np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test), strategy)
# MlAlgs.random_forest_test_model(np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test), strategy)
# MlAlgs.tensorflow_cnn_model(np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test), strategy)

#Metodi utilizzati per l'ottimizzazione dei parametri dei vari modelli
#MlAlgs.random_forest_optimization(np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test), strategy)
#MlAlgs.adaboost_optimization(np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test), strategy)
# MlAlgs.gradient_boosting_optimization(np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test), strategy)
# MlAlgs.svm_optimization(np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test), strategy)
# MlAlgs.tensorflow_optimization(np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test), strategy)
# MlAlgs.tensorflow_cnn_optimization(np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test), strategy)
# MlAlgs.auto_keras_test(np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test), strategy)