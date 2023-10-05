import neptune
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
import MlAlgs

import featureMatrixSeed
import numpy as np
import pandas as pd

#Label per ogni trial in ogni sessione
session1Labels = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2Labels = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3Labels = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
labels_dict = {0: "Neutral", 1: "Sad", 2: "Fear", 3: "Happy"}

#Inizializza delle liste per tenere traccia delle performance tra le varie sessioni
meanAccSession = []
meanTrainAccSession = []
meanLossSession = []
meanTrainLossSession = []

#Lista dei soggetti da esaminare
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

fmt_all = []

# #Crea la Run per il log dei dati su neptune (Presente al di fuori dei metodi dei modelli per evitare un log per ogni validazione del modello)
# run = neptune.init_run(
#     project="",
#     api_token="",
##    Cambiare name in base al modello utilizzato
#     name="SVM",
#     tags=["SVM", "Subject Independent"]
# )  # your credentials

#inizializza un dict di parametri per il modello scelto (Verra' popolato dopo aver chiamato uno dei modelli)
parameters = dict()

# Per ogni sessione
for s in range(1,4):
    #EEG
    #Carica i path dei file .mat per ogni trial della sessione s
    #in caso non si stia usando il dataset, commentare la riga
    path_session_eeg = featureMatrixSeed.load_trial_paths_by_session(s)

    #Crea la feature matrix per la sessione
    # Per creare la feature matrix per segnali EEG da dati differenti dal dataset,
    # Basta passare alla funzione di creazione della feature matrix una lista contenente i path dei file .mat desiderati
    # feature_matrix = featureMatrixSeed.create_feature_matrix_complete(path_session_eeg)

    #EYE
    #Carica i path dei file .mat per ogni trial della sessione s
    #in caso non si stia usando il dataset, commentare la riga
    path_session_eye = featureMatrixSeed.load_trial_eye_data_paths_by_session(s)

    #Crea la feature matrix di train e di test per la sessione (solo segnali oculari)
    # Per creare la feature matrix per segnali EEG da dati differenti dal dataset,
    # Basta passare alla funzione di creazione della feature matrix una lista contenente i path dei file .mat desiderati
    # feature_matrix = featureMatrixSeed.create_feature_matrix_eye_data_complete(path_session_eye)

    #EEG+EYE
    #Unisce le feature matrix di segnali EEG e oculari
    feature_matrix = featureMatrixSeed.create_feature_matrix_eye_and_eeg_complete(path_session_eeg, path_session_eye)


    df = pd.DataFrame(feature_matrix)

    # Serve a rimuovere i valori NaN dovuti a una differenza di tempistiche tra i vari trial
    df = df.replace(np.nan, 0)

    #Inizializza lo scaler per normalizzare i dati tra 0 e 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    #Allena lo scaler e trasforma i dati nella feature matrix
    df = scaler.fit_transform(df)


    # Train Data
    # inizializza la PCA per la riduzione delle feature matrix in 80 componenti
    pca = PCA(n_components=80, whiten=True)
    #Allena la PCA
    pca.fit(df)
    #Riduce le dimensioni della feature matrix
    df = pd.DataFrame(pca.transform(df))
    # print("explained variance ratio (Train data): ", pca.explained_variance_ratio_)
    # print("Preserved Variance (Train data): ", sum(pca.explained_variance_ratio_))

    #Aggiunge una colonna indicante il soggetto alla feature matrix cosi' da poterla dividere in train e test data in base al soggetto
    participantsColumn = []
    participantCounter = 1
    trialCounter = 1
    for i in range(0, len(df.index)):
        participantsColumn.append(participantCounter)
        if trialCounter == 24:
            trialCounter = 1
            participantCounter += 1
        else:
            trialCounter += 1

    df["Subject"] = participantsColumn

    #Inizializza le liste per tenere traccia delle performance tra i vari soggetti scelti come test_set
    train_accuracy_list = []
    accuracy_list = []
    loss_list = []
    train_loss_list = []

    #Per ogni soggetto
    for p in subjects:
        #Seleziona come train_set tutti i soggetti tranne quello scelto e rimuove la colonna Subject
        x_train = df.loc[df["Subject"] != p].drop("Subject", axis=1)
        #Seleziona come test_set il soggetto scelto e rimuove la colonna Subject
        x_test = df.loc[df["Subject"] == p].drop("Subject", axis=1)
        #Crea le label di train
        y_train = session1Labels * 14
        #Crea le label di test
        y_test = session1Labels

        X_train = np.asarray(x_train)
        X_test = np.asarray(x_test)

        #Specifica la strategia di validazione usata per non far loggare le singole trial a Optune
        strategy = "Subject Independent"

        # Metodi per l'applicazione di uno dei modelli
        # Basta decommentare quella desiderata e avviare il programma
        #Per DNN e CNN vengono recuperate anche le misure di loss per poterle rappresentare come grafico ma risultano commentati per evitare conflitti con gli altri modelli

        # val_acc, val_loss, train_acc, train_loss, params = MlAlgs.tensor_flow_test_model(np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), strategy)
        # val_acc, train_acc, precision, recall, f1, params = MlAlgs.svm_test_model(np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), strategy)
        # val_acc, train_acc, precision, recall, f1, params = MlAlgs.gradient_boosting_early_stopping_test_model(np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), strategy)
        # val_acc, train_acc, precision, recall, f1, params = MlAlgs.adaboost_test_model(np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), strategy)
        val_acc, train_acc, precision, recall, f1, params = MlAlgs.random_forest_test_model(np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), strategy)
        # val_acc, val_loss, train_acc, train_loss, params = MlAlgs.tensorflow_cnn_model(np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test), strategy)

        #Recupera i parametri usati
        parameters = params
        #Aggiunge la validation accuracy per questo soggetto come test_set alla lista
        accuracy_list.append(val_acc)
        #Aggiunge la training accuracy per questo soggetto come test_set alla lista
        train_accuracy_list.append(train_acc)
        #Decommentare le due righe seguenti se si usa DNN o CNN
        # loss_list.append(val_loss)
        # train_loss_list.append(train_loss)

    train_accuracy_list_1 = []
    for i in train_accuracy_list:
        mn = np.mean(i)
        train_accuracy_list_1.append(mn)

    train_accuracy_list = train_accuracy_list_1

    #Decommentare le due righe seguenti se si usa DNN o CNN
    # train_loss_list_1 = []
    # for i in train_loss_list:
    #     mn = np.mean(i)
    #     train_loss_list_1.append(mn)
    #
    # train_loss_list = train_loss_list_1

    print("ACCURACY LIST")
    print(accuracy_list)
    print("TRAIN ACCURACY LIST")
    print(train_accuracy_list)

    #Calcola la media delle accuracy di test e di train per la sessione
    accuracy_session = np.asarray(accuracy_list)
    mean = np.mean(accuracy_session)
    mean_train_accuracy = np.asarray(train_accuracy_list)
    mean_train_accuracy = np.mean(mean_train_accuracy)

    #Decommentare le righe seguenti se si usa DNN o CNN per calcolare la media delle loss di test e train per la sessione
    # loss_session = np.asarray(loss_list)
    # mean_loss = np.mean(loss_session)
    # mean_train_loss = np.asarray(train_loss_list)
    # mean_train_loss = np.mean(mean_train_loss)

    print("\nAverage Train Accuracy for session : " + str(s) + ": " + str(mean_train_accuracy))
    print("Average Validation Accuracy for session : " + str(s) + ": " + str(mean))
    # print("Average Train Loss for session : " + str(s) + ": " + str(mean_train_loss))
    # print("Average Validation Loss for session : " + str(s) + ": " + str(mean_loss))
    #Aggiunge le medie di accuracy di test e train alle liste corrispondenti
    meanAccSession.append(mean)
    meanTrainAccSession.append(mean_train_accuracy)
    #Decommentare le due righe seguenti se si usa DNN o CNN
    # meanLossSession.append(mean_loss)
    # meanTrainLossSession.append(mean_train_loss)

    # Plot dei dati di accuracy per sessione
    # Summarize History of Accuracy
    fig1 = plt.figure()
    ax = plt.subplot(1, 1, 1)
    ax.plot(np.asarray(train_accuracy_list))
    ax.plot(accuracy_session)
    plt.title('Average Model Accuracy across session ' + str(s))
    plt.ylabel('Accuracy')
    plt.xlabel('Patients')
    plt.grid(True)
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    #Log del grafico su Neptune
    # run["model_accuracy_session_" + str(s)].upload(fig1)

    #Decommentare le righe seguenti se si usa DNN o CNN per esegire il plot della loss per sessione
    # # Summarize History of loss
    # fig2 = plt.figure()
    # ax = plt.subplot(1, 1, 1)
    # ax.plot(np.asarray(train_loss_list))
    # ax.plot(loss_session)
    # plt.title('Average Model Loss Across Session ' + str(s))
    # plt.ylabel('Loss')
    # plt.xlabel('Patients')
    # plt.grid(True)
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    # run["model_loss_session_" + str(s)].upload(fig2)

#Log dei parametri e dei risultati su Neptune
# run["parameters"] = parameters
# run["test_accuracy"] = np.mean(np.asarray(meanAccSession))
# run["training_accuracy"] = np.mean(np.asarray(meanTrainAccSession))
#Decommentare queste solo se si usa DNN o CNN
# run["train_loss"] = np.mean(np.asarray(meanTrainLossSession))
# run["test_loss"] = np.mean(np.asarray(meanLossSession))

print("\n\nAverage Training Accuracy for all sessions: " + str(np.mean(np.asarray(meanTrainAccSession))))
print("Average Validation Accuracy for all sessions: " + str(np.mean(np.asarray(meanAccSession))))
# print("Average Training Loss for all sessions: " + str(np.mean(np.asarray(meanTrainLossSession))))
# print("Average Validation Loss for all sessions: " + str(np.mean(np.asarray(meanLossSession))))

# plot dei dati dfi accuracy tra le tre sessioni
#Summarize History of Accuracy
fig1 = plt.figure()
ax = plt.subplot(1,1,1)
ax.plot(meanTrainAccSession)
ax.plot(meanAccSession)
plt.title('Average Model Accuracy across sessions')
plt.ylabel('Accuracy')
plt.xlabel('Sessions')
plt.grid(True)
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#Log del grafico su Neptune
# run["model_accuracy"].upload(fig1)

#Decommentare le righe seguenti se si usa DNN o CNN per eseguire il plot della loss tra le tre sessioni
#Summarize History of loss
# fig2 = plt.figure()
# ax = plt.subplot(1,1,1)
# ax.plot(meanTrainLossSession)
# ax.plot(meanLossSession)
# plt.title('Average Model Loss Across Sessions')
# plt.ylabel('Loss')
# plt.xlabel('Sessions')
# plt.grid(True)
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
# run["model_loss"] = fig2

# run.stop()




