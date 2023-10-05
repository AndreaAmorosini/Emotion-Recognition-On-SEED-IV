import re
from scipy.io import loadmat
import pandas as pd

import os

#Sostituire le seguenti con i corrispondenti path ai dati sulle feature estratte dai dati EEG, e alle caratteristiche estratte dal movimento oculare
path_eeg = "/Users/ater/Desktop/Progetto EEG/SEED IV Database/SEED_IV Database/eeg_feature_smooth/"
path_eye_data = "/Users/ater/Desktop/Progetto EEG/SEED IV Database/SEED_IV Database/eye_feature_smooth/"

#Label categoriche per ogni sessione
session1Labels = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2Labels = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3Labels = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
labels_dict = {0: "Neutral", 1: "Sad", 2: "Fear", 3: "Happy"}

columnsFreq = ['delta (1-4 Hz)', 'theta (4-8 Hz)', 'alpha (8-14 Hz)', 'beta (4-31 Hz)', 'gamma (31-50 Hz)']

#Funzioni Per il caricamento dei path

# Carica i path dei file .mat contenenti le feature estratte dai dati EEG per ogni sessione nel dataset
def load_trial_paths():
    sessions = ["1", "2", "3"]
    pathPatients = []
    #Per ogni sessione
    for s in sessions:
        #Unisce il path base con il numero di sessione
        pathSession = path_eeg + s
        #Recupera la lista dei pazienti
        patients = os.listdir(pathSession)
        #Li mette in ordine numerico
        patients.sort(key=lambda x: int(x.split('_')[0]))
        path_ps = []
        #per ognuno dei pazienti trovati
        for p in patients:
            #Unisce il path della sessione con il numero del paziente
            pathPatientSession = pathSession + '/' + p
            #Aggiunge il path completo alla lista dei risultati
            pathPatients.append(pathPatientSession)
    return pathPatients

# Carica i path dei file .mat contenenti le feature estratte dal movimento oculare per ogni sessione nel dataset
def load_trial_eye_data_paths():
    sessions = ["1", "2", "3"]
    pathPatients = []
    #Per ogni sessione
    for s in sessions:
        #Unisce il path base con il numero di sessione
        pathSession = path_eye_data + s
        #Carica la lista dei pazienti
        patients = os.listdir(pathSession)
        #Li mette in ordine numerico
        patients.sort(key=lambda x: int(x.split('_')[0]))
        path_ps = []
        #Per ogni paziente
        for p in patients:
            #Unisce il path della sessione con il numero del paziente
            pathPatientSession = pathSession + '/' + p
            #Aggiunge il path completo alla lista dei risultati
            pathPatients.append(pathPatientSession)
    return pathPatients

# Carica i path dei file .mat contenenti le feature estratte dai dati EEG per una specifica sessione
def load_trial_paths_by_session(session):
    pathPatients = []
    #Unisce il path base con il numero di sessione
    pathSession = path_eeg + str(session)
    #Carica la lista dei pazienti
    patients = os.listdir(pathSession)
    #Li mette in ordine numerico
    patients.sort(key=lambda x: int(x.split('_')[0]))
    path_ps = []
    #Per ogni paziente
    for p in patients:
        #Unisce il path della sessione con il numero del paziente
        pathPatientSession = pathSession + '/' + p
        #Aggiunge il path completo alla lista dei risultati
        pathPatients.append(pathPatientSession)
    return pathPatients

# Carica i path dei file .mat contenenti le feature estratte dal movimento oculare per una specifica sessione
def load_trial_eye_data_paths_by_session(session):
    pathPatients = []
    #Unisce il path base con il numero di sessione
    pathSession = path_eye_data + str(session)
    #Carica la lista dei pazienti
    patients = os.listdir(pathSession)
    #Li mette in ordine numerico
    patients.sort(key=lambda x: int(x.split('_')[0]))
    path_ps = []
    #Per ogni paziente
    for p in patients:
        #Unisce il path della sessione con il numero del paziente
        pathPatientSession = pathSession + '/' + p
        #Aggiunge il path completo alla lista dei risultati
        pathPatients.append(pathPatientSession)
    return pathPatients

#Funzioni per la creazione delle feature matrix

#Crea la feature matrix contenente le feature estratte dai dati EEG per ogni sessione e per ogni soggetto
def create_feature_matrix_complete(paths):
    res_list = []

    #Per ogni path specificato
    for p in paths:
        #Carica il file .mat
        procDataEEGExample = loadmat(p)
        #Recupera le chiavi del dizionario
        keysProcDataEEG = [key for key, values in procDataEEGExample.items() if
                           key != '__header__' and key != '__version__' and key != '__globals__']
        # Per ogni trial
        for t in range(0, 24):
            trialData = []
            # Recupera le chiavi del dizionario che contengono i dati del trial corrente
            keys = [key for key in keysProcDataEEG if re.findall(r'\d+', key)[0] == str(t + 1)]
            # Per ogni feature della sessione
            for k in keys:
                dt = procDataEEGExample[k]
                # Per ogni canale
                for i in range(0, 62):
                    chan = dt[i]
                    # Per ogni banda
                    for c in chan:
                        band = c
                        # Per ogni valore
                        for b in band:
                            trialData.append(b)
            res_list.append(trialData)
    return res_list

#Crea la feature matrix contenente le feature estratte dal movimento oculare per ogni sessione e per ogni soggetto
def create_feature_matrix_eye_data_complete(paths):
    res_list = []

    #Per ogni path specificato
    for p in paths:
        #Carica il file .mat
        procDataEEGExample = loadmat(p)
        #Recupera le chiavi del dizionario
        keysProcDataEEG = [key for key, values in procDataEEGExample.items() if
                           key != '__header__' and key != '__version__' and key != '__globals__']
        # Per ogni trial
        for t in range(0, 24):
            trialData = []
            # Recupera le chiavi del dizionario che contengono i dati del trial corrente
            keys = [key for key in keysProcDataEEG if re.findall(r'\d+', key)[0] == str(t + 1)]
            #Per ogni chiave trovata
            for k in keys:
                dt = procDataEEGExample[k]
                #Per ogni feature
                for f in dt:
                    #per ogni singolo dato
                    for d in f:
                        trialData.append(d)
            res_list.append(trialData)

    return res_list

#Unisce le feature matrix contenenti le feature estratte dai dati EEG e dal movimento oculare per ogni sessione e per ogni soggetto
def create_feature_matrix_eye_and_eeg_complete(paths_eeg, paths_eye):
    res_list = []

    #Crea la feature matrix dei segnali EEG
    eeg = create_feature_matrix_complete(paths_eeg)
    #Crea la feature matrix dei dati sul movimento oculare
    eye = create_feature_matrix_eye_data_complete(paths_eye)

    #Li rende Dataframe
    eeg = pd.DataFrame(eeg)
    eye = pd.DataFrame(eye)

    #Li concatena lungo le colonne
    res_list = pd.concat([eeg, eye], axis=1, ignore_index=True)
    res_list = res_list.values.tolist()

    return res_list

#Crea due feature matrix distinte per i segnali EEG, una per il training dai path specificati in paths_training,
# e una per i test dai path specificati in paths_test
def create_feature_matrix_subIndep_split(paths_training, paths_test):
    train_list = []
    test_list = []

    #Per ogni path di training specificato
    for p in paths_training:
        #Carica il file .mat
        procDataEEGExample = loadmat(p)
        #Recupera le chiavi del dizionario
        keysProcDataEEG = [key for key, values in procDataEEGExample.items() if
                           key != '__header__' and key != '__version__' and key != '__globals__']
        # Per ogni trial
        for t in range(0, 24):
            trialData = []
            # Recupera le chiavi del dizionario che contengono i dati del trial corrente
            keys = [key for key in keysProcDataEEG if re.findall(r'\d+', key)[0] == str(t + 1)]
            # Per ogni feature della sessione
            for k in keys:
                dt = procDataEEGExample[k]
                # Per ogni canale
                for i in range(0, len(dt)):
                    chan = dt[i]
                    # Per ogni banda
                    for c in chan:
                        band = c
                        # Per ogni valore
                        for b in band:
                            trialData.append(b)
            train_list.append(trialData)

        #Per ogni path di test specificato
        for p in paths_test:
            # Carica il file .mat
            procDataEEGExample = loadmat(p)
            # Recupera le chiavi del dizionario
            keysProcDataEEG = [key for key, values in procDataEEGExample.items() if
                               key != '__header__' and key != '__version__' and key != '__globals__']
            # Per ogni trial
            for t in range(0, 24):
                trialData = []
                # Recupera le chiavi del dizionario che contengono i dati del trial corrente
                keys = [key for key in keysProcDataEEG if re.findall(r'\d+', key)[0] == str(t + 1)]
                # Per ogni feature della sessione
                for k in keys:
                    dt = procDataEEGExample[k]
                    # Per ogni canale
                    for i in range(0, 62):
                        chan = dt[i]
                        # Per ogni banda
                        for c in chan:
                            band = c
                            # Per ogni valore
                            for b in band:
                                trialData.append(b)
                test_list.append(trialData)

    return train_list, test_list

#Crea due feature matrix distinte per i dati sul movimento oculare, una per il training dai path specificati in paths_training,
# e una per i test dai path specificati in paths_test
def create_feature_matrix_subIndep_eye_data_split(paths_training, paths_test):
    train_list = []
    test_list = []

    #Per ogni path di training specificato
    for p in paths_training:
        #Carica il file .mat
        procDataEEGExample = loadmat(p)
        #Recupera le chiavi del dizionario
        keysProcDataEEG = [key for key, values in procDataEEGExample.items() if
                           key != '__header__' and key != '__version__' and key != '__globals__']
        # Per ogni trial
        for t in range(0, 24):
            trialData = []
            # Recupera le chiavi del dizionario che contengono i dati del trial corrente
            keys = [key for key in keysProcDataEEG if re.findall(r'\d+', key)[0] == str(t + 1)]
            #Per ogni chiave trovata
            for k in keys:
                dt = procDataEEGExample[k]
                #Per ogni feature
                for f in dt:
                    #per ogni singolo dato
                    for d in f:
                        trialData.append(d)
            train_list.append(trialData)

        #Per ogni path di test specificato
        for p in paths_test:
            # Carica il file .mat
            procDataEEGExample = loadmat(p)
            # Recupera le chiavi del dizionario
            keysProcDataEEG = [key for key, values in procDataEEGExample.items() if
                               key != '__header__' and key != '__version__' and key != '__globals__']
            # Per ogni trial
            for t in range(0, 24):
                trialData = []
                # Recupera le chiavi del dizionario che contengono i dati del trial corrente
                keys = [key for key in keysProcDataEEG if re.findall(r'\d+', key)[0] == str(t + 1)]
                # Per ogni chiave trovata
                for k in keys:
                    dt = procDataEEGExample[k]
                    # Per ogni feature
                    for f in dt:
                        # per ogni singolo dato
                        for d in f:
                            trialData.append(d)
                test_list.append(trialData)

    return train_list, test_list

#Unisce le feature matrix contenenti le feature estratte dai dati EEG e dal movimento oculare per i due metodi precedenti
def create_feature_matrix_subIndep_eeg_and_eye_split(paths_training_eeg, paths_test_eeg, paths_training_eye, paths_test_eye):
    train_list = []
    test_list = []

    matrix_eeg_train, matrix_eeg_test = pd.DataFrame(create_feature_matrix_subIndep_split(paths_training_eeg, paths_test_eeg))
    matrix_eye_train, matrix_eye_test = pd.DataFrame(create_feature_matrix_subIndep_eye_data_split(paths_training_eye, paths_test_eye))

    train_list = pd.concat([matrix_eeg_train, matrix_eye_train], axis=1, ignore_index=True).values.tolist()
    test_list = pd.concat([matrix_eeg_test, matrix_eye_test], axis=1, ignore_index=True).values.tolist()

    return train_list, test_list

#Crea una feature matrix per i segnali EEG dai path specificati in paths_training,
# splittandola in due feature matrix, una per il testing contenente le feature estratte dalle trial specificate in trails_test,
# ed una per il training contenente i dati rimanenti
def create_feature_matrix_subDep_split(paths_training, trials_test):
    train_list = []
    test_list = []

    #Per ogni path fornito
    for p in paths_training:
        #Carica il file .mat
        procDataEEGExample = loadmat(p)
        #Recupera le chiavi del dizionario
        keysProcDataEEG = [key for key, values in procDataEEGExample.items() if
                           key != '__header__' and key != '__version__' and key != '__globals__']
        # Per ogni trial
        for t in range(0, 24):
            trialData = []
            # Recupera le chiavi del dizionario che contengono i dati del trial corrente
            keys = [key for key in keysProcDataEEG if re.findall(r'\d+', key)[0] == str(t + 1)]
            # Per ogni feature della trial
            for k in keys:
                dt = procDataEEGExample[k]
                # Per ogni canale
                for i in range(0, 62):
                    chan = dt[i]
                    # Per ogni banda
                    for c in chan:
                        band = c
                        # Per ogni valore
                        for b in band:
                            trialData.append(b)

            #Decide se aggiungere la trial alla feature matrix di training o di test
            if t in trials_test:
                test_list.append(trialData)
            else:
                train_list.append(trialData)

    return train_list, test_list

#Crea una feature matrix per i dati sul movimento oculare dai path specificati in paths_training,
# splittandola in due feature matrix, una per il testing contenente le feature estratte dalle trial specificate in trails_test,
# ed una per il training contenente i dati rimanenti
def create_feature_matrix_subDep_eye_data_split(paths_training, trials_test):
    train_list = []
    test_list = []

    #Per ogni path fornito
    for p in paths_training:
        #Carica il file .mat
        procDataEEGExample = loadmat(p)
        #Recupera le chiavi del dizionario
        keysProcDataEEG = [key for key, values in procDataEEGExample.items() if
                           key != '__header__' and key != '__version__' and key != '__globals__']
        # Per ogni trial
        for t in range(0, 24):
            trialData = []
            # Recupera le chiavi del dizionario che contengono i dati del trial corrente
            keys = [key for key in keysProcDataEEG if re.findall(r'\d+', key)[0] == str(t + 1)]
            # Per ogni feature della trial
            for k in keys:
                dt = procDataEEGExample[k]
                # Per ogni feature
                for f in dt:
                    # Per ogni dato
                    for d in f:
                        trialData.append(d)

            #Decide se aggiungere la trial alla feature matrix di training o di test
            if t in trials_test:
                test_list.append(trialData)
            else:
                train_list.append(trialData)

    return train_list, test_list

#Unisce le feature matrix contenenti le feature estratte dai dati EEG e dal movimento oculare per i due metodi precedenti
def create_feature_matrix_subDep_eeg_and_eye_split(paths_training_eeg, paths_training_eye, trials_test):
    train_list = []
    test_list = []

    matrix_eeg_train, matrix_eeg_test = create_feature_matrix_subDep_split(paths_training_eeg, trials_test)
    matrix_eeg_train = pd.DataFrame(matrix_eeg_train)
    matrix_eeg_test = pd.DataFrame(matrix_eeg_test)
    matrix_eye_train, matrix_eye_test = create_feature_matrix_subDep_eye_data_split(paths_training_eye, trials_test)
    matrix_eye_train = pd.DataFrame(matrix_eye_train)
    matrix_eye_test = pd.DataFrame(matrix_eye_test)

    train_list = pd.concat([matrix_eeg_train, matrix_eye_train], axis=1, ignore_index=True).values.tolist()
    test_list = pd.concat([matrix_eeg_test, matrix_eye_test], axis=1, ignore_index=True).values.tolist()

    return train_list, test_list


