
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from neptune.utils import stringify_unsupported
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, precision_score, recall_score, \
    f1_score, log_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import scikitplot as skplt
import keras_tuner as kt
import neptune
import neptune.integrations.sklearn as npt_sk
import neptune.integrations.optuna as npt_utils
import optuna
import autokeras as ak

#Labels per ogni trial delle tre sessioni
session1Labels = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
session2Labels = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
session3Labels = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
labels_dict = {0: "Neutral", 1: "Sad", 2: "Fear", 3: "Happy"}

labelsTot = session1Labels * 12 + session2Labels * 12 + session3Labels * 12
labelsTotTest = session1Labels * 3 + session2Labels * 3 + session3Labels * 3

#Esegue il training e il test per il modello DNN
def tensor_flow_test_model(X_train, y_train, x_test, y_test, strategy):
    #Utility per iniziare la run di logging tramite la libreria Neptune
    # if strategy != "Subject Independent":
    #     run = neptune.init_run(
    #         project="",
    #         api_token="",
    #         name="DNN",
    #         tags=["DNN", strategy]
    #     )  # your credentials

    #Parametri per il modello DNN
    params = {"num_layers": 2,
              "units_1": 406,
              "regularization_1": "l1_l2",
              "dropout_1": 0.05139282268657297,
              "units_2": 310,
              "regularization_2": "l2",
              "dropout_2": 0.2812025229422671,
              "units_3": 342,
              "regularization_3": "l2",
              "dropout_3": 0.2812025229422671,
              "units_4": 502,
              "regularization_4": "l2",
              "dropout_4": 0.1,
              "learning_rate": 0.01}

    #Log dei parametri tramite neptune
    # if strategy != "Subject Independent":
    #     run["parameters"] = params

    # Create the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    #Aggiunge tanti layer Dense e di Dropout quanto specificato dal parametro num_layers
    for i in range(1, params["num_layers"] + 1):
        #Aggiunge un layer Dense con units e kernel_regularizer specificati in params
        model.add(tf.keras.layers.Dense(units=params["units_" + str(i)], activation='relu',
                                        kernel_regularizer=params["regularization_" + str(i)]))
        #Aggiunge un layer di Dropout con rate specificato in params
        model.add(tf.keras.layers.Dropout(rate=params["dropout_" + str(i)]))
    #Layer finale di output
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    # Baseline
    # hp_learning_rate = 0.001
    hp_learning_rate = params["learning_rate"]

    #Compila il moello usando il learning_rate specificato in params
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    #Funzione di early_stopping basata sulla validation accuracy
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

    #Allena il modello su 40 epoche
    history = model.fit(X_train, y_train, epochs=40, callbacks=[stop_early], validation_data=(x_test, y_test),
                        shuffle=True)

    #Ottiene le misure di accuracy e loss
    acc = history.history['accuracy']
    loss = history.history['loss']

    # Valuta il modello sul test set
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print("\nEVALUATION (val_loss, val_acc)")
    print(val_loss, val_acc)

    # Make predictions
    predictions = model.predict(x_test)

    #Calcola la precisione e la loss sul training set
    trainingLoss, trainingAccuracy = model.evaluate(X_train, y_train)

    #Logging dei risultati tramite Neptune
    # if strategy != "Subject Independent":
    #     run["training_accuracy"] = trainingAccuracy
    #     run["training_loss"] = trainingLoss
    #     run["test_accuracy"] = val_acc
    #     run["test_loss"] = val_loss


    print("METRICS")
    print("Test accuracy: ", val_acc)
    print("Test loss: ", val_loss)
    print("Training accuracy: ", trainingAccuracy)
    print("Training loss: ", trainingLoss)

    # plot of data
    if strategy != "Subject Independent":
        # Summarize History of Accuracy
        fig1 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.plot(history.history['accuracy'])
        ax.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        #Log del grafico tramite Neptune
        # run["model_accuracy"].upload(fig1)

        # Summarize History of loss
        fig2 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        #Log del grafico tramite Neptune
        # run["model_loss"].upload(fig2)

        #Stop alla run di neptune
        # run.stop()

    return val_acc, val_loss, acc, loss, params

#Metodo per l'ottimizzazione dei parametri della DNN tramite Scikit-Learn o Optuna
def tensorflow_optimization(X_train, y_train, x_test, y_test, strategy):
    #Possibili sampler per l'ottimizzazione Optuna
    # sampler = optuna.samplers.TPESampler(multivariate=True)
    sampler = optuna.samplers.RandomSampler()
    # sampler = optuna.samplers.QMCSampler(qmc_type='sobol', scramble=True, seed = 42)

    #Possibili pruner per l'ottimizzazione Optuna
    pruning = optuna.pruners.HyperbandPruner(min_resource=5, max_resource="auto", reduction_factor=3)
    # pruning = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    # pruning = optuna.pruners.SuccessiveHalvingPruner()

    #Inizio del log tramite Neptune
    # if strategy != "Subject Independent":
    #     run = neptune.init_run(
    #         project="",
    #         api_token="",
    #         name="DNN",
    #         tags=["DNN", "DNN Optimization", strategy, str(sampler).split(".")[-1].split("(")[0],
    #               str(pruning).split(".")[-1].split("(")[0].split(" ")[0]]
    #     )  # your credentials

    # Ottimizzazione tramite scikit-learn

    # Funzione per la costruzione del modello scegliendo i parametri da ottimizzare
    # def build_model(hp):
    #     model = tf.keras.Sequential()
    #     model.add(tf.keras.layers.Flatten())
    #
    #     #Specifica lo spazio di ricerca per num_layers, units, kernel_regularizer e dropout
    #     for i in range(1, hp.Int('num_layers', min_value=2, max_value=5, step=1)):
    #         model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
    #                                                      min_value=150,
    #                                                      max_value=512,
    #                                                      step=32),
    #                                         activation='relu',
    #                                         kernel_regularizer=hp.Choice('regularization_' + str(i),
    #                                                                      values=['l1', 'l2', 'l1_l2'])))
    #         model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_' + str(i), min_value=0, max_value=0.5, step=0.1)))
    #
    #     model.add(tf.keras.layers.Dense(4, activation='softmax'))
    #
    #     hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    #
    #     #Specifica lo spazio di ricerca per learning_rate
    #     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
    #                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #                   metrics=['accuracy'])
    #
    #     return model
    #
    # #Specifica i tre possibili tuner utilizzabili per l'ottimizzazione:
    # #Hyperband, RandomSearch e BayesianOptimization con i loro rispettivi parametri per il tuning della ricerca
    # tuner = kt.Hyperband(build_model,
    #                      objective='val_accuracy',
    #                      max_epochs=30,
    #                      factor=2,
    #                      hyperband_iterations=3,
    #                      distribution_strategy=tf.distribute.MirroredStrategy(),
    #                      directory='tempData',
    #                      overwrite=True,
    #                      project_name='DNN')
    #
    # tuner = kt.RandomSearch(build_model,
    #                         objective='val_accuracy',
    #                         max_trials=100,
    #                         executions_per_trial=4,
    #                         directory='tempData',
    #                         overwrite=True,
    #                         project_name='DNN')
    #
    # tuner = kt.BayesianOptimization(build_model,
    #                      objective='val_accuracy',
    #                      max_trials=100,
    #                     directory="tempData", overwrite=True, project_name="DNN")
    #
    # #Specifica il callback per l'early stopping
    # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    #
    # #Esegue la ricerca dei parametri ottimali
    # tuner.search(X_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[stop_early], verbose=2)
    # print("END SEARCH")
    #
    # # Ottiene gli iperparametri migliori
    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # print("Best_Hps:")
    # print(best_hps.values)
    #
    # # Costruisce il modello con gli iperparametri migliori
    # h_model = tuner.hypermodel.build(best_hps)
    #
    # # Addestra il modello
    # h_model.fit(X_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[stop_early], verbose=2)
    #
    # # Valuta il modello
    # val_loss, val_acc = h_model.evaluate(x_test, y_test)
    # print("Test accuracy: ", val_acc)
    # print("Test loss: ", val_loss)
    #


    # Optuna Optimization
    def objective(trial):
        # Builds Model and sets up hyperparameters space to search
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten())

        # Specifica lo spazio di ricerca per num_layers, units, kernel_regularizer e dropout
        num_layers = trial.suggest_int('num_layers', 2, 5)
        for i in range(1, num_layers + 1):
            model.add(tf.keras.layers.Dense(units=trial.suggest_int('units_' + str(i), 150, 512, 32),
                                            activation='relu',
                                            kernel_regularizer=trial.suggest_categorical('regularization_' + str(i),
                                                                                         ['l1', 'l2', 'l1_l2'])))
            model.add(tf.keras.layers.Dropout(rate=trial.suggest_float(name='dropout_' + str(i), low=0, high=0.5)))

        model.add(tf.keras.layers.Dense(4, activation='softmax'))

        # Specifica lo spazio di ricerca per learning_rate
        hp_learning_rate = trial.suggest_categorical('learning_rate', [1e-2, 1e-3, 1e-4])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        # Specifica la funzione di early_stopping
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

        # Addestra il modello
        model.fit(X_train, y_train, epochs=40, validation_data=(x_test, y_test), callbacks=[stop_early], verbose=2)

        # Restituisce loss e accuracy per ogni combinazione di iperparametri
        return model.evaluate(x_test, y_test)[0], model.evaluate(x_test, y_test)[1]

    #Crea lo studio di ottimizzazione
    study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler, pruner=pruning)
    #Avvia lo studio
    study.optimize(objective, n_trials=100)

    #Recupera i risultati dello studio
    best_trials = study.best_trials
    b_t = dict()
    print("Best trials:", len(best_trials))
    for i in best_trials:
        params = ""
        for j in i.params:
            params += str(j) + " : " + str(i.params[j]) + " \n"
        values = ""
        for j in i.values:
            values += str(j) + ", \n"
        b_t[str(i.number)] = "params : " + params + "\nvalues : " + values
    #Log delle migliori trial su neptune
    # run["best_trials"] = b_t

    #Recupera la trial con accuracy migliore e i suoi parametri
    highest_trial = max(best_trials, key=lambda t: t.values[1])
    best_params = highest_trial.params
    best_score = highest_trial.values[1]

    #Log dei parametri migliori su neptune
    # run["parameters"] = best_params

    print("Best score:", best_score)
    print("Best params:", best_params)

    # TEST THE MODEL con i parametri ottimali
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for i in range(1, best_params["num_layers"] + 1):
        model.add(tf.keras.layers.Dense(units=best_params["units_" + str(i)],
                                        activation='relu',
                                        kernel_regularizer=best_params["regularization_" + str(i)]))
        model.add(tf.keras.layers.Dropout(rate=best_params["dropout_" + str(i)]))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    hp_learning_rate = best_params["learning_rate"]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    #Funzione di early-stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

    # Train the model
    history = model.fit(X_train, y_train, epochs=40, callbacks=[stop_early], validation_data=(x_test, y_test),
                        shuffle=True)

    acc = history.history['accuracy']
    loss = history.history['loss']

    # Evaluate the model
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print("\nEVALUATION (val_loss, val_acc)")
    print(val_loss, val_acc)

    # Make predictions
    predictions = model.predict(x_test)

    trainingLoss, trainingAccuracy = model.evaluate(X_train, y_train)

    # Log dei risultati su neptune
    # if strategy != "Subject Independent":
    #     run["training_accuracy"] = trainingAccuracy
    #     run["training_loss"] = trainingLoss
    #     run["test_accuracy"] = val_acc
    #     run["test_loss"] = val_loss

    # Print dei risultati
    print("METRICS")
    print("Training accuracy: ", trainingAccuracy)
    print("Test accuracy: ", val_acc)
    print("Test loss: ", val_loss)

    # plot of data
    if strategy != "Subject Independent":
        # Summarize History of Accuracy
        fig1 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.plot(history.history['accuracy'])
        ax.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        #Log del grafico su neptune
        # run["model_accuracy"].upload(fig1)

        # Summarize History of loss
        fig2 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        #Log del grafico su neptune
        # run["model_loss"].upload(fig2)

        # run.stop()

#Esegue il modello Random Forest sui dati passati
def random_forest_test_model(X_train, y_train, x_test, y_test, strategy):
    #Crea la run per il log dei dati su Neptune
    # if strategy != "Subject Independent":
    #     run = neptune.init_run(
    #         project="",
    #         api_token="",
    #         name="Random Forest",
    #         tags=["Random Forest", strategy]
    #     )  # your credentials

    #Parametri da utilizzare nel modello
    params = {'n_estimators': 530,
              'max_features': 'sqrt',
              'max_depth': 250,
              'min_samples_split': 20,
              'min_samples_leaf': 2,
              'bootstrap': True}

    # Log dei parametri su neptune
    # if strategy != "Subject Independent":
    #     run["parameters"] = params

    #Crea il modello
    bestModel = RandomForestClassifier(n_estimators=params["n_estimators"],
                                       min_samples_split=params["min_samples_split"],
                                       min_samples_leaf=params["min_samples_leaf"], max_features=params["max_features"],
                                       max_depth=params["max_depth"], bootstrap=params["bootstrap"], random_state=42)
    #Allena il modello
    bestModel.fit(X_train, y_train)

    #Calcola l'accuracy sui dati di training
    traningPredictions = bestModel.predict(X_train)
    trainingAccuracy = accuracy_score(y_train, traningPredictions)

    # Make predictions e calcola le metriche di test
    bestPredictions = bestModel.predict(x_test)
    bestAccuracy = accuracy_score(y_test, bestPredictions)
    bestPrecision = precision_score(y_test, bestPredictions, average='weighted')
    bestRecall = recall_score(y_test, bestPredictions, average='weighted')
    bestF1 = f1_score(y_test, bestPredictions, average='weighted')

    # Log dei risultati su neptune
    # if strategy != "Subject Independent":
    #     run["training_accuracy"] = trainingAccuracy
    #     run["test_accuracy"] = bestAccuracy
    #     run["test_precision"] = bestPrecision
    #     run["test_recall"] = bestRecall
    #     run["test_f1"] = bestF1
    #     run["confusion_matrix"] = npt_sk.create_confusion_matrix_chart(bestModel, X_train, x_test, y_train, y_test)
    #     run["classification_report"] = npt_sk.create_classifier_summary(bestModel, X_train, x_test, y_train, y_test)

    print("METRICS")
    print("Training accuracy: ", trainingAccuracy)
    print("Test accuracy: ", bestAccuracy)
    print("Test Precision : ", bestPrecision)
    print("Test Recall : ", bestRecall)
    print("Test F1 : ", bestF1)
    print("Classification Report: ")
    print(classification_report(y_test, bestPredictions))

    # plotting della learning curve e della confusion matrix per il modello
    if strategy != "Subject Independent":
        fig1 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        skplt.estimators.plot_learning_curve(bestModel, X_train, y_train, title="Learning Curve Random Forest",
                                             scoring="accuracy",
                                             shuffle=True, figsize=(6, 4), ax=ax)
        plt.show()
        #Log del grafico su neptune
        # run["learning_curve"].upload(fig1)
        fig2 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        skplt.metrics.plot_confusion_matrix(y_test, bestPredictions, normalize=True,
                                            title="Confusion Matrix Random Forest", ax=ax)
        plt.show()
        #Log del grafico su neptune
        # run["confusion_matrix"].upload(fig2)

        # run.stop()

    return bestAccuracy, trainingAccuracy, bestPrecision, bestRecall, bestF1, params

#Metodo che esegue l'ottimizzazione del modello Random Forest
def random_forest_optimization(X_train, y_train, x_test, y_test, strategy):
    #Sampler per l'ottimizzazione Optune
    sampler = optuna.samplers.TPESampler(multivariate=True)
    # sampler = optuna.samplers.RandomSampler()
    # sampler = optuna.samplers.QMCSampler(qmc_type='sobol', scramble=True, seed = 42)

    #Pruner per l'ottimizzazione Optune
    pruning = optuna.pruners.HyperbandPruner(min_resource=5, max_resource="auto", reduction_factor=3)
    # pruning = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    # pruning = optuna.pruners.SuccessiveHalvingPruner()

    #Inizializza la run per i log con neptune
    # run = neptune.init_run(
    #     project="",
    #     api_token="",
    #     name="Random Forest Optimization",
    #     tags=["Random Forest Optimization", strategy, str(sampler).split(".")[-1].split("(")[0],
    #           str(pruning).split(".")[-1].split("(")[0].split(" ")[0]]
    # )  # your credentials

    #Funzione obiettivo per l'ottimizzazione Optune
    def objective(trial):
        #Parametri da ottimizzare
        n_estimators = trial.suggest_int("n_estimators", 200, 2000, 10)
        max_depth = trial.suggest_int("max_depth", 10, 310, 10)
        max_features = trial.suggest_categorical("max_features", ["sqrt"])
        min_samples_split = trial.suggest_int("min_samples_split", 15, 60, 5)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10, 1)
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])

        #Crea il modello, lo allena e calcola l'accuracy
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            random_state=42,
        )

        model.fit(X_train, y_train)

        trainingPredictions = model.predict(X_train)
        trainingAccuracy = accuracy_score(y_train, trainingPredictions)

        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        # score based on value of overfitting

        return trainingAccuracy - accuracy, accuracy

    # Optuna-scikilearn optimization con k-fold validation
    # param_distribution = {
    #     "n_estimators": optuna.distributions.IntDistribution(500, 2200, step=10),
    #     "max_depth": optuna.distributions.IntDistribution(100, 310, step=5),
    #     "max_features": optuna.distributions.CategoricalDistribution(["sqrt", "log2", None]),
    #     "min_samples_split": optuna.distributions.IntDistribution(15, 30, step=1),
    #     "min_samples_leaf": optuna.distributions.IntDistribution(5, 30, step=5),
    #     "bootstrap": optuna.distributions.CategoricalDistribution([True, False]),
    # }
    #
    # model = RandomForestClassifier(random_state=42)
    # optuna_search = optuna.integration.OptunaSearchCV(model, param_distribution, n_trials=100, n_jobs=-1, cv=10,
    #                                                   enable_pruning=False, random_state=42, verbose=2)
    # optuna_search.fit(X_train, y_train)
    # y_pred = optuna_search.predict(x_test)
    #
    # best_params = optuna_search.best_params_
    # print("Best parameters found: ", optuna_search.best_params_)
    # print("Best score found: ", optuna_search.best_score_)
    #
    # trainingPredictions = optuna_search.predict(X_train)
    # trainingAccuracy = accuracy_score(y_train, trainingPredictions)
    # testAccuracy = accuracy_score(y_test, y_pred)

    # Optuna Optimization
    #Crea lo studio optuna
    study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler, pruner=pruning)
    #Avvia la ricerca
    study.optimize(objective, n_trials=100)

    #Ottiene i risultati
    best_trials = study.best_trials
    b_t = dict()
    print("Best trials:", len(best_trials))
    for i in best_trials:
        params = ""
        for j in i.params:
            params += str(j) + " : " + str(i.params[j]) + " \n"
        values = ""
        for j in i.values:
            values += str(j) + ", \n"
        b_t[str(i.number)] = "params : " + params + "\nvalues : " + values
    #Log dei risultati su neptune
    # run["best_trials"] = b_t
    #Ottiene la trial con l'accuracy migliore
    highest_trial = max(best_trials, key=lambda t: t.values[1])

    #Ottiene i parametri migliori e l'accuracy migliore
    best_params = highest_trial.params
    best_score = highest_trial.values[1]

    print("Best score:", best_score)
    print("Best params:", best_params)

    # Ottimizzazione con Scikit-Learn

    #Griglia per RandomSearch
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}

    #Griglia per GridSearch
    # grid_grid = {'n_estimators': [386, 1068, 1379, 572, 1255, 863, 400],
    #              'max_features': ['sqrt'],
    #              'max_depth': [290, 240, 280, 20, 30, 150, 70, 100, 90],
    #              'min_samples_split': [30, 35, 40],
    #              'min_samples_leaf': [4, 5, 2, 3, 7],
    #              'bootstrap': [True]}

    # # Create the model
    # bestModel = RandomForestClassifier(random_state=42)
    #
    # # Random search of parameters, using 5 fold cross validation,
    # rf_random = RandomizedSearchCV(estimator= bestModel, param_distributions=random_grid, n_iter=300, cv=5, verbose=2, n_jobs=-1)
    # rf_random.fit(X_train, y_train)
    #
    ##Recupera i parametri migliori
    # bestParams = rf_random.best_params_
    #
    # print("Random Forest best Params")
    # print(rf_random.best_params_)

    # Grid search for best parameters
    # grid_search = GridSearchCV(estimator=bestModel, param_grid=grid_grid, cv=5, n_jobs=-1, verbose=2)
    # bestF = grid_search.fit(X_train, y_train)
    # bestParams = grid_search.best_params_
    # print(grid_search.best_params_)

    # bestModel = RandomForestClassifier(n_estimators= bestParams["n_estimators"], min_samples_split= bestParams["min_samples_split"],
    #                                    min_samples_leaf= bestParams['min_samples_leaf'], max_features = "sqrt",
    #                                    max_depth=bestParams['max_depth'], bootstrap=True, random_state=42)
    #Log dei migliori parametri su neptune
    # run["parameters"] = best_params

    #Crea il modello con i parametri migliori
    bestModel = RandomForestClassifier(**best_params, random_state=42)
    bestModel.fit(X_train, y_train)

    #Calcola l'accuracy sul training set
    traningPredictions = bestModel.predict(X_train)
    trainingAccuracy = accuracy_score(y_train, traningPredictions)

    # Make predictions e calcola le metriche sul test set
    bestPredictions = bestModel.predict(x_test)
    bestAccuracy = accuracy_score(y_test, bestPredictions)
    bestPrecision = precision_score(y_test, bestPredictions, average='weighted')
    bestRecall = recall_score(y_test, bestPredictions, average='weighted')
    bestF1 = f1_score(y_test, bestPredictions, average='weighted')

    #Log dei risultati su neptune
    # run["training_accuracy"] = trainingAccuracy
    # run["test_accuracy"] = bestAccuracy
    # run["test_precision"] = bestPrecision
    # run["test_recall"] = bestRecall
    # run["test_f1"] = bestF1
    # run["confusion_matrix"] = npt_sk.create_confusion_matrix_chart(bestModel, X_train, x_test, y_train, y_test)
    # run["classification_report"] = npt_sk.create_classifier_summary(bestModel, X_train, x_test, y_train, y_test)

    print("METRICS")
    print("Training accuracy: ", trainingAccuracy)
    print("Test accuracy: ", bestAccuracy)
    print("Test Precision : ", bestPrecision)
    print("Test Recall : ", bestRecall)
    print("Test F1 : ", bestF1)
    print("Classification Report: ")
    print(classification_report(y_test, bestPredictions))

    # plotting della learning curve e della confusion matrix
    fig1 = plt.figure()
    ax = plt.subplot(1, 1, 1)
    skplt.estimators.plot_learning_curve(bestModel, X_train, y_train, title="Learning Curve Random Forest",
                                         scoring="accuracy",
                                         shuffle=True, figsize=(6, 4), ax=ax)
    plt.show()
    #Log del grafico su neptune
    # run["learning_curve"].upload(fig1)
    fig2 = plt.figure()
    ax = plt.subplot(1, 1, 1)
    skplt.metrics.plot_confusion_matrix(y_test, bestPredictions, normalize=True, title="Confusion Matrix Random Forest",
                                        ax=ax)
    plt.show()
    #Log del grafico su neptune
    # run["confusion_matrix"].upload(fig2)

    # run.stop()

#Metodo per l'allenamento e testing del modello AdaBoost
def adaboost_test_model(X_train, y_train, x_test, y_test, strategy):
    #Crea la run per il log dei dati su Neptune
    # if strategy != "Subject Independent":
    #     run = neptune.init_run(
    #         project="",
    #         api_token="",
    #         name="AdaBoost",
    #         tags=["AdaBoost", strategy]
    #     )  # your credentials

    #Imposta i parametri del modello
    params = {'n_estimators': 240,
              'learning_rate': 0.0001,
              'algorithm': 'SAMME.R',
              "max_depth": 3}

    #Crea il modello con i parametri specificati
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=params["max_depth"]),
                               n_estimators=params["n_estimators"], algorithm=params["algorithm"],
                               learning_rate=params["learning_rate"], random_state=42)

    #Log dei parametri su neptune
    # if strategy != "Subject Independent":
    #     run["parameters"] = params

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(x_test)

    #Calcola l'accuracy sul training set
    traningPredictions = model.predict(X_train)
    trainingAccuracy = accuracy_score(y_train, traningPredictions)

    #Calcola le metriche sul test set
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    #Log dei risultati su neptune
    # if strategy != "Subject Independent":
    #     run["training_accuracy"] = trainingAccuracy
    #     run["test_accuracy"] = accuracy
    #     run["test_precision"] = precision
    #     run["test_recall"] = recall
    #     run["test_f1"] = f1
    #     run["confusion_matrix"] = npt_sk.create_confusion_matrix_chart(model, X_train, x_test, y_train, y_test)
    #     run["classification_report"] = npt_sk.create_classifier_summary(model, X_train, x_test, y_train, y_test)

    print("METRICS")
    print("Test accuracy: ", accuracy)
    print("Test Precision : ", precision)
    print("Test Recall : ", recall)
    print("Test F1 : ", f1)

    if strategy != "Subject Independent":
        # plotting della learning curve e della confusion matrix
        fig1 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        skplt.estimators.plot_learning_curve(model, X_train, y_train, title="Learning Curve Adaboost",
                                             scoring="accuracy",
                                             shuffle=True, figsize=(6, 4), ax=ax)
        plt.grid()
        plt.show()
        #Log del grafico su neptune
        # run["learning_curve"].upload(fig1)
        fig2 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        skplt.metrics.plot_confusion_matrix(y_test, predictions, normalize=True, title="Confusion Matrix Adaboost",
                                            ax=ax)
        plt.show()
        #Log del grafico su neptune
        # run["confusion_matrix"].upload(fig2)

        # run.stop()

    return accuracy, trainingAccuracy, precision, recall, f1, params

#Metodo per l'ottimizzazione dei parametri del modello AdaBoost
def adaboost_optimization(X_train, y_train, x_test, y_test, strategy):
    #Sampler per l'ottimizzazione optuna
    # sampler = optuna.samplers.TPESampler(multivariate=True)
    sampler = optuna.samplers.RandomSampler()
    # sampler = optuna.samplers.QMCSampler(qmc_type='sobol', scramble=True, seed = 42)

    #Pruner per l'ottimizzazione optuna
    # pruning = optuna.pruners.HyperbandPruner(min_resource=5, max_resource="auto", reduction_factor=3)
    pruning = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    # pruning = optuna.pruners.SuccessiveHalvingPruner()

    #Crea la run per il log dei dati su Neptune
    # run = neptune.init_run(
    #     project="",
    #     api_token="",
    #     name="Adaboost Optimization",
    #     tags=["Adaboost Optimization", "AdaBoost", strategy, str(sampler).split(".")[-1].split("(")[0],
    #           str(pruning).split(".")[-1].split("(")[0].split(" ")[0]]
    # )  # your credentials

    #Funzione obiettivo per l'ottimizzazione
    def objective(trial):
        #Imposta i parametri da ottimizzare
        n_estimators = trial.suggest_int("n_estimators", 200, 2000, 10)
        learning_rate = trial.suggest_categorical("learning_rate", [0.0001, 0.001, 0.01, 0.1, 1.0, 1.1, 1.2])
        algorithm = trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"])
        max_depth = trial.suggest_int("max_depth", 1, 10, 1)

        #Crea il modello con i parametri specificati
        model = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=max_depth),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=42,
        )

        #Esegue il training del modello e calcola l'accuracy
        model.fit(X_train, y_train)

        trainingPredictions = model.predict(X_train)
        trainingAccuracy = accuracy_score(y_train, trainingPredictions)

        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        return trainingAccuracy - accuracy, accuracy

    # Optuna-scikilearn optimization con k-fold validation
    ##Parametri da ottimizzare
    # param_distribution = {
    #     "n_estimators": optuna.distributions.IntDistribution(200, 2000, step=10),
    #     "learning_rate": optuna.distributions.CategoricalDistribution([0.0001, 0.001, 0.01, 0.1, 1.0, 1.1, 1.2]),
    #     "algorithm": optuna.distributions.CategoricalDistribution(["SAMME", "SAMME.R"]),
    #     "base_estimator": optuna.distributions.CategoricalDistribution([DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2),
    #                           DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4),
    #                           DecisionTreeClassifier(max_depth=5)]),
    # }
    #
    # model = AdaBoostClassifier(random_state=42)
    ##Crea la ricerca con optuna
    # optuna_search = optuna.integration.OptunaSearchCV(model, param_distribution, n_trials=100, n_jobs=-1, cv=5,
    #                                                   enable_pruning=False, random_state=42, verbose=2)
    # optuna_search.fit(X_train, y_train)
    # y_pred = optuna_search.predict(x_test)
    #
    # best_params = optuna_search.best_params_
    # print("Best parameters found: ", optuna_search.best_params_)
    # print("Best score found: ", optuna_search.best_score_)
    #
    # trainingPredictions = optuna_search.predict(X_train)
    # trainingAccuracy = accuracy_score(y_train, trainingPredictions)
    # testAccuracy = accuracy_score(y_test, y_pred)

    # Optuna Optimization
    #Crea lo studio per l'ottimizzazione
    study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler, pruner=pruning)
    #Avvia l'ottimizzazione
    study.optimize(objective, n_trials=100)

    #Recupera i risultati dell'ottimizzazione
    best_trials = study.best_trials
    b_t = dict()
    print("Best trials:", len(best_trials))
    for i in best_trials:
        params = ""
        for j in i.params:
            params += str(j) + " : " + str(i.params[j]) + " \n"
        values = ""
        for j in i.values:
            values += str(j) + ", \n"
        b_t[str(i.number)] = "params : " + params + "\nvalues : " + values
    #Log dei risultati su Neptune
    # run["best_trials"] = b_t
    #Recupera la trial con l'accuracy migliore
    highest_trial = max(best_trials, key=lambda t: t.values[1])

    #Recupera i parametri della trial migliore
    best_params = highest_trial.params
    #Rimuove il parametro max_depth poiche' fa riferimento al valore del parametro base_estimator
    mdvalue = best_params.pop("max_depth")
    #Recupera l'accuracy della trial migliore
    best_score = highest_trial.values[1]

    print("Best score:", best_score)
    print("Best params:", best_params)

    # Ottimizzazione con Scikit-learn
    ##Specifica la griglia per la Random Search
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    #
    ##Specifica la griglia per la GridSearch
    # grid_grid = {'n_estimators': [386, 1068, 1379, 572, 1255, 863, 400],
    #              'max_features': ['sqrt'],
    #              'max_depth': [290, 240, 280, 20, 30, 150, 70, 100, 90],
    #              'min_samples_split': [30, 35, 40],
    #              'min_samples_leaf': [4, 5, 2, 3, 7],
    #              'bootstrap': [True]}
    #
    # # Create the model
    # bestModel = RandomForestClassifier(random_state=42)
    #
    # # Random search of parameters, using 5 fold cross validation,
    # rf_random = RandomizedSearchCV(estimator= bestModel, param_distributions=random_grid, n_iter=300, cv=5, verbose=2, n_jobs=-1)
    # rf_random.fit(X_train, y_train)
    #
    #Recupera i parametri migliori dalla RandomSearch
    # bestParams = rf_random.best_params_
    #
    # print("Random Forest best Params")
    # print(rf_random.best_params_)

    # Grid search for best parameters
    # grid_search = GridSearchCV(estimator=bestModel, param_grid=grid_grid, cv=5, n_jobs=-1, verbose=2)
    # bestF = grid_search.fit(X_train, y_train)
    #Recupera i parametri migliori dalla GridSearch
    # bestParams = grid_search.best_params_
    # print(grid_search.best_params_)

    #Crea il weak learner per AdaBoost con il parametro max_depth recuperato in precedenza
    estimator = DecisionTreeClassifier(max_depth=mdvalue)

    #Crea il modello
    bestModel = AdaBoostClassifier(base_estimator=estimator,
                                   **best_params, random_state=42)
    #Allena il modello
    bestModel.fit(X_train, y_train)

    #Log dei risultati su Neptune
    # best_params["base_estimator"] = stringify_unsupported(estimator)
    # run["parameters"] = best_params

    #Calcola l'accuracy sul training_set
    traningPredictions = bestModel.predict(X_train)
    trainingAccuracy = accuracy_score(y_train, traningPredictions)

    # Make predictions e calcola le metriche sul test_set
    bestPredictions = bestModel.predict(x_test)
    bestAccuracy = accuracy_score(y_test, bestPredictions)
    bestPrecision = precision_score(y_test, bestPredictions, average='weighted')
    bestRecall = recall_score(y_test, bestPredictions, average='weighted')
    bestF1 = f1_score(y_test, bestPredictions, average='weighted')

    #Log dei risultati su Neptune
    # run["training_accuracy"] = trainingAccuracy
    # run["test_accuracy"] = bestAccuracy
    # run["test_precision"] = bestPrecision
    # run["test_recall"] = bestRecall
    # run["test_f1"] = bestF1
    # run["confusion_matrix"] = npt_sk.create_confusion_matrix_chart(bestModel, X_train, x_test, y_train, y_test)
    # run["classification_report"] = npt_sk.create_classifier_summary(bestModel, X_train, x_test, y_train, y_test)

    print("METRICS")
    print("Training accuracy: ", trainingAccuracy)
    print("Test accuracy: ", bestAccuracy)
    print("Test Precision : ", bestPrecision)
    print("Test Recall : ", bestRecall)
    print("Test F1 : ", bestF1)
    print("Classification Report: ")
    print(classification_report(y_test, bestPredictions))

    # plotting della learning curve e della confusion matrix
    fig1 = plt.figure()
    ax = plt.subplot(1, 1, 1)
    skplt.estimators.plot_learning_curve(bestModel, X_train, y_train, title="Learning Curve AdaBoost",
                                         scoring="accuracy",
                                         shuffle=True, figsize=(6, 4), ax=ax)
    plt.show()
    #Log del grafico su Neptune
    # run["learning_curve"].upload(fig1)
    fig2 = plt.figure()
    ax = plt.subplot(1, 1, 1)
    skplt.metrics.plot_confusion_matrix(y_test, bestPredictions, normalize=True, title="Confusion Matrix AdaBoost",
                                        ax=ax)
    plt.show()
    #Log del grafico su nNeptune
    # run["confusion_matrix"].upload(fig2)

    # run.stop()

#Metodo per la creazione del modello di Gradient Boosting
def gradient_boosting_early_stopping_test_model(X_train, y_train, x_test, y_test, strategy):
    #Crea la run per il log dei dati su neptune
    # if strategy != "Subject Independent":
    #     run = neptune.init_run(
    #         project="",
    #         api_token="",
    #         name="GradientBoosting",
    #         tags=["GradientBoosting", strategy]
    #     )  # your credentials

    #Parametri per il modello
    pms = {'learning_rate': 1,
           'max_depth': 2,
           'n_estimators': 350}

    #Log dei parametri su Neptune
    # if strategy != "Subject Independent":
    #     run["parameters"] = pms

    # Create the model
    model = GradientBoostingClassifier(**pms, random_state=42)
    #Allena il modello
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(x_test)

    # plotting della learning curve e della confusion matrix
    if strategy != "Subject Independent":
        fig1 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        skplt.estimators.plot_learning_curve(model, X_train, y_train, title="Learning Curve Gradient Boosting",
                                             scoring="accuracy",
                                             shuffle=True, figsize=(6, 4), ax=ax)
        plt.grid()
        plt.show()
        #Log del grafico su Neptune
        # run["learning_curve"].upload(fig1)
        fig2 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        skplt.metrics.plot_confusion_matrix(y_test, predictions, normalize=True,
                                            title="Confusion Matrix Gradient Boosting", ax=ax)
        plt.show()
        #Log del grafico su Neptune
        # run["confusion_matrix"].upload(fig2)

    #Calcola l'accuracy sul training_set
    traningPredictions = model.predict(X_train)
    trainingAccuracy = accuracy_score(y_train, traningPredictions)

    #Calcola le metriche sul test_set
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    print("METRICS")
    print("Test accuracy: ", accuracy)
    print("Test Precision : ", precision)
    print("Test Recall : ", recall)
    print("Test F1 : ", f1)

    #Log dei risultati su Neptune
    # if strategy != "Subject Independent":
    #     run["training_accuracy"] = trainingAccuracy
    #     run["test_accuracy"] = accuracy
    #     run["test_precision"] = precision
    #     run["test_recall"] = recall
    #     run["test_f1"] = f1
    #     run["confusion_matrix"] = npt_sk.create_confusion_matrix_chart(model, X_train, x_test, y_train, y_test)
    #     run["classification_report"] = npt_sk.create_classifier_summary(model, X_train, x_test, y_train, y_test)

        # run.stop()

    return accuracy, trainingAccuracy, precision, recall, f1, pms

#Metodo per l'ottimizzazione del modello di Gradient Boosting
def gradient_boosting_optimization(X_train, y_train, x_test, y_test, strategy):
    #Sampler per l'ottimizzazione optuna
    # sampler = optuna.samplers.TPESampler(multivariate=True)
    # sampler = optuna.samplers.RandomSampler()
    sampler = optuna.samplers.QMCSampler(qmc_type='sobol', scramble=True, seed=42)

    #Pruner per l'ottimizzazione optuna
    # pruning = optuna.pruners.HyperbandPruner(min_resource=5, max_resource="auto", reduction_factor=3)
    # pruning = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    pruning = optuna.pruners.SuccessiveHalvingPruner()

    #Crea la run per il log dei dati su neptune
    # run = neptune.init_run(
    #     project="",
    #     api_token="",
    #     name="GradientBoosting",
    #     tags=["GradientBoosting", "GradientBoosting Optimization", strategy, str(sampler).split(".")[-1].split("(")[0],
    #           str(pruning).split(".")[-1].split("(")[0].split(" ")[0]]
    # )  # your credentials


    # Funzione Obiettivo per ottimizzazione Optuna
    def objective(trial):
        #Parametri da ottimizzare
        n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
        learning_rate = trial.suggest_float("learning_rate", 0.0001, 1)
        max_depth = trial.suggest_int("max_depth", 2, 10, step=1)

        #Crea il modello
        model = GradientBoostingClassifier(n_estimators=n_estimators,
                                           learning_rate=learning_rate,
                                           max_depth=max_depth,
                                           random_state=42)

        #Allena il modello e calcola l'accuracy
        model.fit(X_train, y_train)

        trainingPredictions = model.predict(X_train)
        trainingAccuracy = accuracy_score(y_train, trainingPredictions)

        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        return trainingAccuracy - accuracy, accuracy

    # Optuna-scikilearn optimization con k-fold validation
    ##Parametri da ottimizzare
    # param_distribution = {
    #     "n_estimators": optuna.distributions.IntDistribution(100, 500, step=50),
    #     "learning_rate": optuna.distributions.FloatDistribution(0.0001,1),
    #     "max_depth": optuna.distributions.IntDistribution(2, 10, step=1),
    # }
    #
    # model = GradientBoostingClassifier(random_state=42)
    # optuna_search = optuna.integration.OptunaSearchCV(model, param_distribution, n_trials=100, n_jobs=-1, cv=5,
    #                                                   enable_pruning=False, random_state=42, verbose=2)
    # optuna_search.fit(X_train, y_train)
    # y_pred = optuna_search.predict(x_test)
    #
    # best_params = optuna_search.best_params_
    # print("Best parameters found: ", optuna_search.best_params_)
    # print("Best score found: ", optuna_search.best_score_)
    #
    # trainingPredictions = optuna_search.predict(X_train)
    # trainingAccuracy = accuracy_score(y_train, trainingPredictions)
    # testAccuracy = accuracy_score(y_test, y_pred)
    #
    # print("Testing Accuracy Opti: ", testAccuracy)
    # print("Training Accuracy Opti: ", trainingAccuracy)

    # Optuna Optimization
    #Crea lo studio per l'ottimizzazione otpuna
    study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler, pruner=pruning)
    #Avvia l'ottimizzazione
    study.optimize(objective, n_trials=100)

    #Recupera i risultati dell'ottimizzazione
    best_trials = study.best_trials
    b_t = dict()
    print("Best trials:", len(best_trials))
    for i in best_trials:
        params = ""
        for j in i.params:
            params += str(j) + " : " + str(i.params[j]) + " \n"
        values = ""
        for j in i.values:
            values += str(j) + ", \n"
        b_t[str(i.number)] = "params : " + params + "\nvalues : " + values
    #Log dei risultati su Neptune
    # run["best_trials"] = b_t
    #Recupera la trial con l'accuracy migliore
    highest_trial = max(best_trials, key=lambda t: t.values[1])

    #Recupera i parametri e l'accuracy migliore
    best_params = highest_trial.params
    best_score = highest_trial.values[1]
    #Log dei parametri su Neptune
    # run["parameters"] = best_params

    print("Best score:", best_score)
    print("Best params:", best_params)

    # Ottimizzazione Scikit-Learn
    ##Parametri da ottimizzare
    # param = {
    #     'n_estimators': [150, 200, 250, 275],
    #     'learning_rate': [0.8, 0.9, 1, 1.1],
    #     'max_depth': [2, 3, 4]
    # }

    # model = GradientBoostingClassifier()
    # Random search of parameters, using 5fold cross validation,
    # rf_random = RandomizedSearchCV(estimator= model, param_distributions=param, n_iter=100, cv=5, verbose=2, n_jobs=-1)
    # rf_random.fit(X_train, y_train)
    #
    # print("Random Forest best Params")
    # print(rf_random.best_params_)
    #
    ##Recupera i parametri migliori dalla Random Search
    # bestParams = rf_random.best_params_

    # #Grid search for best parameters
    # grid_search = GridSearchCV(estimator=model, param_grid=param, cv=5, n_jobs=-1, verbose=2)
    # bestF = grid_search.fit(X_train, y_train)
    # print(grid_search.best_params_)
    ##Recupera i parametri migliori dalla Grid Search
    # bestParams = grid_search.best_params_

    # Bayesian Optimization
    # #Parametri da ottimizzare
    # space = {'n_estimators' : hp.choice('n_estimators', [150, 200, 250, 300, 350, 400, 450]),
    #         'learning_rate' : hp.choice('max_features', [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    #         'max_depth' : hp.choice('max_depth', [1, 2, 3, 4, 5])}
    #
    # #Objective function
    # def objective(params):
    #     gradBoost = GradientBoostingClassifier(**params)
    #     scores = cross_val_score(gradBoost, X_train, y_train, cv=5, scoring="accuracy", n_jobs = -1 )
    #     best_score = mean(scores)
    #     loss = 1 - best_score
    #     return {"loss": loss, "params": params, "status": STATUS_OK}
    #
    # trials = Trials()
    # best = fmin(fn = objective, space = space, algo=tpe.suggest, max_evals=100, trials=trials)
    #
    # print(space_eval(space, best))

    # BEST PARAMS TESTING
    model = GradientBoostingClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(x_test)

    # plotting delle learning curve e della confusion matrix
    if strategy != "Subject Independent":
        fig1 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        skplt.estimators.plot_learning_curve(model, X_train, y_train, title="Learning Curve Gradient Boosting",
                                             scoring="accuracy",
                                             shuffle=True, figsize=(6, 4), ax=ax)
        plt.grid()
        plt.show()
        #Log del grafico su Neptune
        # run["learning_curve"].upload(fig1)
        fig2 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        skplt.metrics.plot_confusion_matrix(y_test, predictions, normalize=True,
                                            title="Confusion Matrix Gradient Boosting", ax=ax)
        plt.show()
        #Log del grafico su Neptune
        # run["confusion_matrix"].upload(fig2)

    #Calcola l'accuracy sul training set
    traningPredictions = model.predict(X_train)
    trainingAccuracy = accuracy_score(y_train, traningPredictions)

    #Calcola le metriche sul test set
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    print("METRICS")
    print("Test accuracy: ", accuracy)
    print("Test Precision : ", precision)
    print("Test Recall : ", recall)
    print("Test F1 : ", f1)

    #Log delle metriche su Neptune
    # run["training_accuracy"] = trainingAccuracy
    # run["test_accuracy"] = accuracy
    # run["test_precision"] = precision
    # run["test_recall"] = recall
    # run["test_f1"] = f1
    # run["confusion_matrix"] = npt_sk.create_confusion_matrix_chart(model, X_train, x_test, y_train, y_test)
    # run["classification_report"] = npt_sk.create_classifier_summary(model, X_train, x_test, y_train, y_test)

    # run.stop()

#Funzione per l'allenamento e il testing del modello SVM
def svm_test_model(X_train, y_train, x_test, y_test, strategy):
    #Crea la run per il log dei dati su Neptune
    # if strategy != "Subject Independent":
    #     run = neptune.init_run(
    #         project="",
    #         api_token="",
    #         name="SVM",
    #         tags=["SVM", strategy]
    #     )  # your credentials

    #Parametri del modello
    params = {'C': 0.1,
              'gamma': 'scale',
              'kernel': 'linear'}

    #Log dei parametri su Neptune
    # if strategy != "Subject Independent":
    #     run["parameters"] = params

    #Crea il modello
    model = svm.SVC(**params)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    print("PREDICTIONS")
    predictions = model.predict(x_test)

    #Calcola l'accuracy sul training set
    traningPredictions = model.predict(X_train)
    trainingAccuracy = accuracy_score(y_train, traningPredictions)

    #Calcola le metriche sul test set
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    #Log delle metriche su Neptune
    # if strategy != "Subject Independent":
    #     run["training_accuracy"] = trainingAccuracy
    #     run["test_accuracy"] = accuracy
    #     run["test_precision"] = precision
    #     run["test_recall"] = recall
    #     run["test_f1"] = f1
    #     run["confusion_matrix"] = npt_sk.create_confusion_matrix_chart(model, X_train, x_test, y_train, y_test)
    #     run["classification_report"] = npt_sk.create_classifier_summary(model, X_train, x_test, y_train, y_test)

    print("ACCURACY")
    print(accuracy_score(y_test, predictions))

    print("METRICS")
    print("Test accuracy: ", accuracy)
    print("Test Precision : ", precision)
    print("Test Recall : ", recall)
    print("Test F1 : ", f1)

    # plotting delle learning curve e della confusion matrix
    if strategy != "Subject Independent":
        fig1 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        skplt.estimators.plot_learning_curve(model, X_train, y_train, title="Learning Curve SVM", scoring="accuracy",
                                             shuffle=True, figsize=(6, 4), ax=ax)
        plt.grid()
        plt.show()
        #Log del grafico su Neptune
        # run["learning_curve"].upload(fig1)
        fig2 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        skplt.metrics.plot_confusion_matrix(y_test, predictions, normalize=True, title="Confusion Matrix SVM", ax=ax)
        plt.show()
        #Log del grafico su Neptune
        # run["confusion_matrix"].upload(fig2)

        # run.stop()

    return accuracy, trainingAccuracy, precision, recall, f1, params

#Metodo per l'ottimizzazione dei parametri del modello SVM
def svm_optimization(X_train, y_train, X_test, y_test, strategy):
    #Sampler per l'ottimizzazione Optuna
    sampler = optuna.samplers.TPESampler(multivariate=True)
    # sampler = optuna.samplers.RandomSampler()
    # sampler = optuna.samplers.QMCSampler(qmc_type='sobol', scramble=True, seed = 42)

    #Pruner per l'ottimizzazione Optuna
    pruning = optuna.pruners.HyperbandPruner(min_resource=5, max_resource="auto", reduction_factor=3)
    # pruning = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    # pruning = optuna.pruners.SuccessiveHalvingPruner()

    #Crea la run per il log dei dati su Neptune
    # if strategy != "Subject Independent":
    #     run = neptune.init_run(
    #         project="",
    #         api_token="",
    #         name="SVM",
    #         tags=["SVM", "SVM Optimization", strategy, str(sampler).split(".")[-1].split("(")[0],
    #               str(pruning).split(".")[-1].split("(")[0].split(" ")[0]]
    #     )  # your credentials

    # Ottimizzazione Scikit-Learn
    # Create the model
    # model = svm.SVC()

    # Params to examine
    # C_range = np.logspace(-1, 1, 3)
    # gamma_range = np.logspace(-1, 1, 3)
    # C_range = np.logspace(-5, 5, 10)
    # gamma_range = np.logspace(-5, 5, 10)
    # kernels = ['rbf', 'linear', "poly"]
    #
    ##Spazio di ricerca
    # random_grid = {'C': C_range,
    #                'gamma': gamma_range,
    #                'kernel': kernels}
    #
    # scoring = ["accuracy"]

    # Inizializza la GridSearch
    # kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # rf_random = GridSearchCV(estimator= model, param_grid=random_grid, cv=kfold, scoring = scoring, refit = "accuracy", verbose=2, n_jobs=-1)
    #Inizializza la RandomSearch
    # rf_random = RandomizedSearchCV(estimator= model, param_grid=random_grid, cv=kfold, scoring = scoring, refit = "accuracy", verbose=2, n_jobs=-1, random_state=42, n_iter = 100)
    # rf_random.fit(X_train, y_train)
    #
    # print("BEST PARAMS")
    # print(rf_random.best_params_)

    # Bayesian Optimization
    ##Spazio di ricerca
    # space = {'C' : hp.choice('C', C_range),
    #         'gamma' : hp.choice('gamma', gamma_range),
    #         'kernel' : hp.choice('kernel', kernels)}
    #
    # #Objective function
    # def objective(params):
    #     svc = SVC(**params)
    #     scores = cross_val_score(svc, X_train, y_train, cv=kfold, scoring="accuracy", n_jobs = -1 )
    #     best_score = mean(scores)
    #     loss = 1 - best_score
    #     return {"loss": loss, "params": params, "status": STATUS_OK}
    #
    # trials = Trials()
    # best = fmin(fn = objective, space = space, algo=tpe.suggest, max_evals=100, trials=trials)

    # print(space_eval(space, best))

    # Ottimizzazione optuna

    # Funzione Obiettivo per ottimizzazione optuna
    def objective(trial):
        #Parametri da ottimizzare
        C = trial.suggest_categorical("C", [0.1, 1, 10, 100, 1000])
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])

        #Crea il modello
        model = SVC(C=C, gamma=gamma, kernel=kernel, random_state=42)

        model.fit(X_train, y_train)

        #Calcola l'accuracy sul training_set
        trainingPredictions = model.predict(X_train)
        trainingAccuracy = accuracy_score(y_train, trainingPredictions)

        #Calcola l'accuracy sul test_set
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        return trainingAccuracy - accuracy, accuracy

    # Crea lo studio per l'ottimizzazione optuna
    study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler, pruner=pruning)
    #Avvia l'ottimizzazione
    study.optimize(objective, n_trials=100)

    #Recupera i risultati dell'ottimizzazione
    best_trials = study.best_trials
    b_t = dict()
    print("Best trials:", len(best_trials))
    for i in best_trials:
        params = ""
        for j in i.params:
            params += str(j) + " : " + str(i.params[j]) + " \n"
        values = ""
        for j in i.values:
            values += str(j) + ", \n"
        b_t[str(i.number)] = "params : " + params + "\nvalues : " + values
    #Log dei risultati su Neptune
    # run["best_trials"] = b_t

    #Recupera il trial con l'accuracy migliore
    highest_trial = max(best_trials, key=lambda t: t.values[0])

    #Recupera i parametri migliori
    best_params = highest_trial.params
    #Log dei parametri migliori su Neptune
    # run["parameters"] = best_params
    #Recupera l'accuracy migliore
    best_score = highest_trial.values[0]

    print("Best score:", best_score)
    print("Best params:", best_params)

    # Optuna-scikilearn optimization con k-fold validation
    ##Spazio di ricerca
    # param_distribution = {
    #     "C": optuna.distributions.CategoricalDistribution([0.1, 1, 10, 100, 1000]),
    #     "gamma": optuna.distributions.CategoricalDistribution(["scale", "auto"]),
    #     "kernel": optuna.distributions.CategoricalDistribution(["rbf", "linear", "poly"]),
    # }
    #
    # model = SVC(random_state=42)
    # optuna_search = optuna.integration.OptunaSearchCV(model, param_distribution, n_trials=100, n_jobs=-1, cv=5,
    #                                                   enable_pruning=False, random_state=42, verbose=2)
    # optuna_search.fit(X_train, y_train)
    # y_pred = optuna_search.predict(X_test)
    #
    # best_params = optuna_search.best_params_
    # print("Best parameters found: ", optuna_search.best_params_)
    # print("Best score found: ", optuna_search.best_score_)
    #
    # trainingPredictions = optuna_search.predict(X_train)
    # trainingAccuracy = accuracy_score(y_train, trainingPredictions)
    # testAccuracy = accuracy_score(y_test, y_pred)
    #
    # print("Testing Accuracy Opti: ", testAccuracy)
    # print("Training Accuracy Opti: ", trainingAccuracy)

    #Log dei parametri migliori su Neptune
    # run["parameters"] = best_params

    #Crea il modello
    model = svm.SVC(**best_params)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    print("PREDICTIONS")
    predictions = model.predict(X_test)

    #Calcola l'accuracy sul training_set
    traningPredictions = model.predict(X_train)
    trainingAccuracy = accuracy_score(y_train, traningPredictions)

    #Calcola le metriche sul test_set
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    #Log delle metriche su Neptune
    # run["training_accuracy"] = trainingAccuracy
    # run["test_accuracy"] = accuracy
    # run["test_precision"] = precision
    # run["test_recall"] = recall
    # run["test_f1"] = f1
    # run["confusion_matrix"] = npt_sk.create_confusion_matrix_chart(model, X_train, X_test, y_train, y_test)
    # run["classification_report"] = npt_sk.create_classifier_summary(model, X_train, X_test, y_train, y_test)

    print("ACCURACY")
    print(accuracy_score(y_test, predictions))

    print("METRICS")
    print("Test accuracy: ", accuracy)
    print("Test Precision : ", precision)
    print("Test Recall : ", recall)
    print("Test F1 : ", f1)

    # plotting della learning curve e della confusion matrix
    fig1 = plt.figure()
    ax = plt.subplot(1, 1, 1)
    skplt.estimators.plot_learning_curve(model, X_train, y_train, title="Learning Curve SVM", scoring="accuracy",
                                         shuffle=True, figsize=(6, 4), ax=ax)
    plt.grid()
    plt.show()
    #Log del grafico su Neptune
    # run["learning_curve"].upload(fig1)
    fig2 = plt.figure()
    ax = plt.subplot(1, 1, 1)
    skplt.metrics.plot_confusion_matrix(y_test, predictions, normalize=True, title="Confusion Matrix SVM", ax=ax)
    plt.show()
    #Log del grafico su neptune
    # run["confusion_matrix"].upload(fig2)

    # run.stop()

#Metodo per l'allenamento e la validazione del modello CNN
def tensorflow_cnn_model(X_train, y_train, X_test, y_test, strategy):
    #Inizializza il logger di Neptune
    # if strategy != "Subject Independent":
    #     run = neptune.init_run(
    #         project="",
    #         api_token="",
    #         name="CNN",
    #         tags=["CNN", strategy]
    #     )  # your credentials

    #Parametri per il modello
    params = {'conv_layers': 3,
              'filters_0': 32,
              'kernel_size_0': 3,
              'max_pooling_0': 2,
              'filters_1': 64,
              'kernel_size_1': 3,
              "kernel_regularizer_1": 'l1',
              'max_pooling_1': 2,
              'filters_2': 128,
              'kernel_size_2': 3,
              'max_pooling_2': 2,
              'units_end': 128,
              'regularizer_end': 'l2',
              'dropout_end': 0.5
              }

    #Log dei parametri su Neptune
    # if strategy != "Subject Independent":
    #     run['parameters'] = params

    #Creazione del modello aggiungendo layer manualmente e non tramite ciclo for poiche' dava problemi con l'inseirmento dei layer dinamico
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu', kernel_regularizer="l1"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer="l2"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    #Compilazione del modello
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    #Funzione di earl-stopping per evitare l'overfitting
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

    #Allenamento del modello in 40 epoche
    history = model.fit(X_train, y_train, epochs=40, batch_size=32,
                        validation_data=(X_test, y_test), callbacks=[stop_early], verbose=2)

    acc = history.history['accuracy']
    loss = history.history['loss']

    #Recupera accuracy e lost per il test_set
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f'Test loss: {val_loss:.3f}, Test accuracy: {val_acc:.3f}')

    # Make predictions
    predictions = model.predict(X_test)

    #Recuper accuracy e loss per il training set
    trainingLoss, trainingAccuracy = model.evaluate(X_train, y_train)

    #Log delle metriche su Neptune
    # if strategy != "Subject Independent":
    #     run["training_accuracy"] = trainingAccuracy
    #     run["training_loss"] = trainingLoss
    #     run["test_accuracy"] = val_acc
    #     run["test_loss"] = val_loss


    print("METRICS")
    print("Test accuracy: ", val_acc)

    # plot del training score contro il test score, e della training loss contro la test loss
    if strategy != "Subject Independent":
        # Summarize History of Accuracy
        fig1 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.plot(history.history['accuracy'])
        ax.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        #Log del grafico su Neptune
        # run["model_accuracy"].upload(fig1)

        # Summarize History of loss
        fig2 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        #Log del grafico su Neptune
        # run["model_loss"].upload(fig2)

        # run.stop()

    return val_acc, val_loss, acc, loss, params

#Metodo per l'ottimizzazione del modello CNN
def tensorflow_cnn_optimization(X_train, y_train, x_test, y_test, strategy):
    #Sampler per ottimizzazione Optuna
    # sampler = optuna.samplers.TPESampler(multivariate=True)
    # sampler = optuna.samplers.RandomSampler()
    sampler = optuna.samplers.QMCSampler(qmc_type='sobol', scramble=True, seed = 42)

    #Pruner per ottimmizzazione Optuna
    # pruning = optuna.pruners.HyperbandPruner(min_resource=5, max_resource="auto", reduction_factor=3)
    # pruning = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    pruning = optuna.pruners.SuccessiveHalvingPruner()

    #Inizializza il logger di Neptune
    # if strategy != "Subject Independent":
    #     run = neptune.init_run(
    #         project="",
    #         api_token="",
    #         name="CNN",
    #         tags=["CNN", "CNN Optimization", strategy, str(sampler).split(".")[-1].split("(")[0],
    #               str(pruning).split(".")[-1].split("(")[0].split(" ")[0]]
    #     )  # your credentials

    # Ottimizzazione KerasTuner
    # #Funzione di ottimizzazione
    # def build_model(hp):
    #     inputs = tf.keras.Input(shape=(X_train.shape[1], 1))
    #     x = inputs
    #   #Seleziona il numero di convolution layer
    #     for i in range(hp.Int('conv_layers', 1, 5, default=3)):
            #Seleziona il numero di filtri e la dimensione del kernel
    #         x = tf.keras.layers.Conv1D(
    #             filters=hp.Int('filters_' + str(i), 4, 32, step=4, default=8),
    #             kernel_size=hp.Int('kernel_size_' + str(i), 2, 5),
    #             activation='relu',
    #             padding='same')(x)
    #          #Seleziona il tipo di pooling
    #         if hp.Choice('pooling_' + str(i), ['max', 'avg']) == 'max':
    #             x = tf.keras.layers.MaxPooling1D()(x)
    #         else:
    #             x = tf.keras.layers.AveragePooling1D()(x)
    #
    #         x = tf.keras.layers.BatchNormalization()(x)
    #         x = tf.keras.layers.ReLU()(x)
    #
    #     if hp.Choice('global_pooling', ['max', 'avg']) == 'max':
    #         x = tf.keras.layers.GlobalMaxPooling1D()(x)
    #     else:
    #         x = tf.keras.layers.GlobalAveragePooling1D()(x)
    #     outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    #
    #     model = tf.keras.Model(inputs, outputs)
    #
    #     #Seleziona il tipo di ottimizzatore
    #     optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
    #     model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #     return model
    #

    ##Inizializza i vari tuner
    # tuner = kt.Hyperband(build_model2,
    #                      objective='accuracy',
    #                      max_epochs=30,
    #                      factor=2,
    #                      hyperband_iterations=3,
    #                      distribution_strategy=tf.distribute.MirroredStrategy(),
    #                      directory='tempData',
    #                      overwrite=True,
    #                      project_name='DNN')

    # tuner = kt.RandomSearch(build_model2,
    #                      objective='val_accuracy',
    #                      max_trials=70,
    #                      executions_per_trial=5,
    #                      directory='tempData',
    #                      overwrite=True,
    #                      project_name='DNN')

    # tuner = kt.BayesianOptimization(build_model2, objective='val_accuracy', max_trials=100, directory="tempData", overwrite=True, project_name="DNN")

    ##Esegue la ricerca dei parametri
    # tuner.search(X_train, y_train, steps_per_epoch=int(np.ceil(X_train.shape[0]/32)), validation_data=(X_test, y_test),
    #              validation_steps=int(np.ceil(X_test.shape[0]/32)), epochs = 20,
    #              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)])
    # print("END SEARCH")
    #
    # # Get the optimal hyperparameters
    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # print("Best_Hps:")
    # print(best_hps.values)
    #
    ## Costruisce il modello con i parametri ottimali
    # h_model = tuner.hypermodel.build(best_hps)
    #
    ##Allena il modello
    # h_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss')], verbose=2)
    #
    ##Ottiene loss e accuracy del modello sul test_set
    # val_loss, val_acc = h_model.evaluate(X_test, y_test)
    # print("Test accuracy: ", val_acc)
    # print("Test loss: ", val_loss)

    #Ottimizzazione Optuna
    def objective(trial):

        #Inizializza preventivamente fino a 4 layer di pooling
        # Manual
        pooling1 = tf.keras.layers.AveragePooling1D(2)
        pooling2 = tf.keras.layers.AveragePooling1D(2)
        pooling3 = tf.keras.layers.AveragePooling1D(2)
        pooling4 = tf.keras.layers.AveragePooling1D(2)
        if trial.suggest_categorical('pooling_1', ['max', 'avg']) == 'max':
            pooling1 = tf.keras.layers.MaxPooling1D(2)
        if trial.suggest_categorical('pooling_2', ['max', 'avg']) == 'max':
            pooling2 = tf.keras.layers.MaxPooling1D(2)
        if trial.suggest_categorical('pooling_3', ['max', 'avg']) == 'max':
            pooling3 = tf.keras.layers.MaxPooling1D(2)
        if trial.suggest_categorical('pooling_4', ['max', 'avg']) == 'max':
            pooling4 = tf.keras.layers.MaxPooling1D(2)

        #Inizializza preventivamente fino a 4 convoluzionali con i corrispettivi layer di pooling
        #Biosgna commentare un layer alla volta per testare un numero variabile di layer
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=trial.suggest_int('filters_1', 4, 64, step=4),
                                   kernel_size=trial.suggest_int('kernel_size_1', 2, 5),
                                   kernel_regularizer=trial.suggest_categorical('regularizer_1', ['l1', 'l2', "l1_l2"]),
                                   input_shape=(X_train.shape[1], 1),
                                   activation='relu'),
            pooling1,
            tf.keras.layers.Conv1D(filters=trial.suggest_int('filters_2', 4, 64, step=4),
                                   kernel_size=trial.suggest_int('kernel_size_2', 2, 5),
                                   kernel_regularizer=trial.suggest_categorical('regularizer_2',
                                                                                ['l1', 'l2', "l1_l2"]),
                                   activation='relu'),
            pooling2,
            tf.keras.layers.Conv1D(filters=trial.suggest_int('filters_3', 4, 64, step=4),
                                   kernel_size=trial.suggest_int('kernel_size_3', 2, 5),
                                   kernel_regularizer=trial.suggest_categorical('regularizer_3',
                                                                                ['l1', 'l2', "l1_l2"]),
                                   activation='relu'),
            pooling3,
            tf.keras.layers.Conv1D(filters=trial.suggest_int('filters_4', 4, 64, step=4),
                                   kernel_size=trial.suggest_int('kernel_size_4', 2, 5),
                                   kernel_regularizer=trial.suggest_categorical('regularizer_4',
                                                                                ['l1', 'l2', "l1_l2"]),
                                   activation='relu'),
            pooling4,
            tf.keras.layers.Flatten(),
            #Layer finale Dense
            tf.keras.layers.Dense(trial.suggest_int("units_end", 70, 150, 10),
                                  activation='relu',
                                  kernel_regularizer=trial.suggest_categorical("regularizer_end",
                                                                               ['l1', 'l2', "l1_l2"])),
            # tf.keras.layers.Dropout(tf.keras.layers.Dropout(rate=trial.suggest_float('dropout_end', 0, 0.5))),
            #Layer Finale di output
            tf.keras.layers.Dense(4, activation='softmax')
        ])

        #Decide il learning_rate
        hp_learning_rate = trial.suggest_categorical('learning_rate', [1e-2, 1e-3, 1e-4])

        #Compila il modello
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        #Definisce una funzione di early_stopping
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

        #Allena il modello
        model.fit(X_train, y_train, epochs=40, validation_data=(x_test, y_test), callbacks=[stop_early], verbose=2)

        return model.evaluate(x_test, y_test)[0], model.evaluate(x_test, y_test)[1]

    #Crea lo studio di ottimizzazione con Optuna
    study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler, pruner=pruning)
    #Avvia la ricerca
    study.optimize(objective, n_trials=100)

    #Ottiene i risultati
    best_trials = study.best_trials
    b_t = dict()
    print("Best trials:", len(best_trials))
    for i in best_trials:
        params = ""
        for j in i.params:
            params += str(j) + " : " + str(i.params[j]) + " \n"
        values = ""
        for j in i.values:
            values += str(j) + ", \n"
        b_t[str(i.number)] = "params : " + params + "\nvalues : " + values
    #Log dei risultati su Neptune
    # run["best_trials"] = b_t
    #Ottiene la trial con l'accuracy maggiore
    highest_trial = max(best_trials, key=lambda t: t.values[1])

    #Ottiene i parametri migliori
    best_params = highest_trial.params
    #Ottiene l'accuracy migliore
    best_score = highest_trial.values[1]

    #Log dei parametri su Neptune
    # run["parameters"] = best_params

    print("Best score:", best_score)
    print("Best params:", best_params)


    # Crea il modello manualmente commentando i layer non necessari
    pooling1 = tf.keras.layers.AveragePooling1D(2)
    pooling2 = tf.keras.layers.AveragePooling1D(2)
    pooling3 = tf.keras.layers.AveragePooling1D(2)
    pooling4 = tf.keras.layers.AveragePooling1D(2)
    if best_params['pooling_1']== 'max':
        pooling1 = tf.keras.layers.MaxPooling1D(2)
    if best_params['pooling_2'] == 'max':
        pooling2 = tf.keras.layers.MaxPooling1D(2)
    if best_params['pooling_3'] == 'max':
        pooling3 = tf.keras.layers.MaxPooling1D(2)
    if best_params['pooling_4'] == 'max':
        pooling4 = tf.keras.layers.MaxPooling1D(2)

    bestModel = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=best_params['filters_1'],
                               kernel_size=best_params['kernel_size_1'],
                               kernel_regularizer=best_params['regularizer_1'],
                               input_shape=(X_train.shape[1], 1),
                               activation='relu'),
        pooling1,
        tf.keras.layers.Conv1D(filters=best_params['filters_2'],
                               kernel_size=best_params['kernel_size_2'],
                               kernel_regularizer=best_params['regularizer_2'],
                               activation='relu'),
        pooling2,
        tf.keras.layers.Conv1D(filters=best_params['filters_3'],
                               kernel_size=best_params['kernel_size_3'],
                               kernel_regularizer=best_params['regularizer_3'],
                               activation='relu'),
        pooling3,
        tf.keras.layers.Conv1D(filters=best_params['filters_4'],
                               kernel_size=best_params['kernel_size_4'],
                               kernel_regularizer=best_params['regularizer_4'],
                               activation='relu'),
        pooling4,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(best_params["units_end"],
                              activation='relu',
                              kernel_regularizer=best_params["regularizer_end"]),
        # tf.keras.layers.Dropout(tf.keras.layers.Dropout(rate=trial.suggest_float('dropout_end', 0, 0.5))),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Decide il learning_rate
    hp_learning_rate = best_params['learning_rate']

    # Compila il modello
    bestModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # Definisce una funzione di early_stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

    # Allena il modello
    history = bestModel.fit(X_train, y_train, epochs=40, validation_data=(x_test, y_test), callbacks=[stop_early],
                            verbose=2)

    acc = history.history['accuracy']
    loss = history.history['loss']

    #Calcola l'accuracy e loss per test_set
    val_loss, val_acc = bestModel.evaluate(x_test, y_test)
    print(f'Test loss: {val_loss:.3f}, Test accuracy: {val_acc:.3f}')

    # Make predictions
    predictions = bestModel.predict(x_test)

    # Calcola l'accuracy e loss per training_set
    trainingLoss, trainingAccuracy = bestModel.evaluate(X_train, y_train)

    # Log dei risultati su Neptune
    # if strategy != "Subject Independent":
    #     run["training_accuracy"] = trainingAccuracy
    #     run["training_loss"] = trainingLoss
    #     run["test_accuracy"] = val_acc
    #     run["test_loss"] = val_loss


    print("METRICS")
    print("Test accuracy: ", val_acc)

    # plot del training score contro il test score, e della training loss contro la test loss
    if strategy != "Subject Independent":
        # Summarize History of Accuracy
        fig1 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.plot(history.history['accuracy'])
        ax.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        #Log del grafico su Neptune
        # run["model_accuracy"].upload(fig1)

        # Summarize History of loss
        fig2 = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        #Log del grafico su neptune
        # run["model_loss"].upload(fig2)

        # run.stop()

#Funzione per testare AutoKeras ed ottenere un modello ottimale in automatico
def auto_keras_test(x_train, y_train, x_test, y_test, strategy):
    #Inizializza il logger di Neptune
    # if strategy != "Subject Independent":
    #     run = neptune.init_run(
    #         project="",
    #         api_token="",
    #         name="AutoKeras",
    #         tags=["AutoKeras", strategy]
    #     )  # your credentials

    #Inizializza la ricerca
    clf = ak.StructuredDataClassifier(max_trials=100, num_classes=4, overwrite=True)

    #Definisce una funzione di early_stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

    #Esegue la ricerca
    clf.fit(x_train, y_train, epochs=40, validation_data=(x_test, y_test), callbacks=[stop_early])

    #Ottiene il modello migliore
    bestModel = clf.export_model()


    stringList = []
    bestModel.summary(print_fn=lambda x: stringList.append(x))
    short_model_summary = "\n".join(stringList)
    #Log del modello migliore su Neptune
    # run["model_summary"] = short_model_summary

    #Valuta il modello migliore
    train_accuracy = clf.evaluate(x_train, y_train)
    test_accuracy = clf.evaluate(x_test, y_test)
    #Log delle metriche su Neptune
    # run["training_accuracy"] = train_accuracy[1]
    # run["test_accuracy"] = test_accuracy[1]
    # run["training_loss"] = train_accuracy[0]
    # run["test_loss"] = test_accuracy[0]
    print("Train accuracy = ", train_accuracy)
    print("Test accuracy = ", test_accuracy)


    # run.stop()
