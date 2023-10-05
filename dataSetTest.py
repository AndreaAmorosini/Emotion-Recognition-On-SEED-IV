from scipy.io import loadmat
import pandas as pd

#Il Seguente Codice serve come esplorazione del dataset sotto forma di Dataframe Pandas

#Label per ogni Trial all'interno di ogni sessione per ogni soggetto (per i dati degli occhi le trial corrispondono alle clips mostrate)
#0 : neutral, 1: sad, 2: fear, 3: happy
#session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3];
#session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1];
#session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0];

#RAWDATA EEG (Dati Raw dei segnali EEG)
print('RAWDATA EEG - Subject 1 - Session 1')

channelsEEG = ["FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8","FT7","FC5","FC3","FC1",
           "FCZ","FC2","FC4","FC6","FT8","T7","C5","C3","C1","CZ","C2","C4","C6","T8","TP7","CP5","CP3","CP1",
           "CPZ","CP2","CP4","CP6","TP8","P7","P5","P3","P1","PZ","P2","P4","P6","P8","PO7","PO5","PO3","POZ",
           "PO4","PO6","PO8","CB1","O1","OZ","O2","CB2"]

#Sostituire path al file desiderato
rawDataEEGExample = loadmat('/Users/ater/Desktop/Progetto EEG/SEED IV Database/SEED_IV Database/eeg_raw_data/1/1_20160518.mat')
keysRawData = [key for key, values in rawDataEEGExample.items() if key != '__header__' and key != '__version__' and key != '__globals__']

print(keysRawData)
print(type(rawDataEEGExample['cz_eeg1']))
print(rawDataEEGExample['cz_eeg1'].shape)

rawDataEEG_df = pd.DataFrame(rawDataEEGExample['cz_eeg2'], index=channelsEEG)
print("1st Trial")
print(rawDataEEG_df)

print("///////////////////////////////////////////////\n")
#PROCESSED DATA EEG (Feature estratte dai raw signals)
print('PROCESSED DATA EEG - Subject 1 - Session 1')
columnsFreq = ['delta (1-4 Hz)', 'theta (4-8 Hz)', 'alpha (8-14 Hz)', 'beta (4-31 Hz)', 'gamma (31-50 Hz)']
#Cambiare path al file richiesto
procDataEEGExample = loadmat('/Users/ater/Desktop/Progetto EEG/SEED IV Database/SEED_IV Database/eeg_feature_smooth/1/1_20160518.mat')
keysProcDataEEG = [key for key, values in procDataEEGExample.items() if key != '__header__' and key != '__version__' and key != '__globals__']
print(keysProcDataEEG)

print("Differential Entropy - Moving Average - Trial 1 (de_movingAve1)")
print(type(procDataEEGExample['de_movingAve1']))
print(procDataEEGExample['de_movingAve1'].shape)
procDataEEG_df1 = pd.DataFrame(procDataEEGExample['de_movingAve1'][0], columns=columnsFreq)
print("channel number : 1; rows : time frame; columns : frequency_bands")
print(procDataEEG_df1)

print("Differential Entropy - Linear Dynamic System - Trial 1 (de_LDS1)")
print(type(procDataEEGExample['de_LDS1']))
print(procDataEEGExample['de_LDS1'].shape)
procDataEEG_df2 = pd.DataFrame(procDataEEGExample['de_LDS1'][0], columns=columnsFreq)
print("channel number : 1; rows : time frame; columns : frequency_bands")
print(procDataEEG_df2)

print("\nPower Spectral Density - Moving Average - Trial 1 (psd_movingAve1)")
print(type(procDataEEGExample['psd_movingAve1']))
print(procDataEEGExample['psd_movingAve1'].shape)
procDataEEG_df3 = pd.DataFrame(procDataEEGExample['psd_movingAve1'][0], columns=columnsFreq)
print("channel number : 1; rows : time frame; columns : frequency_bands")
print(procDataEEG_df3)

print("\nPower Spectral Density - Linear Dynamic System - Trial 1 (psd_LDS1)")
print(type(procDataEEGExample['psd_LDS1']))
print(procDataEEGExample['psd_LDS1'].shape)
procDataEEG_df4 = pd.DataFrame(procDataEEGExample['psd_LDS1'][0], columns=columnsFreq)
print("channel number : 1; rows : time frame; columns : frequency_bands")
print(procDataEEG_df4)

print("///////////////////////////////////////////////\n")
#RAW DATA EYES (dati raw del rilevamento degli occhi)
print('\nRAW DATA EYES - Subject 1 - Session 1')
print("Blinking Data")
#Cambiare path al file richiesto
rawDataEyesExample = loadmat('/Users/ater/Desktop/Progetto EEG/SEED IV Database/SEED_IV Database/eye_raw_data/1_20160518_blink.mat')
keysRawDataEyes = [key for key, values in rawDataEyesExample.items() if key != '__header__' and key != '__version__' and key != '__globals__']
print(keysRawDataEyes)
print(type(rawDataEyesExample['Eye_Blink']))
print(rawDataEyesExample['Eye_Blink'].shape)
print(type(rawDataEyesExample['Eye_Blink'][0]))
rawDataEyes_df = pd.DataFrame(rawDataEyesExample['Eye_Blink'], columns=['Blinking Time'])
print("Rows : movie clips; Nr. of Elements in Row : nr. of blinks in the movie clip; Data in row : time of the blink")
print(rawDataEyes_df)

print("\nStatistical Data")
rawDataEyesExample = loadmat('/Users/ater/Desktop/Progetto EEG/SEED IV Database/SEED_IV Database/eye_raw_data/1_20160518_event.mat')
keysRawDataEyes = [key for key, values in rawDataEyesExample.items() if key != '__header__' and key != '__version__' and key != '__globals__']
print(keysRawDataEyes)
print(type(rawDataEyesExample['Eye_Event']))
print(rawDataEyesExample['Eye_Event'].shape)
print(type(rawDataEyesExample['Eye_Event'][0]))
rawDataEyes_df = pd.DataFrame(rawDataEyesExample['Eye_Event'], columns=["Statistics"])
print("Rows : movie clips; Data in row : Statistical Data")
print(rawDataEyes_df)

print("\nEye Fixation Data")
rawDataEyesExample = loadmat('/Users/ater/Desktop/Progetto EEG/SEED IV Database/SEED_IV Database/eye_raw_data/1_20160518_fixation.mat')
keysRawDataEyes = [key for key, values in rawDataEyesExample.items() if key != '__header__' and key != '__version__' and key != '__globals__']
print(keysRawDataEyes)
print(type(rawDataEyesExample['Eye_Fixation']))
print(rawDataEyesExample['Eye_Fixation'].shape)
print(type(rawDataEyesExample['Eye_Fixation'][0]))
rawDataEyes_df = pd.DataFrame(rawDataEyesExample['Eye_Fixation'], columns=['Fixation Time'])
print("Rows : movie clips; Nr. of Elements in Row : nr. of fixation in the movie clip; Data in row : time of the fixation")
print(rawDataEyes_df)

print("\nPupil Data")
rawDataEyesExample = loadmat('/Users/ater/Desktop/Progetto EEG/SEED IV Database/SEED_IV Database/eye_raw_data/1_20160518_pupil.mat')
keysRawDataEyes = [key for key, values in rawDataEyesExample.items() if key != '__header__' and key != '__version__' and key != '__globals__']
print(keysRawDataEyes)
print(type(rawDataEyesExample['Eye_Pupil']))
print(rawDataEyesExample['Eye_Pupil'].shape)
print(type(rawDataEyesExample['Eye_Pupil'][0]))
rawDataEyes_df = pd.DataFrame(rawDataEyesExample['Eye_Pupil'])
print("Rows : movie clips; Nr. of Elements in Row : nr. of pupil recordings in the movie clip; Data in row : Four Features for every pupil recording")
print("Average Pupil Size [px] X, Average Pupils Size [px] Y, Dispersion X, Dispersion Y")
print(rawDataEyes_df)

print("\nSaccade Data")
rawDataEyesExample = loadmat('/Users/ater/Desktop/Progetto EEG/SEED IV Database/SEED_IV Database/eye_raw_data/1_20160518_saccade.mat')
keysRawDataEyes = [key for key, values in rawDataEyesExample.items() if key != '__header__' and key != '__version__' and key != '__globals__']
print(keysRawDataEyes)
print(type(rawDataEyesExample['Eye_Saccade']))
print(rawDataEyesExample['Eye_Saccade'].shape)
print(type(rawDataEyesExample['Eye_Saccade'][0]))
rawDataEyes_df = pd.DataFrame(rawDataEyesExample['Eye_Saccade'])
print("Rows : movie clips; Nr. of Elements in Row : nr. of saccade in the movie clip; Data in row : Feature of every saccade record")
print("Saccade Duration [ms], Amplitude[°]")
print(rawDataEyes_df)



print("///////////////////////////////////////////////\n")
#Processed DATA EYES (feature estratte dal movimento oculare)
print('PROCESSED DATA EYES - Subject 1')
rows = []
procDataEyesExample = loadmat('/Users/ater/Desktop/Progetto EEG/SEED IV Database/SEED_IV Database/eye_feature_smooth/1/1_20160518.mat')
keysProcDataEyes = [key for key, values in procDataEyesExample.items() if key != '__header__' and key != '__version__' and key != '__globals__']
print("These are one for each session")
print(keysProcDataEyes)
print(type(procDataEyesExample['eye_1']))
print(procDataEyesExample['eye_1'].shape)
procDataEyes_df = pd.DataFrame(procDataEyesExample['eye_1'])
print("Session 1")
print("Rows : type of feature; Columns : data samples")
print("Features : 1-12 : Pupil diameter (X and Y); 13-16: Dispersion (X and Y); 17-18: Fixation duration (ms); 19-22: Saccade; 23-31: Event statistics")

procDataEyes_df = procDataEyes_df.transpose()
eye_features_index = ["Mean Pupil Diameter X", "Mean Pupil Diameter Y", "STD Pupil Diameter X", "STD Pupil Diameter Y", "DE (0-0.2Hz) Pupil Diameter X",
                      "DE (0-0.2) Pupil Diameter Y", " DE (0.2-0.4Hz) Pupil Diameter X", "DE (0.2-0.4Hz) Pupil Diameter Y", "DE (0.4-0.6Hz) Pupil Diameter X",
                      "DE (0.4-0.6Hz) Pupil Diameter Y", "DE (0.6-1Hz) Pupil Diameter X", "DE (0.6-1Hz) Pupil Diameter Y", "Mean Dispersion X", "Mean Dispersion Y",
                      "STD Dispersion X", "STD Dispersion Y", "Mean Fixation Duration", "STD Fixation Duration", #"Mean Blink Duration", "STD Blink Duration",
                      "Mean Saccade Duration", "Mean Saccade amplitude", "STD Saccade Duration", "STD Saccade amplitude", "Blink Frequency", "Fixation Frequency",
                      "Maximum Fixation duration", "Total Fixation Dispersion", "Maximum Fixation Dispersion", "Saccade Frequency", "Average Saccade Duration",
                      "Average Saccade Amplitude", "Average Saccade Latency"]
procDataEyes_df.columns = eye_features_index
print(procDataEyes_df)


