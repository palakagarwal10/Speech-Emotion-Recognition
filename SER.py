import sys
import pandas as pd
import numpy as np
import os
import soundfile as sf
import joblib

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cycler



from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from IPython.display import Audio

import keras
from keras.callbacks import ReduceLROnPlateau

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LSTM
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category = DeprecationWarning)

Tess = "D:\Projects\Speech Emotion Reconition\TESS Toronto emotional speech set data\TESS Toronto emotional speech set data"

tess_directory_list = os.listdir(Tess)

file_emotion = []
file_path = []

for dir in tess_directory_list:
    directories = os.listdir(os.path.join(Tess, dir))
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[2]
        
        if part == 'ps':
            file_emotion.append('surprise')
        else:
             file_emotion.append(part)
        file_path.append(os.path.join(Tess, dir, file))

emotion_df = pd.DataFrame(file_emotion , columns = ['Emotions'])

path_df = pd.DataFrame (file_path, columns = ['Path'])
Tess_df = pd.concat([emotion_df, path_df], axis = 1)
Tess_df.head()

data_path = pd.concat([Tess_df], axis = 0) 
data_path.to_csv("data_path.csv", index = False)
data_path.head()

plt.title('Count of Emotions', size = 16)
sns.countplot(x = 'Emotions', data = data_path)
plt.ylabel('Count', size = 12)
plt.xlabel('Emotions', size = 12)
sns.despine (top = True, right= True, left = False, bottom = False)
plt.show()

def create_waveplot (data, sr, e):
    plt.figure(figsize = (10,3))
    plt.title('Waveplot for audio with {} emotion'.format(e), size = 15)
    librosa.display.waveshow(data, sr=sr, color='blue') 
    plt.show()

# stft(short term fourier transform) represents signals in the time-frequency domain by computing discrete fourier transformation over short overlapping windows.
#stft function converts the data into short term fourier transform 
def create_spectrogram(data, sr, e):
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize = (12,3))
    plt.title('Spectrogram for audio with {} emotion'.format(e), size = 15)
    librosa.display.specshow(Xdb, sr=sr, x_axis = 'time', y_axis = 'hz')
    
    plt.colorbar()

emotion = 'fear'
path = np.array(data_path.Path[data_path.Emotions==emotion])[1]
data, sampling_rate = librosa.load(path) #sampling rate is the number of samples taken per second)
create_waveplot(data, sampling_rate, emotion)
create_spectrogram(data, sampling_rate, emotion)
Audio(path)

emotion='angry'
path = np.array(data_path.Path[data_path.Emotions==emotion])[1]
data, sampling_rate = librosa.load(path)
create_waveplot(data, sampling_rate, emotion)
create_spectrogram(data, sampling_rate, emotion)
Audio(path)

emotion = 'sad'
path = np.array(data_path.Path[data_path.Emotions == emotion])[1]
data, sampling_rate = librosa.load(path)
create_waveplot(data, sampling_rate, emotion)
create_spectrogram (data, sampling_rate, emotion)
Audio(path)

emotion = 'happy'
path= np.array(data_path.Path[data_path.Emotions==emotion])[1]
data, sampling_rate = librosa.load(path)
create_waveplot(data, sampling_rate, emotion)
create_spectrogram(data, sampling_rate, emotion)
Audio(path)


# Data Augmentation

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data+noise_amp*np.random.normal(size = data.shape[0])
    return data

def stretch (data):
    return librosa.effects.time_stretch(data, rate = 0.8)

def shift (data):
    shift_range = int(np.random.uniform(low= -5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch (data, sampling_rate, pitch_factor = 0.7):
    n_steps = int(pitch_factor*12)
    return librosa.effects.pitch_shift(data, sr = sampling_rate, n_steps = n_steps)


path = np.array(data_path.Path)[1]
data, sample_rate = librosa.load(path)


# 1. Simple Audio
plt.figure(figsize = (14,4))
librosa.display.waveshow(data, sr=sample_rate, color='blue') 
Audio(path)


# 2. Noise Injection
x = noise (data)
plt.figure(figsize=(14, 4))
librosa.display.waveshow(y = x, sr = sample_rate, color = 'blue')
Audio(x , rate = sample_rate)


# 3. Stretching
x= stretch(data)
plt.figure(figsize = (14,4))
librosa.display.waveshow(y=x, sr = sample_rate, color = 'blue')
Audio(x, rate = sample_rate)


# 4. Shifting 
x = shift(data)
plt.figure(figsize = (14,4))
librosa.display.waveshow(y = x, sr = sample_rate, color = 'blue')
Audio(x , rate= sample_rate)


# 5. pitching 
x = pitch(data, sampling_rate)
plt.figure(figsize = (14,4))
librosa.display.waveshow(y =x, sr= sample_rate, color = 'blue')
Audio(x, rate = sample_rate)


# Feature selection
def extract_features(data):
    #zero crossing rate: The rate of sign-changes of signal during the duration of a particular frame
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y = data).T, axis = 0)
    result = np.hstack((result,zcr)) #stacking horizontally
    
    # #chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.mfcc(y = data, sr = sample_rate).T, axis = 0)
    result = np.hstack((result, chroma_stft))
    
    #MFCCs Mel Frequency Ceptral Coefficients from a Ceptral representation where the frequency bamds are not linear but distributed according to mel-scale
    
    mfcc = np.mean(librosa.feature.mfcc(y = data, sr = sample_rate).T, axis = 0)
    result = np.hstack((result,mfcc))
    
    #Root mean Square Value
    rms= np.mean(librosa.feature.rms(y = data).T, axis = 0)
    result = np.hstack((result , rms))
    
    # MelSpectrogram 
    mel = np.mean(librosa.feature.melspectrogram(y = data, sr = sample_rate).T, axis = 0)
    result = np.hstack((result, mel))
    
    return result

def get_features(path):
    #duration and offset are used to take care of the no audio in the start and the end of each of the audio files seen above.
    data, sample_rate = librosa.load(path, duration = 2.5, offset = 0.6)
    
    #without augmentation
    res1 = extract_features(data)
    result = np.array(res1)
    
    #data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))
    
    #data with stretching and pitching
    new_data = stretch (data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))
    
    return result

X, Y = [], []
for path, emotion in zip(data_path.Path, data_path.Emotions):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
        Y.append(emotion)


len (X), len (Y), data_path.Path.shape
Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('features.csv', index = False)
Features.head()


# Preparation
X = Features.iloc[:,:-1].values
Y = Features['labels'].values

encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

#data compatible to model
x_train = np.expand_dims(x_train, axis = 2)
x_test = np.expand_dims(x_test, axis = 2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# Modelling
model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(170,1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# history = model.fit(x_train, y_train, validation_split=0.2, epochs=50, batch_size=128)


# early_stopping = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(x_train, y_train, validation_split=0.2, epochs=50, batch_size=64) #callbacks=[early_stopping])

print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")

epochs = [i for i in range(50)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(epochs , train_loss , label = 'Training Loss')
ax[0].plot(epochs , test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()

# predicting on test data.
pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)

y_test = encoder.inverse_transform(y_test)

df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df['Predicted Labels'] = y_pred.flatten()
df['Actual Labels'] = y_test.flatten()

df.head(10)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (12, 10))
cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()

print(classification_report(y_test, y_pred))