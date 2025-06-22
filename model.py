import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

# Конфигурация
AUDIO_DIR = 'dataset'
CSV_FILE = os.path.join(AUDIO_DIR, 'prompt.xlsx')  # предполагаем, что excel файл называется labels.xlsx
SAMPLE_RATE = 22050
DURATION = 4  # секунды
N_MFCC = 13
MAX_PAD_LEN = 174  # примерная длина MFCC для 4 секунд


# Загрузка и предобработка данных
def load_data(csv_file):
    df = pd.read_excel(csv_file)
    features = []
    labels = []

    for index, row in df.iterrows():
        audio_path = os.path.join(AUDIO_DIR, row['audio'])
        try:
            # Загрузка аудио и извлечение MFCC
            signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

            # Паддинг или обрезка до фиксированной длины
            if mfccs.shape[1] < MAX_PAD_LEN:
                pad_width = MAX_PAD_LEN - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :MAX_PAD_LEN]

            features.append(mfccs)
            labels.append(row['negative'])
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    return np.array(features), np.array(labels)


# Загрузка данных
X, y = load_data(CSV_FILE)

# Преобразование данных для модели
X = X[..., np.newaxis]  # Добавляем размерность канала (для CNN)
y = to_categorical(y)  # Преобразуем метки в one-hot encoding

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Создание модели CNN+RNN
def create_model(input_shape):
    model = models.Sequential()

    # CNN часть с padding='same' для сохранения размеров
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

    # Подготовка для RNN
    model.add(layers.Reshape((-1, 64)))

    # RNN часть
    model.add(layers.SimpleRNN(64, return_sequences=True))
    model.add(layers.SimpleRNN(64))

    # Полносвязные слои
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model


# Создаем и обучаем модель
input_shape = (N_MFCC, MAX_PAD_LEN, 1)
model = create_model(input_shape)
history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_test, y_test))

# Оценка модели
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Сохранение модели
model.save('audio_sentiment_model.h5')