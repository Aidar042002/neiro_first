import os
import librosa
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Конфигурация (должна совпадать с обучающим скриптом)
SAMPLE_RATE = 22050
DURATION = 4  # секунды
N_MFCC = 13
MAX_PAD_LEN = 174

# Загрузка обученной модели
model = load_model('audio_sentiment_model.h5')


def preprocess_audio(audio_path):
    """Предобработка аудиофайла для модели"""
    try:
        # Загрузка аудио
        signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)

        # Извлечение MFCC
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)

        # Паддинг или обрезка
        if mfccs.shape[1] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_PAD_LEN]

        return mfccs[np.newaxis, ..., np.newaxis]  # Добавляем batch и channel размерности
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def predict_sentiment(audio_path):
    """Предсказание тональности аудио"""
    processed_audio = preprocess_audio(audio_path)
    if processed_audio is None:
        return "Ошибка обработки аудио"

    # Предсказание
    prediction = model.predict(processed_audio)
    negative_prob = prediction[0][1]  # Вероятность класса 1 (негатив)

    # Интерпретация результата
    if negative_prob > 0.5:
        return f"Негативное аудио (вероятность: {negative_prob:.2f})"
    else:
        return f"Не негативное аудио (вероятность: {1 - negative_prob:.2f})"


# Пример использования
if __name__ == "__main__":
    audio_file = input("Введите путь к аудиофайлу: ")
    if os.path.exists(audio_file):
        result = predict_sentiment(audio_file)
        print(result)
    else:
        print("Файл не найден!")