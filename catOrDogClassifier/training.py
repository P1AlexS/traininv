import os
import cv2
import numpy as np
import random
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input


class Training:
    def train_model(self):
        print("Data Init")
        #Hier müssen später die Daten erst geladen werden ins projekt, wenn data leer neuste daten ziehen, wenn data nicht leer überspringen, in ingist data einen test ausführen der die bilder durchgeht ob fehler

        print("Train Model")
        directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
        categories = ['cat', 'dog']
        image_size = 100  #Sollte angepasst werden können
        data = []

        for category in categories:
            folder = os.path.join(directory, category)
            label = categories.index(category)
            for img in os.listdir(folder):
                img_path = os.path.join(folder, img)
                img_arr = cv2.imread(img_path)
                img_arr = cv2.resize(img_arr, (image_size, image_size))
                data.append([img_arr, label])

        random.shuffle(data)
        x = []
        y = []

        for features, labels in data:
            x.append(features)
            y.append(labels)

        x = np.array(x)
        y = np.array(y)

        x = x / 255.0

        model = Sequential([
            Input(shape=(image_size, image_size, 3)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(x, y, epochs=5, validation_split=0.1, batch_size=32)
        directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
        model_path = os.path.join(directory, 'model.keras')
        model.save(model_path)
