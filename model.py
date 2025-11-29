from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

def build_model():
    model = Sequential()

    # 1st Conv Block
    model.add(Conv2D(60, (5, 5), activation='relu', input_shape=(32, 32, 1)))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # 2nd Conv Block
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Dense Layers
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(44, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
