from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam


IMG_SIZE = (28, 28)
NUM_CLASSES = 10


(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train, X_test = X_train / 255.0, X_test / 255.0

input_shape = (28, 28, 1)


# build CNN (3 CONV , 2 Dense, output)
model = Sequential()


# 1 layer
model.add(
    Conv2D(filters=64, kernel_size=(5, 5), activation="relu", input_shape=input_shape)
)
model.add(MaxPooling2D(pool_size=(3, 3)))

# 2 layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3 layer
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))

# flatten layer
model.add(Flatten())

# 5 layer
model.add(Dense(64, activation="relu"))

# 6 layer
model.add(Dense(32, activation="relu"))

# output layer
model.add(Dense(10, activation="softmax"))


# optimizer
optimizer = Adam(learning_rate=0.001)


model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
