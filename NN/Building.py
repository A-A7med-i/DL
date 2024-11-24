from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

IMG_SIZE = (28, 28)
NUM_CLASSES = 10


# Read data
# mnist  it's a dataset of handwritten digits
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Normalization the data to become from 0 to 1
X_train, X_test = X_train / 255.0, X_test / 255.0


# build NN (input , one hidden layer, output)
model = Sequential()
model.add(Flatten(input_shape=IMG_SIZE))
model.add(Dense(128, activation="relu"))
model.add(Dense(NUM_CLASSES, activation="softmax"))


# Optimizer is SGD if you don't understand it go to for my repo about ML-MATH
optimizer = SGD(
    learning_rate=0.01, momentum=0.9, decay=0.01, nesterov=True, clipnorm=1.0
)


model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
