from keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

IMG_SIZE = (28, 28)
NUM_CLASSES = 10


(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train, X_test = X_train / 255.0, X_test / 255.0


model = Sequential()
model.add(Flatten(input_shape=IMG_SIZE))
model.add(Dense(128, activation="relu"))
model.add(Dense(NUM_CLASSES, activation="softmax"))


optimizer = Adam(
    learning_rate=0.001,
    weight_decay=0.004,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
)


model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
