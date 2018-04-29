import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

np.random.seed(0)

model = Sequential([
    # Dense(input_dim=2, output_dim=1),
    Dense(input_dim=2, units=1),
    Activation('sigmoid')
    ])

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

#model.fit(X, Y, nb_epoch=200, batch_size=1)
model.fit(X, Y, epochs=200, batch_size=1)

classes = model.predict_classes(X, batch_size=1)
prob = model.predict_proba(X, batch_size=1)

print('classified:')
print(Y==classes)
print()
print('output probability:')
print(prob)
