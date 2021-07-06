# 1: Imports from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from kymatio.keras import Scattering2D
# Above, we import the Scattering2D class from the kymatio.keras package.
# 2: Model definitioninputs = Input(shape=(28, 28))
x = Scattering2D(J=3, L=8)(inputs)
x = Flatten()(x)
x_out = Dense(10, activation='softmax')(x)
model_kymatio = Model(inputs, x_out)
print(model_kymatio.summary())
# 3: Compile and trainmodel_kymatio.compile(optimizer='adam'), loss='sparse_categorical_crossentropy', metrics=['accuracy'}
# We train the model_kymatio using model_kymatio.fit on a subset of the MNIST data
model_kymatio.fit(x_train[:10000], y_train[:10000], epochs=15, batch_size=64, validation_split=0.2)
# Finally, we evaluate the mode_kymatio on the held-out text data.model_kymatio.evaluate(x_tet, y_test)Model: "model"
