import numpy as np
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Lambda, Input
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import backend as K
import cvlib as cb
import cv2 as cv

# some params

INPUT_SHAPE = (125,100)

class engine:
    def __init__(self, image_shape):
        if len(image_shape) !=2:
            print("Invalid image shape")
            return
        self.input_shape = (*image_shape, 1)

        # construct model
        inputleft = Input(self.input_shape)
        inputright = Input(self.input_shape)

        model = Sequential()
        model.add(Conv2D(64,(10,10),activation='relu', input_shape = self.input_shape, kernel_regularizer = l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(Conv2D(128,(7,7),activation='relu', input_shape = self.input_shape, kernel_regularizer = l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(Conv2D(128,(3,3),activation='relu', input_shape = self.input_shape, kernel_regularizer = l2(2e-4)))
        model.add(MaxPooling2D())
        model.add(Conv2D(256,(3,3),activation='relu', input_shape = self.input_shape, kernel_regularizer = l2(2e-4)))
        model.add(Flatten())
        model.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3)))

        encodeleft = model(inputleft)
        encoderight = model(inputright)

        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1])) #Vector subtraction
        L1_distance = L1_layer([encodeleft, encoderight])
        prediction = Dense(1, activation='sigmoid')(L1_distance)

        self.model = Model(inputs=[inputleft,inputright], outputs = prediction)
        self.model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = 0.00006))

    def ssd(self, pic, padding):
        h = pic.shape[0]
        w = pic.shape[1]
        f, _ = cb.detect_face(pic)
        if len(f) < 1:
            return pic, False
        f = f[0]
        pic = pic[max(0, f[1]-padding):min(h, f[3]+padding), max(0, f[0]-padding):min(w, f[2]+padding)]
        pic = cv.resize(pic, (100,125))
        return cv.cvtColor(pic, cv.COLOR_BGR2GRAY), True
    
    def predict(self, a, b):
        if not type(a) == np.ndarray or not type(b) == np.ndarray:
            print("Input tensors need to be of type numpy.array")
            return
        if not a.dtype == 'uint8' or not a.dtype == 'uint8':
            print("Input tensor dtype needs to be uint8")
            return
        a, a_result = self.ssd(a, 15)
        b, b_result = self.ssd(b, 15)
        if not a_result or not b_result:
            print("No face detected in image {}".format("a and b" if (not a_result and not b_result) else ("a" if not a_result else "b")))
            return
        try:
            a, b = a.reshape(1, *self.input_shape), b.reshape(1, *self.input_shape)
        except Exception as e:
            print("Input tensors could not be reshaped. Exception:\n", e)
        return self.model.predict([a.astype('uint32'), b.astype('uint32')])

    #TODO: Make shit uint32
