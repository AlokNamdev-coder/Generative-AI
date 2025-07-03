import numpy as np
import tensorflow as tf
from keras import datasets,layers,models,optimizers
from tensorflow.keras import backend as K


(x_train,y_train),(x_test,y_test)=datasets.fashion_mnist.load_data()

def preprocess(img):
    img=img.astype('float32')/255.0
    img=np.pad(img,((0,0),(2,2),(2,2)),constant_values=0.0)
    img=np.expand_dims(img,axis=-1)

    return img

x_train=preprocess(x_train)

x_test=preprocess(x_test)

# Encoder-Part

encoder_input=layers.Input(shape=(32,32,1))

x=layers.Conv2D(filters=32,kernel_size=(3,3),strides=2,activation='relu',padding='same')(encoder_input)

x=layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',strides=2,activation='relu')(x)

x=layers.Conv2D(filters=128,kernel_size=(3,3),padding='same',strides=2,activation='relu')(x)

shape_before_flattering=K.int_shape(x)[1:]

x=layers.Flatten()(x)

encoder_output=layers.Dense(2,)(x)

encoder_model=models.Model(encoder_input,encoder_output)

# Decoder Part

decoder_input=layers.Input(shape=(2,))

x=layers.Dense(np.prod(shape_before_flattering))(decoder_input)

x=layers.Reshape(shape_before_flattering)(x)

x=layers.Conv2DTranspose(128,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)

x=layers.Conv2DTranspose(64,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)

x=layers.Conv2DTranspose(32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)

decoder_output=layers.Conv2D(filters=1,kernel_size=(3,3),strides=1,padding='same',activation='sigmoid')(x)

decoder=models.Model(decoder_input,decoder_output)

autoencoder_output=decoder(encoder_output)

autoencoder=models.Model(encoder_input,autoencoder_output)

opt=optimizers.Adam(learning_rate=0.005)

autoencoder.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

autoencoder.fit(x_train,x_train,epochs=5,batch_size=32,shuffle=True,validation_data=(x_test,x_test))

prediction_data=x_test[:5000]

autoencoder_prediction=autoencoder.predict(prediction_data)

import matplotlib.pyplot as plt

fig=plt.figure(figsize=(20,4))



for i in range(10):
    # Original
    ax = fig.add_subplot(2, 10, i + 1)
    plt.imshow(prediction_data[i].reshape(32, 32), cmap='gray')
    plt.title("Original")
    plt.axis("off")

    # Reconstructed
    ax = fig.add_subplot(2, 10, i + 11)
    plt.imshow(autoencoder_prediction[i].reshape(32, 32), cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()







