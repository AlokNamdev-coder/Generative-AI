import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets,optimizers,layers,models,metrics,losses
from tensorflow.keras import backend as K

(x_train,y_train),(x_test,y_test)=datasets.fashion_mnist.load_data()


def preprocess(img):
    img=img.astype('float32')/255.0
    img=np.pad(img,((0,0),(2,2),(2,2)),)
    img=np.expand_dims(img,axis=-1)

    return img

x_train=preprocess(x_train)
x_test=preprocess(x_test)

# Sampling Layer

class Sampling(layers.Layer):
    def call(self,inputs):
        z_mean,z_log_var=inputs
        batch=tf.shape(z_mean)[0]
        dim=tf.shape(z_mean)[1]
        epsilon=K.random_normal(shape=(batch,dim))

        return z_mean+tf.exp(z_log_var*0.5)*epsilon

# Encoder Part

encoder_input=layers.Input(shape=(32,32,1))

x=layers.Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(encoder_input)
x=layers.Conv2D(filters=64,kernel_size=(32,32),strides=2,padding='same',activation='relu')(x)
x=layers.Conv2D(filters=128,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)
x=layers.Conv2D(filters=256,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)

shape_before_flattening=K.int_shape(x)[1:]

x=layers.Flatten()(x)

z_mean=layers.Dense(64,name='z_mean')(x)
z_log_var=layers.Dense(64,name='z_log_var')(x)

z=Sampling()([z_mean,z_log_var])

encoder=models.Model(encoder_input,[z_mean,z_log_var,z],name='encoder')

# Decoder Part

decoder_input=layers.Input(shape=(64,))

x=layers.Dense(np.prod(shape_before_flattening))(decoder_input)

x=layers.Reshape(shape_before_flattening)(x)

x=layers.Conv2DTranspose(filters=256,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)
x=layers.Conv2DTranspose(filters=128,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)
x=layers.Conv2DTranspose(filters=64,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)
x=layers.Conv2DTranspose(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)

decoder_output=layers.Conv2D(filters=1,strides=1,kernel_size=(3,3),padding='same',activation='sigmoid')(x)

decoder=models.Model(decoder_input,decoder_output,name='decoder')

# VAE

class VAE(models.Model):
    def __init__(self,decoder,encoder ):
        super().__init__()
        self.decoder=decoder
        self.encoder=encoder
        self.total_loss_tracker=metrics.Mean(name='total_loss_tracker')
        self.reconstruction_loss_tracker=metrics.Mean(name='reconstruction_loss_tracker')
        self.Kl_loss_tracker=metrics.Mean(name='Kl_loss_tracker')


    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.Kl_loss_tracker
        ]
        
    def call(self,inputs):
        z_mean,z_log_var,z=encoder(inputs)
        reconstruction=decoder(z)

        return z_mean,z_log_var,reconstruction
        
    def train_step(self,data):
        with tf.GradientTape() as tape:
            z_mean,z_log_var,reconstruction=self(data)

            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction), axis=(1, 2, 3))
)


            kl_loss=tf.reduce_mean(tf.reduce_sum(-0.5*(1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var)),axis=1))

            total_loss=reconstruction_loss+kl_loss

            grads=tape.gradient(total_loss,self.trainable_weights)

            self.optimizer.apply_gradients(zip(grads,self.trainable_weights))

            self.total_loss_tracker.update_state(total_loss)
            self.Kl_loss_tracker.update_state(kl_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)

            return {m.name:m.result()  for m in self.metrics}
            
    def test_step(self,data):
        z_mean,z_log_var,reconstruction=self(data)

        reconstruction_loss=tf.reduce_mean(losses.mean_squared_error(data,reconstruction),axis=(1,2,3))

        kl_loss=tf.reduce_mean(tf.reduce_sum(-0.5*(1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var)),axis=1))

        total_loss=reconstruction_loss+kl_loss

          

        self.total_loss_tracker.update_state(total_loss)
        self.Kl_loss_tracker.update_state(kl_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)


vae=VAE(encoder,decoder)

vae.compile(optimizer='adam')

vae.fit(x_train,batch_size=100,epochs=5,shuffle=True)

pred_data=x_test[:5000]

_,_,pred=vae.predict(pred_data)


n_to_sum=10

fig=plt.figure(figsize=(20,4))

for i in range(n_to_sum):

    ax=fig.add_subplot(2,n_to_sum,i+1)

    ax.imshow(pred_data[i].reshape(32,32),cmap='gray')
    ax.set_title("Original")
    ax.axis('off')

    ax=fig.add_subplot(2,n_to_sum,i+11)

    ax.imshow(pred[i].reshape(32,32),cmap='gray')
    ax.set_title("Reconstructed")
    ax.axis('off')

plt.tight_layout()
plt.show()







