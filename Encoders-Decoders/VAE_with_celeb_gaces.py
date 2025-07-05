import tensorflow as tf
import numpy as np
from keras import layers,models,optimizers,utils,metrics,losses
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

train_data=utils.image_dataset_from_directory(
    directory="E:\dataset\img_align_celeba\img_align_celeba",
    labels=None,
    color_mode='rgb',
    image_size=(64,64),
    batch_size=128,
    seed=42,
    interpolation='bilinear',
    shuffle=True
)

def preprocess(img):
    img=tf.cast(img,'float32')/255.0

    return img

train=train_data.map(lambda x:preprocess(x))


class Sampling(layers.Layer):
    def call(self,inputs):
        z_mean,z_log_var=inputs
        batch=tf.shape(z_mean)[0]
        dim=tf.shape(z_mean)[1]
        epsilon=K.random_normal(shape=(batch,dim))

        return z_mean+tf.exp(z_log_var*0.5)*epsilon

# Encoder Part

encoder_input=layers.Input(shape=(64,64,3))

x=layers.Conv2D(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(encoder_input)
x=layers.BatchNormalization(momentum=0.9)(x)
x=layers.Dropout(0.3)(x)

x=layers.Conv2D(filters=64,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)
x=layers.BatchNormalization(momentum=0.9)(x)
x=layers.Dropout(0.3)(x)

x=layers.Conv2D(filters=128,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)
x=layers.BatchNormalization(momentum=0.9)(x)
x=layers.Dropout(0.3)(x)

x=layers.Conv2D(filters=256,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)
x=layers.BatchNormalization(momentum=0.9)(x)
x=layers.Dropout(0.3)(x)

shape_before_flattering=K.int_shape(x)[1:]

x=layers.Flatten()(x)

z_mean=layers.Dense(200,name='z_mean')(x)
z_log_var=layers.Dense(200,name='z_log_var')(x)
z=Sampling()([z_mean,z_log_var])

encoder=models.Model(encoder_input,[z_mean,z_log_var,z],name='encoder')

# Decoder-Part

decoder_input=layers.Input(shape=(200,))

x=layers.Dense(np.prod(shape_before_flattering))(decoder_input)
x=layers.Reshape(shape_before_flattering)(x)

x=layers.Conv2DTranspose(filters=256,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)

x=layers.Conv2DTranspose(filters=128,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)

x=layers.Conv2DTranspose(filters=64,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)

x=layers.Conv2DTranspose(filters=32,kernel_size=(3,3),strides=2,padding='same',activation='relu')(x)

decoder_output=layers.Conv2D(filters=3,kernel_size=(3,3),strides=1,padding='same',activation='sigmoid')(x)

decoder=models.Model(decoder_input,decoder_output)


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
            self.Kl_loss_tracker,
            self.reconstruction_loss_tracker
        ]
    
    def call(self,inputs):
        z_mean,z_log_var,z=encoder(inputs)
        reconstruction=decoder(z)

        return z_mean,z_log_var,reconstruction
    
    def train_step(self,data):
        with tf.GradientTape() as tape:
            z_mean,z_log_var,reconstruction=self(data)

            reconstruction_loss=tf.reduce_mean(500*losses.binary_crossentropy(tf.reshape(data,[-1,64*64*3]),tf.reshape(reconstruction,[-1,64*64*3])))
            kl_loss=tf.reduce_mean(tf.reduce_sum(-0.5*(1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var)),axis=1))

            total_loss=reconstruction_loss+kl_loss

            grads=tape.gradient(total_loss,self.trainable_weights)

            self.optimizer.apply_gradients(zip(grads,self.trainable_weights))

            self.total_loss_tracker.update_state(total_loss)
            self.Kl_loss_tracker.update_state(kl_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)

            return {m.name:m.result() for m in self.metrics}
        
    def test_step(self,data):
        z_mean,z_log_var,reconstruction=self(data)

        reconstruction_loss=tf.reduce_mean(500*losses.binary_crossentropy(tf.reshape(data,[-1,64*64*3]),tf.reshape(reconstruction,[-1,64*64*3])))

        kl_loss=tf.reduce_mean(tf.reduce_sum(-0.5*(1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var)),axis=1))

        total_loss=reconstruction_loss+kl_loss

          

        self.total_loss_tracker.update_state(total_loss)
        self.Kl_loss_tracker.update_state(kl_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

vae=VAE(encoder,decoder)
vae.compile(optimizer='adam')
vae.fit(train,epochs=5)



grid_width,grid_height=(10,3)

z_sample=np.random.random(size=(grid_width*grid_height,200))

reconstruction=decoder.predict(z_sample)

fig=plt.figure(figsize=(18,5))

fig.subplots_adjust(hspace=0.4,wspace=0.4)

for i in range(grid_height*grid_width):
    ax=fig.add_subplot(grid_height,grid_width,i+1)
    ax.axis('off')
    ax.set_title('Predicted_Faces')
    ax.imshow(reconstruction[i])

plt.tight_layout()
plt.show()


        












