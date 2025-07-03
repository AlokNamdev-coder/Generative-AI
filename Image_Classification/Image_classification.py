import tensorflow as tf
import numpy as np
from keras import datasets,utils,optimizers,layers,models
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()

x_train=x_train.astype('float32')/255.0

x_test=x_test.astype('float32')/255.0

y_train=utils.to_categorical(y_train,10)

y_test=utils.to_categorical(y_test,10)

input_layer=layers.Input(shape=(32,32,3))
x=layers.Flatten()(input_layer)
x=layers.Dense(units=250,activation='relu')(x)
x=layers.Dense(units=200,activation='relu')(x)
x=layers.Dense(units=150,activation='relu')(x)
x=layers.Dense(units=100,activation='relu')(x)

output_layer=layers.Dense(10,activation='softmax')(x)

model=models.Model(input_layer,output_layer)

opt=optimizers.Adam(learning_rate=0.0005)

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model.fit(x_train,y_train,epochs=15,batch_size=32,shuffle=True)

model.evaluate(x_test,y_test)

Classes = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

preds=model.predict(x_test)

preds_single = Classes[np.argmax(preds, axis=-1)]  # was wrong
act_single = Classes[np.argmax(y_test, axis=-1)]

n_to_show=10

fig=plt.figure(figsize=(15,3))

fig.subplots_adjust(hspace=0.5,wspace=0.5)

indices=np.random.choice(range(len(x_test)),n_to_show)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i + 1)
    ax.axis('off')
    ax.text(0.5, -0.35, "pred=" + str(preds_single[idx]), ha='center')
    ax.text(0.5, -0.7, "act=" + str(act_single[idx]), ha='center')
    ax.imshow(img)

plt.show()
