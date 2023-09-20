# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.
![112](https://user-images.githubusercontent.com/75235090/191042148-df16e2a4-ce0e-4ea3-863b-dae598286e34.png)

## Neural Network Model

![111](https://user-images.githubusercontent.com/75235090/191042310-ce6d71f4-570f-40d4-ab21-7e1c23022743.png)

## DESIGN STEPS

## STEP-1:
Import tensorflow and preprocessing libraries

## STEP 2:
Download and load the dataset

## STEP 3:
Scale the dataset between it's min and max values

## STEP 4:
Using one hot encode, encode the categorical values

## STEP-5:
Split the data into train and test

## STEP-6:
Build the convolutional neural network model

## STEP-7:
Train the model with the training data

## STEP-8:
Plot the performance plot

## STEP-9:
Evaluate the model with the testing data

## STEP-10:
Fit the model and predict the single input



## PROGRAM
```python3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(x_train,y_train),(x_test,y_test)=mnist.load_data()
plt.imshow(x_train[0],cmap='gray')
x_train_scaled=x_train/255
x_test_scaled=x_test/255
print(x_train_scaled.min())
x_train_scaled.max()
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
x_train_scaled = x_train_scaled.reshape(-1,28,28,1)
x_test_scaled = x_test_scaled.reshape(-1,28,28,1)
model=Sequential([layers.Input(shape=(28,28,1)),
                  Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding='valid',activation='relu'),
                  MaxPool2D(pool_size=(2,2)),
                  Conv2D(filters=64,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu'),
                  MaxPool2D(pool_size=(2,2)),
                  layers.Flatten(),
                  Dense(8,activation='relu'),
                  Dense(10,activation='softmax')
                  ])
model.summary()
model.compile(optimizer='Adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

model.fit(x_train_scaled ,y_train_onehot, epochs=15,
          batch_size=256, 
          validation_data=(x_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(x_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))

img = image.load_img('img.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
## ACCURACY VS VAL_ACCURACY
![image](https://user-images.githubusercontent.com/75235090/190903676-0b51e313-f8b0-41a5-9ebd-6f3ae8d71ba6.png)
## TRAINING_LOSS VS VAL_LOSS 
![image](https://user-images.githubusercontent.com/75235090/190903768-9e92b42c-b3cc-49e2-8526-0d92f0a6bc24.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/75235090/190903638-cd86fa6c-9c30-433e-aa77-9a4a022c9e6d.png)

### Classification report

![image](https://user-images.githubusercontent.com/75235090/190903605-09122071-80c9-4e51-b057-432a9975d900.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/75235090/190903581-a94192f3-af1d-4ca5-ba69-e381a8542f98.png)


## RESULT
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
