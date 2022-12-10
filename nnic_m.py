#!/usr/bin/env python
# coding: utf-8

# In[63]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
# from Ipython.display import HTML
from scipy import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# In[1]:

class_names = ['Apple___Apple_scab',
'Apple___Black_rot',
'Apple___Cedar_apple_rust',
'Apple___healthy',
'Blueberry___healthy',
'Cherry_(including_sour)___healthy',
'Cherry_(including_sour)___Powdery_mildew',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize)___Common_rust_',
'Corn_(maize)___healthy',
'Corn_(maize)___Northern_Leaf_Blight',
'Grape___Black_rot',
'Grape___Esca_(Black_Measles)',
'Grape___healthy',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
'Orange___Haunglongbing_(Citrus_greening)',
'Peach___Bacterial_spot',
'Peach___healthy',
'Pepper_bell___Bacterial_spot',
'Pepper_bell___healthy',
'Potato___Early_blight',
'Potato___healthy',
'Potato___Late_blight',
'Raspberry___healthy',
'Soybean___healthy',
'Squash___Powdery_mildew',
'Strawberry___healthy',
'Strawberry___Leaf_scorch',
'Tomato___Bacterial_spot',
'Tomato___Early_blight',
'Tomato___healthy',
'Tomato___Late_blight',
'Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot',
'Tomato___Tomato_mosaic_virus',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus']


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# In[64]:


IMAGE_SIZE = 256
CHANNELS = 3

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=10,
)

training_generator = train_datagen.flow_from_directory(
    'DATA/train',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode='sparse',
    #     save_to_dir="AugmentedImages",
)

# In[65]:


for image_batch, label_batch in training_generator:
    print(image_batch.shape)
    break

# In[66]:


validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=10,
)

validation_generator = validation_datagen.flow_from_directory(
    'DATA/val',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode='sparse',
    #     save_to_dir="AugmentedImages",
)

# In[67]:


test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=10,
)

test_generator = test_datagen.flow_from_directory(
    'DATA/test',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode='sparse',
    #     save_to_dir="AugmentedImages",
)

# In[68]:


input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 38

model = models.Sequential([
    layers.InputLayer(input_shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)

# In[69]:


model.summary()

# In[70]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# In[71]:


history = model.fit(
    training_generator,
    steps_per_epoch=1921,
    batch_size=32,
    validation_data=validation_generator,
    validation_steps=274,
    verbose=1,
    epochs=50,
)

# In[ ]:
model.save("saved_models/MnM")
print("done //////////////////////////////////////////////////////////////////////")
score = model.evaluate(test_generator)

# In[ ]:


score

# In[ ]:


history

# In[ ]:


history.params

# In[ ]:


history.history.keys()

# In[ ]:


type(history.history['loss'])

# In[ ]:


len(history.history['loss'])

# In[ ]:


history.history['loss'][:5]

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# In[ ]:


val_acc

# In[ ]:


acc

# In[ ]:


EPOCHS = 50

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# In[ ]:


import numpy as np

for images_batch, labels_batch in test_generator:
    first_image = images_batch[0]
    first_label = int(labels_batch[0])

    print("first image to predict")
    plt.imshow(first_image)
    print("first image's actual label: ", class_names[first_label])

    batch_prediction = model.predict(images_batch)
    print("predicted label:", class_names[np.argmax(batch_prediction[0])])

    break

# scipy tensorflow tensorflow-gpu matplotlib
# In[ ]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i])
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return predicted_class, confidence


# In[ ]:


plt.figure(figsize=(15, 15))
for images, labels in test_generator:
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])

        predicted_class, confidence = predict(model, images[i])
        actual_class = class_names[int(labels[i])]

        plt.title(f"actual class {actual_class},\n predicted class {predicted_class},\n confidence = {confidence}")

        plt.axis("off")
    break

# In[ ]:


model.save("../doc.h5")

# In[ ]:




