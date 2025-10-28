#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


# In[3]:


TRAIN_DR = "Dataset/train"
TEST_DR = "Dataset/test"


# In[4]:


def create_dataframe(directory):
    image_paths = []
    labels = []
    for label_entry in os.scandir(directory):
        if label_entry.is_dir():
            label = label_entry.name
            for image_entry in os.scandir(label_entry.path):
                if image_entry.is_file():
                    image_paths.append(image_entry.path)
                    labels.append(label)
            print(label, "completed")
    return pd.DataFrame({"Image_Path": image_paths, "Label": labels})


# In[5]:


# Create the DataFrame for training data
train = create_dataframe(TRAIN_DR)

# Display the first few rows of the DataFrame
print(train)
print(train['Image_Path'])
print(train['Label'].value_counts())


# In[6]:


# Create the DataFrame for test data
test = create_dataframe(TEST_DR)

# Display the first few rows of the DataFrame
print(test)
print(test['Image_Path'])
print(test['Label'].value_counts())


# In[7]:


def extract_features(images, target_size=(48, 48)):
    features = []
    for image in tqdm(images):
        img = load_img(image, target_size=target_size, color_mode='grayscale')
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), target_size[0], target_size[1], 1)
    return features


# In[8]:


train_features = extract_features(train['Image_Path'])


# In[9]:


test_features = extract_features(test['Image_Path'])


# In[10]:


x_train = train_features/250.0  # Scale pixle values to [0, 1]
x_test = test_features/250.0


# In[11]:


# Assuming train is your DataFrame containing image paths and labels
le = LabelEncoder()
le.fit(train['Label'])


# In[12]:


y_train = le.transform(train['Label'])
y_test = le.transform(test['Label'])


# In[13]:


y_train = to_categorical(y_train,num_classes = 7)
y_test = to_categorical(y_test,num_classes = 7)


# In[14]:


model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(7, activation='softmax'))


# In[15]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[16]:


model.fit(x= x_train,y = y_train, batch_size = 128, epochs = 100, validation_data = (x_test,y_test))


# In[17]:


model_json = model.to_json()

with open("Emotiondetector.json", 'w') as json_file:
    json_file.write(model_json)

model.save("Emotiondetector.h5")


# In[2]:


from keras.models import model_from_json


# In[3]:


json_file = open("Emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("Emotiondetector.h5")


# In[4]:


label = ['angry','disgust','fear','happy','neutral','sad','surprise']


# In[5]:


def ef(image):
    img = load_img(image, color_mode='grayscale', target_size=(48,48))  # Ensure correct size
    feature = img_to_array(img)  # Convert image to numpy array
    feature = feature.reshape(1, 48, 48, 1)  # Reshape to match model input shape
    feature = feature / 255.0  # Normalize pixel values
    return feature


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


image = 'Dataset/train/sad/Training_99823693.jpg'
print("original image is of sad")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')


# In[9]:


image = 'Dataset/train/fear/Training_99984859.jpg'
print("original image is of fear")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')


# In[10]:


image = 'Dataset/train/disgust/Training_8937293.jpg'
print("original image is of disgust")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')


# In[11]:


image = 'Dataset/train/happy/Training_99973350.jpg'
print("original image is of happy")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')


# In[12]:


image = 'Dataset/train/surprise/Training_99924420.jpg'
print("original image is of surprise")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')


# In[16]:


import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the pre-trained emotion detection model
model_path = 'Emotiondetector.h5'
classifier = load_model(model_path)

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start webcam (use 0 for default camera, 1 for external camera)
cap = cv2.VideoCapture(0)

# Check if webcam opens correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Error: Failed to capture image.")
        break  # Exit loop if no frame is captured

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray[y:y+h, x:x+w]

        # Resize the face to 48x48 pixels
        face = cv2.resize(face, (48, 48))
        
        # Normalize and expand dimensions for model prediction
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)  # Shape should be (1, 48, 48, 1) for the model

        # Predict emotion using the classifier
        prediction = classifier.predict(face)
        label = emotion_labels[np.argmax(prediction)]

        # Draw a rectangle around the face and add the emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with emotion label
    cv2.imshow('Emotion Detector', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Ensure proper indentation here

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()


# In[ ]:




