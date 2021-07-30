# its a just a copy of python function which is use in this model.
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

files = filedialog.askopenfilenames()
print("Path of the Image selected by the user : ", files[0])
print("[WAIT]... Working on prediction of age and gender on your input file...")
import numpy as np
from tensorflow.keras.models import load_model # load_model is to call the trained model, that is model.h5 that we have created.
import cv2
modelPath = "./model.h5"
model = load_model(modelPath)
outputPath = r"./" 
imagePath = files[0]
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
pic = cv2.imread(imagePath)
gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.3,5)
age = [] # Creating empty list, age.
gender = [] # Creating empty list gender.
for (x,y,w,h) in faces:
  img = gray[y-50:y+40+h,x-10:x+10+w] 
  img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) # Converting our image from GRAY to RGB, as we have extracted the features.
  img = cv2.resize(img,(200,200)) # Resizing the original image to 200 X 200.
  predict = model.predict(np.array(img).reshape(-1,200,200,3)) # Predicting from our imported model, after changing the image into an array and reshaping it.
  age.append(predict[0]) # Appending predicted age from predict variable in age list.
  gender.append(np.argmax(predict[1])) # Appending the maximum, that is the value with more weight in gender list from the predicted values.
  gend = np.argmax(predict[1]) # Keeping the gender value in seperate variable to change the categorical value into string.
  if gend == 0:
    gend = 'Man'
    col = (255,0,0)
  else:
    gend = 'Female'
    col = (203,12,255)
  cv2.rectangle(pic,(x,y),(x+w,y+h),(0,225,0),4)
  cv2.putText(pic,"Age : "+str(int(predict[0]))+" / "+str(gend),(x,y),cv2.FONT_HERSHEY_SIMPLEX,w*0.005,col,4) # Writing the predicted values on the images.
cv2.imwrite("predicted.jpg",pic) # Saving the image to our output path by the name we provided below.
print("Done. Check the predicted file at the place where you have stored this python file along with the model.")