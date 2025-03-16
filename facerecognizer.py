import cv2
import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_list = []
face_class_list = []

path = 'Dataset/'

def train():
    for idx, class_path in enumerate(os.listdir(path)):
        for image_path in os.listdir(f'{path}{class_path}'):
            full_path = f'{path}{class_path}/{image_path}'
            gray = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            faces = classifier.detectMultiScale(bilateral, scaleFactor = 1.2, minNeighbors = 5)
            if len(faces) < 1:
                continue
            else:
                for face in faces:
                    x, y, w, h = face
                    face_image = bilateral[y:y+h, x:x+w]
                    face_list.append(face_image)
                    face_class_list.append(idx)
    
    X_train, X_test, y_train, y_test = train_test_split(face_list, face_class_list, train_size = 0.75, random_state = 42)

    face_recognizer.train(X_train, np.array(y_train))

    y_pred = []

    for i in  X_test:
        result, _ = face_recognizer.predict(i)
        y_pred.append(result)

    accuracy = accuracy_score(y_test, y_pred)

    print("Training and Testing Finished")
    print(f"Average Accuracy: {accuracy * 100}%")

    face_recognizer.write('trained_model.yml')

def recognize():
    if os.path.exists('trained_model.yml'):
        face_recognizer.read('trained_model.yml')

        absolute_path = input("Input absolute path for image to predict >> ")

        if os.path.exists(absolute_path):
            img = cv2.imread(absolute_path)
            gray = cv2.imread(absolute_path, cv2.IMREAD_GRAYSCALE)
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            faces = classifier.detectMultiScale(bilateral, scaleFactor = 1.2, minNeighbors = 5)
            if len(faces) < 1:
                    print("Face not found.")
            else:
                for face in faces:
                    x, y, w, h = face
                    face_image = bilateral[y:y+h, x:x+w]
                    result, confidence = face_recognizer.predict(face_image)
                    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0))
                    confidence = 100 - float(confidence)
                    confidence = math.floor(confidence * 100) / 100
                    text = f'{os.listdir(path)[result]} : {confidence}%'
                    cv2.putText(img, text, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('result', img)
                    cv2.waitKey()
                    cv2.destroyAllWindows()
        else:
            print("File not found. Please input a valid absolute path.")
    else:
        print("Trained model not found. Please train the model first.")           

def menu():
    print ("Football Player Recognition")
    print("1. Train and Test Model")
    print("2. Predict")
    print("3. Exit")

menu()
option = input(">> ")

while option != '3':
    if option == '1':
      print("Training and Testing") 
      train()
      input("Press enter to continue...")
    elif option == '2':
        recognize()
        input("Press enter to continue...")
    else:
        print("Invalid option")
        input("Press enter to continue...")
    
    print()
    menu()
    option = input(">> ")