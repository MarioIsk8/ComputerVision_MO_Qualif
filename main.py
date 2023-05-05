import math
import os
import cv2
import numpy as np
from scipy.cluster.vq import *
import matplotlib.pyplot as plt

def clearScreen():
  os.system('cls' if os.name == 'nt' else 'clear')

def ShapeDetection():
  image = cv2.imread('./ShapeFolder/gambar.png')
  new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  new_image = cv2.resize(new_image, (700, 500))

  _, threshold = cv2.threshold(new_image, 127, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  first = False
  for contour in contours:
    if first == False:
      first = True
      continue

    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    M = cv2.moments(contour)
    x = int(M['m10'] / M['m00'])
    y = int(M['m01'] / M['m00'])
    
    if(len(approx)==3):
      cv2.putText(new_image, "Triangle", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0))
    elif(len(approx)==4):
      cv2.putText(new_image, "Rectangle", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0))
    elif(len(approx)==5):
      cv2.putText(new_image, "Pentagon", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0))
    elif(len(approx)==6):
      cv2.putText(new_image, "Hexagon", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0))
    elif(len(approx)==7):
      cv2.putText(new_image, "Heptagon", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0))
    elif(len(approx)==8):
      cv2.putText(new_image, "Octagon", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0))
    elif(len(approx)==9):
      cv2.putText(new_image, "Nonagon", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0))
    elif(len(approx)==10):
      cv2.putText(new_image, "Decagon", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0))  
    else:
      area = cv2.contourArea(contour)
      perimeter = cv2.arcLength(contour, True)
      circularity = 4 * math.pi * area / (perimeter * perimeter)
      if circularity > 0.8:
        cv2.putText(new_image, 'Circle', (x, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 0, 0))

  cv2.imshow('Shape Detection', new_image)
  cv2.waitKey()

def edgeDetectionResult(nrow = None, ncol = None, res_stack = None):

  plt.figure(figsize=(12, 12))

  for i, (label, image) in enumerate(res_stack):
    plt.subplot(nrow, ncol, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(label)
    plt.axis('off')

  plt.show()

def EdgeDetection():
  image = cv2.imread('./Edge Detection/IU.jpg')
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  laplace8u = cv2.Laplacian(gray_image, cv2.CV_8U)
  laplace16s = cv2.Laplacian(gray_image, cv2.CV_16S)
  laplace32f = cv2.Laplacian(gray_image, cv2.CV_32F)
  laplace64f = cv2.Laplacian(gray_image, cv2.CV_64F)

  laplace_labels = ['8U', '16S', '32F', '64F']
  laplace_images = [laplace8u, laplace16s, laplace32f, laplace64f]

  edgeDetectionResult(2, 2, zip(laplace_labels, laplace_images))


def PatternDetection():
  classifier = cv2.CascadeClassifier('./Pattern Recognition/haarcascade_frontalface_default.xml')
  train_path = './Pattern Recognition/train'

  dir = os.listdir(train_path)

  face_list = []
  class_list = []

  for idx, train_dir in enumerate(dir):
    for image_path in os.listdir(f'{train_path}/{train_dir}'):
      path = f'{train_path}/{train_dir}/{image_path}'
      gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
      faces = classifier.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)

      if len(faces) < 1:
        continue
      else:
        for face_rect in faces:
          x, y, w, h = face_rect
          face_image = gray[y:y+w, x:x+h]
          face_list.append(face_image)
          class_list.append(idx)

  face_recognizer = cv2.face.LBPHFaceRecognizer_create()
  face_recognizer.train(face_list, np.array(class_list))

  test_path = './Pattern Recognition/test'

  for path in os.listdir(test_path):
    print(path)
    fullpath = f'{test_path}/{path}'
    image = cv2.imread(fullpath)
    igray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(igray, scaleFactor = 1.2, minNeighbors = 5)

    if len(faces) < 1:
      continue
    
    for face_rect in faces:
      x, y, w, h = face_rect
      face_image = igray[y:y+w, x:x+h]
      res, conf = face_recognizer.predict(face_image)

      conf = math.floor(conf * 100) / 100
      cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

      image_text = f'{dir[res]} : {str(conf)}%'

      cv2.putText(image, image_text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 1)
      cv2.imshow('Result', image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

def FaceRecog():
  face_cascade = cv2.CascadeClassifier('./Face Recognition/haarcascade_frontalface_default.xml')
  cap = cv2.VideoCapture(0)

  while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
      detected = frame[y:y + h, x: x + w]
      detected = cv2.GaussianBlur(detected, (23, 23), 30)
      frame[y:y + h, x: x + w] = detected

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cv2.imwrite('face_detection_result.jpg', frame)
  cap.release()
  cv2.destroyAllWindows()

def mainMenu():
  print('CompVis')
  print('1. Shape Detection') 
  print('2. Edge Detection') 
  print('3. Pattern Recognition') 
  print('4. Face Detection') 
  print('5. Exit')  

while (True):
  clearScreen()
  mainMenu()
  menu = input('>> ')
  if (menu == '1'): 
    ShapeDetection()
  elif (menu == '2'): 
    EdgeDetection()
  elif (menu == '3'): 
    PatternDetection()
  elif (menu == '4'): 
    FaceRecog()
  elif (menu == '5'):
    clearScreen()
    input('Exit')
    exit(0)