import cv2

img_file='car_img.jpg'
classifier_file='cars.xml'

img= cv2.imread('more_cars.jpg')

black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect cars
cars = car_tracker.detectMultiScale(black_n_white)
# Draw squares on Cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)

#print(cars)

cv2.imshow('car detector',img)

cv2.waitKey()



print("car and pedestrian tracking code completed ")

