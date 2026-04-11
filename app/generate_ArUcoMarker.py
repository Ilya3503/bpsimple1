import cv2

id = 1
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
img = cv2.aruco.generateImageMarker(dictionary, id, 800)
cv2.imwrite(f"aruco.png", img)
cv2.imshow("1", img)
cv2.waitKey(0)
print(f"Сохранён aruco.png")