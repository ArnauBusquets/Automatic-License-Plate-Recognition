import numpy as np
import cv2
import argparse
import imutils


def load_image(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Check if image is loaded and print image shape
    print(image.shape)
	
    # # Resize image
    image = cv2.resize(image, (2016, 1134))

    # # Show image
    # cv2.imshow("Image", image)      # En aquest cas es veu molt gran per la alta resolució de la imatge	
    # cv2.waitKey(0)

    # # Save the image to disk
    # cv2.imwrite("novaimg.jpg", image)

    return image


image = load_image("C:/Users/arnau/Desktop/4t Eng/1r Semestre/PSIV 2/Reptes/Matricules/Lateral/PXL_20210921_095142454.jpg")


# Convert the image to grayscale, blur it, and threshold it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1] # 150 es el threshold que he posat jo, es pot canviar


cv2.imshow("Image", thresh)      # En aquest cas es veu molt gran per la alta resolució de la imatge	
cv2.waitKey(0)



# class ShapeDetector:
# 	def __init__(self):
# 		pass
# 	def detect(self, c):
# 		# initialize the shape name and approximate the contour
# 		shape = "unidentified"
# 		peri = cv2.arcLength(c, True)
# 		approx = cv2.approxPolyDP(c, 0.04 * peri, True)
		
#         # if the shape is a triangle, it will have 3 vertices
# 		if len(approx) == 3:
# 			shape = "triangle"
# 		# if the shape has 4 vertices, it is either a square or
# 		# a rectangle
# 		elif len(approx) == 4:
# 			# compute the bounding box of the contour and use the
# 			# bounding box to compute the aspect ratio
# 			(x, y, w, h) = cv2.boundingRect(approx)
# 			ar = w / float(h)
# 			# a square will have an aspect ratio that is approximately
# 			# equal to one, otherwise, the shape is a rectangle
# 			shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
# 		# if the shape is a pentagon, it will have 5 vertices
# 		elif len(approx) == 5:
# 			shape = "pentagon"
# 		# otherwise, we assume the shape is a circle
# 		else:
# 			shape = "circle"
# 		# return the name of the shape
# 		return shape


# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# sd = ShapeDetector()


# Find contours in the image 
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
 
# Loop through the contours and check for rectangles 
for contour in contours: 
    # Get the bounding box of the contour 
    x, y, w, h = cv2.boundingRect(contour) 
     
    # Check if the contour is a rectangle 
    if abs(w - h) <= 3: 
        # Draw a rectangle around the contour 
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) 
 
# Display the image with the detected rectangles 
cv2.imshow('Rectangles', image) 
cv2.waitKey(0) 
cv2.destroyAllWindows()