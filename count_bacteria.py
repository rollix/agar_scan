#!/usr/bin/env python3

# TODO: Remove outer contours (arcLength)
# TODO: Interactive GUI (crop, adjust threshold)

import os
import numpy as np
import cv2

input_folder = "input_images/"
output_folder = "output_images/"

area_min = 2
area_max = 2000

CANNY_THRESHOLD = 170

GREEN = (0,255,0)
RED = (0,0,255)

def get_largest_convex_hull(contours):
    # TODO: Use contourArea instead
    all_contours = np.vstack(contours[x] for x in range(len(contours)))
    convex_hull = cv2.convexHull(all_contours)

    return [convex_hull]

def get_number_of_cells(image_path, image_name):
     # Load the image file
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Detect edges
    canny_edges = cv2.Canny(image, CANNY_THRESHOLD, CANNY_THRESHOLD * 2)

    # Find contours
    contours = cv2.findContours(canny_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    contours_filtered = []

    for contour in contours:
        if area_min < cv2.contourArea(contour) < area_max:
            contours_filtered.append(contour)

    print(f"{image_name} : {len(contours_filtered)}")

    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    output = cv2.drawContours(image_color, contours_filtered, -1, GREEN, 3)
    
    cv2.imwrite(f"{output_folder}{image_name.split('.')[0]}_out.jpg", output)

if __name__=="__main__":
   image_list = os.listdir(input_folder) 
   for image_name in image_list:
       image_path = input_folder + image_name
       get_number_of_cells(image_path, image_name)
