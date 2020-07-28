#!/usr/bin/env python3

# TODO: Interactive GUI (crop, adjust threshold, area bounds)
# TODO: Red color filter?
# TODO: Use contour hierarchy instead of dividing by 2
# TODO: Relate arcLength and contourArea to find actual circular shapes
# TODO: Identify duplictes with cv2.moments
# TODO: Color invariance

import os
import numpy as np
import cv2

input_folder = "input_images/"
output_folder = "output_images/"
DIRECTORIES = [input_folder, output_folder]

GREEN = (0,255,0)
RED = (0,0,255)

# Parameters to optimize
area_min = 2
area_max = 2000
perimeter_max = 500
dish_interior = 0.95
CANNY_THRESHOLD = 170


def downsample(image, k_size, n_iter):
    for i in range(n_iter):
        image = cv2.medianBlur(image, k_size)
    return image

def get_dish(image):
    # Find largest circle with a Hough transform
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    image_blur = downsample(image, 5, 1)
    circles = cv2.HoughCircles(image_blur, cv2.HOUGH_GRADIENT, 1, image.shape[0]/100)
    if circles is None:
        return cv2.bitwise_not(mask), None 
    circles = np.uint16(np.around(circles))
   
    c = circles[0,:][0]

    # Get rectangle inscribing the dish
    m = (c[0], c[1])
    r = int(c[2]*dish_interior)
    top_left = (c[0] - r, c[1] - r)
    bottom_right = (c[0] + r, c[1] + r)
    cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    image_dish = cv2.bitwise_and(image, image, mask=mask)
    
    return image_dish

def get_dish_interior(image):
    image_dish = get_dish(image)

    # Find largest circle with a Hough transform
    mask = np.zeros((image_dish.shape[0], image_dish.shape[1]), dtype=np.uint8)
    image_blur = downsample(image_dish, 5, 1)
    circles = cv2.HoughCircles(image_blur, cv2.HOUGH_GRADIENT, 1, image_dish.shape[0]/100)
    if circles is None:
        return cv2.bitwise_not(mask), None 
    circles = np.uint16(np.around(circles))
   
    c = circles[0,:][0]
    cv2.circle(mask, (c[0], c[1]), c[2], 255, -1)
    
    image_interior = cv2.bitwise_and(image_dish, image_dish, mask=mask)
    return image_interior, c

def area_in_range(contour):
    return area_min < cv2.contourArea(contour) < area_max

def perimeter_in_range(contour):
    return cv2.arcLength(contour, False) < perimeter_max

def get_contours(image):
    # Detect edges
    canny_edges = cv2.Canny(image, CANNY_THRESHOLD, CANNY_THRESHOLD * 2)
    # Find contours
    contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_filtered = []
    for n, contour in enumerate(contours):
        if area_in_range(contour) and perimeter_in_range(contour):
            contours_filtered.append(contour)
    return contours_filtered


def find_cells(image_path, image_name):
     # Load the image file
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    out_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Use only the dish region, if found
    image_dish, dish_circle = get_dish_interior(image)

    contours = get_contours(image_dish)
    # Every cell has an outer and inner contour
    n_contours = int(len(contours) / 2)

    print(f"{image_name} : {n_contours}")

    if dish_circle is not None:
        out_image = cv2.circle(out_image, (dish_circle[0], dish_circle[1]), dish_circle[2], RED, 4)
    out_image = cv2.drawContours(out_image, contours, -1, GREEN, 1)
    
    output_path = f"{output_folder}{image_name.split('.')[0]}_out.jpg"
    cv2.imwrite(output_path, out_image)

if __name__=="__main__":
    for directory in DIRECTORIES:
        if not os.path.exists(directory):
            os.makedirs(directory)

    image_list = os.listdir(input_folder) 
    image_list = [x for x in image_list if not (x.startswith('.'))]

    for image_name in image_list:
        image_path = input_folder + image_name
        find_cells(image_path, image_name)
