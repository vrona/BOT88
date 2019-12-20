import numpy as np
from PIL import ImageGrab
# from draw_lines import draw_lines
import cv2
import time

# explicite what are the lines 
def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [255,0,0], 2)
    except:

        pass
# region of interest
def roi(img, vertices):
    # blank mask:
    mask = np.zeros_like(img)

    # defining the number of  color channel to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[3]  # 3 channels 
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    # fill the mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked

# processed the grabbed image into Gray > Edge > Smooth > Mask (region of interest) > lines drawing
# from the edge detector
def hough_lines(original_image):
    min_line_len = 100 # 300
    max_line_gap = 100 # 100
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)  #  GRAY HSV hue, saturation, value
    processed_img = cv2.Canny(processed_img, threshold1=210, threshold2=500)
    processed_img = cv2.GaussianBlur(processed_img, (1,1), 0)
    vertices = np.array([[0,635],[0,317], [200,270], [600,270], [800,317], [800,635]], np.int32)
    roi_img = roi(processed_img, [vertices])

    # rho=2 and theta= np.pi/180 are the distance and angular resolution of the grid in Hough space.
    # threshold = 180 is minimum number of intersections in a grid for candidate line to go to output
    lines = cv2.HoughLinesP(roi_img, 2, np.pi/180, 180, np.array([]), minLineLength= min_line_len, maxLineGap=max_line_gap)
    line_processed_img = np.zeros((roi_img.shape[0], roi_img.shape[1], 3), dtype=np.uint8)

    # draw_lines(roi_img, lines)
    draw_lines(line_processed_img, lines)
    return line_processed_img

# Merge of image processed + image with lines
def weighted_img(processed_img, original_image, α=0.8, β=1., λ=0.):
    
    return cv2.addWeighted(original_image, α, processed_img, β, λ)

# Grabbing the image from screen upper left (position 0,40) with resolution of 635x800
# then processed the previous functions until resize the inital input to 89x120 while avoiding loosing min. info
def grab2np():
    last_time = time.time()
    while True:
        
        # 800x635 windowed mode
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 635)))
        
        last_time = time.time()
        new_screen = hough_lines(screen)
        new_screen = weighted_img(new_screen, screen, α=0.8, β=1., λ=0.)
        new_screen = cv2.resize(new_screen, None,fx=0.15,fy=0.15,interpolation=cv2.INTER_AREA)
        
        # cv2.imshow('window', new_screen)
        cv2.imshow('window',cv2.cvtColor(new_screen, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        return new_screen
# grab2np()