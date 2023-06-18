#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install opencv-python 


# In[4]:


import cv2
import numpy as np


# In[ ]:


def nothing(x):
    pass

# Create a window to display the webcam feed
cv2.namedWindow("Color Detection")

# Set up trackbars to adjust the color range
cv2.createTrackbar("Lower H", "Color Detection", 0, 255, nothing)
cv2.createTrackbar("Lower S", "Color Detection", 0, 255, nothing)
cv2.createTrackbar("Lower V", "Color Detection", 0, 255, nothing)
cv2.createTrackbar("Upper H", "Color Detection", 255, 255, nothing)
cv2.createTrackbar("Upper S", "Color Detection", 255, 255, nothing)
cv2.createTrackbar("Upper V", "Color Detection", 255, 255, nothing)

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
   
    ret, frame = cap.read()

    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

   
    lower_h = cv2.getTrackbarPos("Lower H", "Color Detection")
    lower_s = cv2.getTrackbarPos("Lower S", "Color Detection")
    lower_v = cv2.getTrackbarPos("Lower V", "Color Detection")
    upper_h = cv2.getTrackbarPos("Upper H", "Color Detection")
    upper_s = cv2.getTrackbarPos("Upper S", "Color Detection")
    upper_v = cv2.getTrackbarPos("Upper V", "Color Detection")

    # Define the lower and upper color range arrays
    lower_range = np.array([lower_h, lower_s, lower_v])
    upper_range = np.array([upper_h, upper_s, upper_v])

    # Create a mask to detect the specified color range
    mask = cv2.inRange(hsv, lower_range, upper_range)

   
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the original frame and the color detection result
    cv2.imshow("Original", frame)
    cv2.imshow("Color Detection", result)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




