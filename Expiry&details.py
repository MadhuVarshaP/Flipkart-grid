# import easyocr
# import cv2
# from matplotlib import pyplot as plt

# # Path to the image
# Image_Path = 'Image4.jpg'

# # Initialize the EasyOCR reader
# reader = easyocr.Reader(['en'], gpu=False)

# # Perform OCR to detect text
# result = reader.readtext(Image_Path)

# # Load the image using OpenCV
# img = cv2.imread(Image_Path)

# # Loop through the results to draw rectangles and text
# for detection in result:
#     # Get the top-left and bottom-right coordinates of the bounding box
#     top_left = tuple([int(val) for val in detection[0][0]])
#     bottom_right = tuple([int(val) for val in detection[0][2]])
    
#     # Get the detected text
#     text = detection[1]
    
#     # Define font for the text
#     font = cv2.FONT_HERSHEY_SIMPLEX
    
#     # Draw the rectangle around the detected text
#     img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 5)
    
#     # Put the text on the image
#     img = cv2.putText(img, text, top_left, font, 1, (255, 255, 255), 2, cv2.LINE_AA)

# # Display the final image with all bounding boxes and text at once
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying with matplotlib
# plt.axis('off')  # Hide axis
# plt.show()

# # Print all the detected text in the output
# for detection in result:
#     print(f"Detected text: {detection[1]} with confidence {detection[2]}")
import easyocr
import cv2
from matplotlib import pyplot as plt
import re

# Path to the image
Image_Path = r'C:\Users\uvara\OneDrive\Desktop\lets begin\Brandrecognition2\env\IMG20241016191526.jpg'

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Perform OCR to detect text
result = reader.readtext(Image_Path)

# Load the image using OpenCV
img = cv2.imread(Image_Path)

# Initialize an empty string to store the detected text
detected_paragraph = ""

# Expanded regular expression pattern for date detection
date_pattern = r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[.\-]?\s?\d{2,4})|' \
               r'(\d{1,2}[-/]\d{2,4})|' \
               r'(\d{4}[-/]\d{1,2})'

# List to store detected dates
detected_dates = []

# Loop through the results to draw rectangles and accumulate text
for detection in result:
    # Get the top-left and bottom-right coordinates of the bounding box
    top_left = tuple([int(val) for val in detection[0][0]])
    bottom_right = tuple([int(val) for val in detection[0][2]])
    
    # Get the detected text
    text = detection[1]
    
    # Append the text to the paragraph
    detected_paragraph += text + " "
    
    # Check if the text matches the expanded date pattern
    dates = re.findall(date_pattern, text)
    if dates:
        detected_dates.extend([max(date) for date in dates if max(date)])  # Add only the valid match
    
    # Define font for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Draw the rectangle around the detected text
    img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 5)
    
    # Put the text on the image
    img = cv2.putText(img, text, top_left, font, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Display the final image with all bounding boxes and text at once
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying with matplotlib
plt.axis('off')  # Hide axis
# plt.show()

# Print the detected text as a paragraph
print("Detected Text:")
print(detected_paragraph.strip())

# Print detected dates if any are found
if detected_dates:
    print("\nDetected Dates:")
    for date in detected_dates:
        print(date)
else:
    print("\nNo dates detected.")


