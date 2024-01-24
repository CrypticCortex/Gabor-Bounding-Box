import os
import json
import cv2
# Load image
img_path = '/home/kalyan/gitrepo/NeedToStartARepo/iitb/ilovepdf_pages-to-jpg/Sanskrit_Text_page-0005.jpg'
img = cv2.imread(img_path) 

# Create Gabor filter bank
gabor_kernels = cv2.getGaborKernel((5, 5), 4.0, theta=0, lambd=10.0, gamma=0.5)

# Filter image with Gabor filters
gabor_imgs = []
for kernel in gabor_kernels:
    filtered_img = cv2.filter2D(img, -1, kernel)
    gabor_imgs.append(filtered_img)

# Detect edges
edges = cv2.Canny(gabor_imgs[0], 50, 200)

# Detect lines using Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=10, maxLineGap=5)

# Group lines based on y-coordinates
lines_sorted = sorted(lines, key=lambda line: line[0][1])  # Sort lines based on y-coordinates
grouped_lines = []
current_group = [lines_sorted[0]]
for i in range(1, len(lines_sorted)):
    if abs(lines_sorted[i][0][1] - lines_sorted[i-1][0][1]) < 6:  # Tweak the threshold value for grouping
        current_group.append(lines_sorted[i])
    else:
        grouped_lines.append(current_group)
        current_group = [lines_sorted[i]]
grouped_lines.append(current_group)

# Create a new folder with the same name as the original image
img_folder = os.path.splitext(os.path.basename(img_path))[0]  # Get the image file name without extension
output_folder = os.path.join(os.path.dirname(img_path), img_folder)
os.makedirs(output_folder, exist_ok=True)

# Create a dictionary to store bounding box coordinates
bbox_dict = {}

def convert_np_int_to_int(obj):
    if isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, list):
        return [convert_np_int_to_int(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_np_int_to_int(value) for key, value in obj.items()}
    else:
        return obj
# Draw bounding boxes on original image for each text line and save coordinates to the dictionary
for idx, line_group in enumerate(grouped_lines, start=1):
    min_x = min(line[0][0] for line in line_group)
    min_y = min(line[0][1] for line in line_group)
    max_x = max(line[0][2] for line in line_group)
    max_y = max(line[0][3] for line in line_group)
    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2) #(0, 255, 0) is 

    # Save the cropped and boxed image in the new folder
    cropped_img = img[min_y:max_y, min_x:max_x]
    if not cropped_img.any():  # Check if cropped_img is empty
        print(f"Warning: Cropped image is empty for box {idx}. Skipping...")
    else:
        boxed_img_path = os.path.join(output_folder, f"boxed_line_{idx}.jpg")
        cv2.imwrite(boxed_img_path, cropped_img)

        # Store coordinates in the dictionary
        box_name = f"box{idx}"
        bbox_dict[box_name] = {
            "top_left": [min_x, min_y],
            "top_right": [max_x, min_y],
            "bottom_left": [min_x, max_y],
            "bottom_right": [max_x, max_y]
        }

# Save the dictionary as a JSON file in the same folder as the original image
json_file_path = os.path.join(output_folder, 'bounding_boxes.json')
with open(json_file_path, 'w') as json_file:
    json.dump(convert_np_int_to_int(bbox_dict), json_file, indent=4)


# Save the modified image with bounding boxes
cv2.imwrite(os.path.join(output_folder, 'Overall.jpg'), img)
