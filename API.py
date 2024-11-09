'''
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2


# Flask/ Server
app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return "Hello, Flask!"

@app.route('/detect',methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Ensure the uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Verify the file was saved
        if not os.path.exists(file_path):
            return jsonify({"error": "File not saved"}), 500

        # Verify OpenCV can read the file
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({"error": "cv2.imread could not read the image. Check format compatibility"}), 500

        layout = detect_colors(image)
        return jsonify({"layout": layout})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


# convert image for opencv
def convert_image(base64_str):
    decoded = base64.b64decode(base64_str)
    img = Image.open(BytesIO(decoded))
    return np.array(img)

# Define color ranges
color_ranges = {
    'red': [(0, 100, 100), (10, 255, 255)],
    'green': [(40, 40, 40), (70, 255, 255)],
    'blue': [(90, 50, 50), (130, 255, 255)],
    'yellow': [(20, 100, 100), (30, 255, 255)],
    'white': [(0, 0, 200), (180, 30, 255)],
    'orange': [(10, 100, 100), (20, 255, 255)]
}

# Function to classify color of a region
def classify_color(hsv_region):
    avg_hue = np.mean(hsv_region[:, :, 0])
    avg_saturation = np.mean(hsv_region[:, :, 1])
    avg_value = np.mean(hsv_region[:, :, 2])
    
    # Loop through color ranges and classify based on HSV value
    for color, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)
        if lower_bound[0] <= avg_hue <= upper_bound[0] and lower_bound[1] <= avg_saturation <= upper_bound[1] and lower_bound[2] <= avg_value <= upper_bound[2]:
            return color
    return 'white'  # If no color matches

# Function to detect Rubik's cube colors
def detect_colors(image):
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get image dimensions and calculate the bounding box for the cube (60% of the image)
    height, width, _ = image.shape
    side_length = int(min(height, width) * 0.6)  # 60% of the smaller dimension
    x_start = (width - side_length) // 2
    y_start = (height - side_length) // 2

    # Crop the bounding box
    cube_image = hsv_image[y_start:y_start+side_length, x_start:x_start+side_length]

    # Divide the cropped image into a 3x3 grid
    grid_size = side_length // 3
    layout = []

    for row in range(3):
        row_colors = []
        for col in range(3):
            # Extract the region corresponding to each square on the cube
            x = col * grid_size
            y = row * grid_size
            grid_region = cube_image[y:y+grid_size, x:x+grid_size]

            # Classify the color in this grid region
            dominant_color = classify_color(grid_region)
            row_colors.append(dominant_color)

        layout.append(row_colors)
        print(layout)

    return layout
    
def model(white,yellow,red,orange,green,blue):
    return

if __name__ == '__main__':
    app.run(debug=True)
'''
import os
import cv2
import numpy as np
from skimage import color
from skimage.segmentation import slic
from skimage.future import graph
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define reference colors for Rubik's Cube colors in LAB format (more robust for color matching)
COLOR_LABELS = {
    'red': color.rgb2lab(np.array([[[255, 0, 0]]], dtype=np.uint8))[0][0],
    'green': color.rgb2lab(np.array([[[0, 255, 0]]], dtype=np.uint8))[0][0],
    'blue': color.rgb2lab(np.array([[[0, 0, 255]]], dtype=np.uint8))[0][0],
    'yellow': color.rgb2lab(np.array([[[255, 255, 0]]], dtype=np.uint8))[0][0],
    'white': color.rgb2lab(np.array([[[255, 255, 255]]], dtype=np.uint8))[0][0],
    'orange': color.rgb2lab(np.array([[[255, 165, 0]]], dtype=np.uint8))[0][0]
}

def get_color_label(detected_color):
    # Calculate the Euclidean distance between the detected color and each reference color in LAB space
    distances = {color_name: np.linalg.norm(detected_color - lab_color)
                 for color_name, lab_color in COLOR_LABELS.items()}
    
    # Find the color with the smallest distance
    closest_color = min(distances, key=distances.get)
    return closest_color

def detect_cube_pieces_scikit(image):
    # Resize for consistent processing
    image = cv2.resize(image, (300, 300))

    # Convert image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to LAB color space for better color differentiation
    lab_image = color.rgb2lab(image_rgb)

    # Perform SLIC segmentation to divide image into regions (segments)
    segments = slic(lab_image, n_segments=9, compactness=10, start_label=1)

    # Calculate the mean LAB color of each segment and classify it
    piece_colors = []
    for segment_value in np.unique(segments):
        # Mask to select pixels in this segment
        mask = segments == segment_value
        mean_lab_color = lab_image[mask].mean(axis=0)
        
        # Get color label for the mean color
        color_label = get_color_label(mean_lab_color)
        piece_colors.append(color_label)

    # Reshape piece_colors list into a 3x3 grid for the Rubik's Cube face
    piece_colors_grid = [piece_colors[i:i+3] for i in range(0, 9, 3)]
    
    return piece_colors_grid

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Ensure the uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Load the saved image
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({"error": "Failed to read image"}), 500

        # Detect the colors in a 3x3 grid format
        piece_colors = detect_cube_pieces_scikit(image)

        # Return the 3x3 grid of color labels as JSON
        return jsonify({"piece_colors": piece_colors})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
