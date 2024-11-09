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
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define reference colors for the Rubik's Cube in RGB
COLOR_LABELS = {
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255],
    'yellow': [255, 255, 0],
    'white': [255, 255, 255],
    'orange': [255, 165, 0]  # Approximate RGB for orange
}

def get_color_label(detected_color):
    # Calculate the Euclidean distance between the detected color and each reference color
    distances = {color: np.linalg.norm(np.array(detected_color) - np.array(rgb))
                 for color, rgb in COLOR_LABELS.items()}
    # Find the color with the smallest distance
    closest_color = min(distances, key=distances.get)
    return closest_color

def detect_cube_pieces(image, k=6):
    # Resize the image for consistent processing
    image = cv2.resize(image, (300, 300))

    # Convert to RGB for color processing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define dimensions for a 3x3 grid
    height, width, _ = image.shape
    grid_size = width // 3  # Assuming the cube face is square

    piece_colors = []

    # Loop through each 3x3 grid cell
    for row in range(3):
        row_colors = []
        for col in range(3):
            # Crop each cell
            x_start = col * grid_size
            y_start = row * grid_size
            cell = image[y_start:y_start + grid_size, x_start:x_start + grid_size]

            # Apply K-means on this cell to find dominant color
            cell_reshaped = cell.reshape((-1, 3))
            cell_reshaped = np.float32(cell_reshaped)
            _, labels, centers = cv2.kmeans(cell_reshaped, k, None, 
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
                                            attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

            # Find the most dominant color in this cell
            center_counts = np.bincount(labels.flatten())
            dominant_color = centers[center_counts.argmax()]
            # Convert detected color to the nearest color label
            color_label = get_color_label(np.uint8(dominant_color).tolist())
            row_colors.append(color_label)

        piece_colors.append(row_colors)

    return piece_colors

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
        piece_colors = detect_cube_pieces(image, k=6)

        # Return the 3x3 grid of color labels as JSON
        return jsonify({"piece_colors": piece_colors})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
