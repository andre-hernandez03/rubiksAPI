import os
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import base64
from PIL import Image
import numpy as np
import cv2
import pyvista as pv
import io

os.environ["DISPLAY"] = ":0"         # Avoids X server errors
os.environ["PYVISTA_OFF_SCREEN"] = "true"  # Ensures PyVista stays off-screen
os.environ["PYVISTA_USE_IPYVTK"] = "false" # Disables IPython-specific rendering

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
    img = Image.open(io.BytesIO(decoded))
    return np.array(img)

# Define color ranges
color_ranges = {
    'red': [(0, 100, 100), (10, 255, 255)],
    'green': [(40, 40, 40), (70, 255, 255)],
    'blue': [(90, 50, 50), (130, 255, 255)],
    'yellow': [(20, 100, 100), (30, 255, 255)],
    'orange': [(10, 100, 100), (20, 255, 255)]
}
#'white': [(0, 0, 200), (180, 30, 255)]

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
    
# Define colors for the cube faces
COLOR_MAP = {
    'blue': [0, 0, 1],
    'green': [0, 1, 0],
    'red': [1, 0, 0],
    'yellow': [1, 1, 0],
    'white': [1, 1, 1],
    'orange': [1, 0.5, 0],
    'grey': [0.5, 0.5, 0.5],  # Default color for hidden faces
}

def render_cube(colors):
    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = 'white'

    # Create the 3x3x3 Rubik's cube as individual blocks
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                if x == 0 and y == 0 and z == 0:
                    continue  # Skip the center cube

                block = pv.Cube(center=(x, y, z), x_length=0.9, y_length=0.9, z_length=0.9)
                # Apply color to the block based on the position
                block_color = COLOR_MAP.get(colors.get((x, y, z), 'grey'))
                plotter.add_mesh(block, color=block_color, show_edges=True)

    # Render to an image
    img_array = plotter.screenshot(return_img=True)
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

@app.route('/render_cube', methods=['POST'])
def render_cube_endpoint():
    data = request.json
    colors = data.get('cubeLayouts', {})

    # Render the cube and get the image in a buffer
    img_buf = render_cube(colors)
    return send_file(img_buf, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
