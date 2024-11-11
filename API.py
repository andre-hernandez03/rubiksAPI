import os
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import base64
from PIL import Image
import numpy as np
import cv2
import pyvista as pv
import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



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
    


# Define the colors for each face of the cube
COLOR_MAP = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "yellow": "yellow",
    "white": "white",
    "orange": "orange",
    "grey": "grey"  # Default color for hidden/empty faces
}

def plot_mini_cube(ax, x, y, z, face_colors):
    # Define vertices for each face of a mini-cube
    faces = [
        [[x-0.5, y-0.5, z+0.5], [x+0.5, y-0.5, z+0.5], [x+0.5, y+0.5, z+0.5], [x-0.5, y+0.5, z+0.5]],  # z+ face
        [[x-0.5, y-0.5, z-0.5], [x+0.5, y-0.5, z-0.5], [x+0.5, y+0.5, z-0.5], [x-0.5, y+0.5, z-0.5]],  # z- face
        [[x-0.5, y+0.5, z-0.5], [x+0.5, y+0.5, z-0.5], [x+0.5, y+0.5, z+0.5], [x-0.5, y+0.5, z+0.5]],  # y+ face
        [[x-0.5, y-0.5, z-0.5], [x+0.5, y-0.5, z-0.5], [x+0.5, y-0.5, z+0.5], [x-0.5, y-0.5, z+0.5]],  # y- face
        [[x+0.5, y-0.5, z-0.5], [x+0.5, y+0.5, z-0.5], [x+0.5, y+0.5, z+0.5], [x+0.5, y-0.5, z+0.5]],  # x+ face
        [[x-0.5, y-0.5, z-0.5], [x-0.5, y+0.5, z-0.5], [x-0.5, y+0.5, z+0.5], [x-0.5, y-0.5, z+0.5]],  # x- face
    ]

    for i, face in enumerate(faces):
        color = COLOR_MAP.get(face_colors[i], "grey")
        poly3d = [face]
        ax.add_collection3d(Poly3DCollection(poly3d, color=color, edgecolor="black"))

def get_face_color(colors, x, y, z, face):
    # Use coordinates to get the correct color from the `colors` dictionary for each face
    if face == "red" and z == 1:
        return colors["red"][2 - (y + 1)][x + 1]
    elif face == "orange" and z == -1:
        return colors["orange"][2 - (y + 1)][x + 1]
    elif face == "green" and x == -1:
        return colors["green"][2 - (y + 1)][z + 1]
    elif face == "blue" and x == 1:
        return colors["blue"][2 - (y + 1)][z + 1]
    elif face == "white" and y == 1:
        return colors["white"][2 - (z + 1)][x + 1]
    elif face == "yellow" and y == -1:
        return colors["yellow"][2 - (z + 1)][x + 1]
    return "grey"  # Default color for hidden faces

def render_rubiks_cube(colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                if x == 0 and y == 0 and z == 0:
                    continue  # Skip the center mini-cube

                # Define the color of each face for this mini-cube
                face_colors = [
                    get_face_color(colors, x, y, z, "red"),  # z+ face
                    get_face_color(colors, x, y, z, "orange"),   # z- face
                    get_face_color(colors, x, y, z, "white"),    # y+ face
                    get_face_color(colors, x, y, z, "yellow"), # y- face
                    get_face_color(colors, x, y, z, "blue"),  # x+ face
                    get_face_color(colors, x, y, z, "green")    # x- face
                ]

                plot_mini_cube(ax, x, y, z, face_colors)

    ax.set_box_aspect([1, 1, 1])
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf

@app.route('/render_cube', methods=['POST'])
def render_cube_endpoint():
    colors = request.json.get("cubeLayouts", {})
    print(colors)
    img_buf = render_rubiks_cube(colors)
    return send_file(img_buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
