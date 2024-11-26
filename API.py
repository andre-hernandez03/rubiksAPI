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

def rotate_face_90_clockwise(face):
    return [list(reversed(col)) for col in zip(*face)]

def rotate_face_90_counterclockwise(face):
    return rotate_face_90_clockwise(rotate_face_90_clockwise(rotate_face_90_clockwise(face)))

def flip_columns(face):
    return [row[::-1] for row in face]

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
    if face == "yellow" and z == 1:
        return colors["yellow"][2 - (y + 1)][x + 1]
    elif face == "white" and z == -1:
        return colors["white"][2 - (y + 1)][x + 1]
    elif face == "blue" and x == -1:
        return colors["blue"][2 - (y + 1)][z + 1]
    elif face == "green" and x == 1:
        right_face = rotate_face_90_counterclockwise(colors["green"])  # 90° clockwise rotation
        right_face = flip_columns(right_face)  # Flip columns after rotation
        return right_face[2 - (y + 1)][z + 1]
    elif face == "orange" and y == 1:
        return colors["orange"][2 - (z + 1)][x + 1]
    elif face == "red" and y == -1:
        return colors["red"][2 - (z + 1)][x + 1]
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
                    get_face_color(colors, x, y, z, "yellow"),  # z+ face
                    get_face_color(colors, x, y, z, "white"),   # z- face
                    get_face_color(colors, x, y, z, "orange"),    # y+ face
                    get_face_color(colors, x, y, z, "red"), # y- face
                    get_face_color(colors, x, y, z, "green"),  # x+ face
                    get_face_color(colors, x, y, z, "blue")    # x- face
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
    colors = request.json.get("colors", {})
    print(colors)
    img_buf = render_rubiks_cube(colors)
    return send_file(img_buf, mimetype='image/png')

# ROTATIONS

@app.route('/rot', methods=['POST']) 
def rotate():
    data = request.get_json()
    colors = data.get("colors")
    rot = data.get("rot")
    match rot:
        case "ff":
            ff(colors)
        case "bf":
            bf(colors)
        case "lf":
            lf(colors)
        case "rf":
            rf(colors)
        case "bof":
            bof(colors)
        case "tf":
            tf(colors)
        case "ffc":
            ff_ccw(colors)
        case "bfc":
            bf_ccw(colors)
        case "lfc":
            lf_ccw(colors)
        case "rfc":
            rf_ccw(colors)
        case "bofc":
            bof_ccw(colors)
        case "tfc":
            tf_ccw(colors)
        case __ :
            return
    img_buf = render_rubiks_cube(colors)
    img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    return jsonify({
        'colors': colors
    }), 200

'''
# Front face clockwise
def ff(colors):
    temp_redface = colors.get('red')
    temp_blueface = colors.get('blue')
    temp_greenface = colors.get('green')
    temp_yellowface = colors.get('yellow')
    temp_whiteface = colors.get('white')

    # Rotate the front face
    for r in range(3):
        for c in range(3):
            temp_redface[r][c] = colors.get('red')[3-1-c][r]

    # Adjust adjacent faces
    temp_yellow_row = [colors['yellow'][2][c] for c in range(3)]
    temp_blue_col = [colors['blue'][r][2] for r in range(3)]
    temp_white_row = [colors['white'][0][c] for c in range(3)]
    temp_green_col = [colors['green'][r][0] for r in range(3)]

    for c in range(3):
        colors['yellow'][2][c] = temp_green_col[2-c]
        colors['blue'][c][2] = temp_yellow_row[c]
        colors['white'][0][c] = temp_blue_col[2-c]
        colors['green'][c][0] = temp_white_row[c]

    # Update the front face
    for r in range(3):
        for c in range(3):
            colors['red'][r][c] = temp_redface[r][c]


# Back face clockwise
def bf(colors):
    temp_orangeface = colors.get('orange')
    temp_blueface = colors.get('blue')
    temp_greenface = colors.get('green')
    temp_yellowface = colors.get('yellow')
    temp_whiteface = colors.get('white')

    # Rotate the back face
    for r in range(3):
        for c in range(3):
            temp_orangeface[r][c] = colors.get('orange')[3-1-c][r]

    # Adjust adjacent faces
    temp_yellow_row = [colors['yellow'][0][c] for c in range(3)]
    temp_green_col = [colors['green'][r][2] for r in range(3)]
    temp_white_row = [colors['white'][2][c] for c in range(3)]
    temp_blue_col = [colors['blue'][r][0] for r in range(3)]

    for c in range(3):
        colors['yellow'][0][c] = temp_blue_col[2-c]
        colors['green'][c][2] = temp_yellow_row[c]
        colors['white'][2][c] = temp_green_col[2-c]
        colors['blue'][c][0] = temp_white_row[c]

    # Update the back face
    for r in range(3):
        for c in range(3):
            colors['orange'][r][c] = temp_orangeface[r][c]


def lf(colors):
    temp_blueface = colors.get('blue')
    temp_redface = colors.get('red')
    temp_yellowface = colors.get('yellow')
    temp_orangeface = colors.get('orange')
    temp_whiteface = colors.get('white')

    # Rotate the left face clockwise
    for r in range(3):
        for c in range(3):
            temp_blueface[r][c] = colors.get('blue')[3-1-c][r]

    # Adjust adjacent faces
    temp_white_col = [colors['white'][r][0] for r in range(3)]
    temp_red_col = [colors['red'][r][0] for r in range(3)]
    temp_yellow_col = [colors['yellow'][2-r][0] for r in range(3)]
    temp_orange_col = [colors['orange'][r][2] for r in range(3)]

    for r in range(3):
        colors['white'][r][0] = temp_orange_col[r]
        colors['red'][r][0] = temp_white_col[r]
        colors['yellow'][2-r][0] = temp_red_col[r]
        colors['orange'][r][2] = temp_yellow_col[r]

    # Update the left face
    for r in range(3):
        for c in range(3):
            colors['blue'][r][c] = temp_blueface[r][c]


def tf(colors):
    temp_yellowface = colors.get('yellow')
    temp_redface = colors.get('red')
    temp_greenface = colors.get('green')
    temp_orangeface = colors.get('orange')
    temp_blueface = colors.get('blue')

    # Rotate the top face clockwise
    for r in range(3):
        for c in range(3):
            temp_yellowface[r][c] = colors.get('yellow')[3-1-c][r]

    # Adjust adjacent faces
    temp_red_row = [colors['red'][0][c] for c in range(3)]
    temp_blue_row = [colors['blue'][0][c] for c in range(3)]
    temp_orange_row = [colors['orange'][0][c] for c in range(3)]
    temp_green_row = [colors['green'][0][c] for c in range(3)]

    for c in range(3):
        colors['red'][0][c] = temp_blue_row[c]
        colors['green'][0][c] = temp_red_row[c]
        colors['orange'][0][c] = temp_green_row[c]
        colors['blue'][0][c] = temp_orange_row[c]

    # Update the top face
    for r in range(3):
        for c in range(3):
            colors['yellow'][r][c] = temp_yellowface[r][c]


def bof(colors):
    temp_whiteface = colors.get('white')
    temp_redface = colors.get('red')
    temp_greenface = colors.get('green')
    temp_orangeface = colors.get('orange')
    temp_blueface = colors.get('blue')

    # Rotate the bottom face clockwise
    for r in range(3):
        for c in range(3):
            temp_whiteface[r][c] = colors.get('white')[3-1-c][r]

    # Adjust adjacent faces
    temp_red_row = [colors['red'][2][c] for c in range(3)]
    temp_blue_row = [colors['blue'][2][c] for c in range(3)]
    temp_orange_row = [colors['orange'][2][c] for c in range(3)]
    temp_green_row = [colors['green'][2][c] for c in range(3)]

    for c in range(3):
        colors['red'][2][c] = temp_green_row[c]
        colors['green'][2][c] = temp_orange_row[c]
        colors['orange'][2][c] = temp_blue_row[c]
        colors['blue'][2][c] = temp_red_row[c]

    # Update the bottom face
    for r in range(3):
        for c in range(3):
            colors['white'][r][c] = temp_whiteface[r][c]


def rf(colors):
    temp_greenface = colors.get('green')
    temp_redface = colors.get('red')
    temp_yellowface = colors.get('yellow')
    temp_orangeface = colors.get('orange')
    temp_whiteface = colors.get('white')

    # Rotate the right face clockwise
    for r in range(3):
        for c in range(3):
            temp_greenface[r][c] = colors.get('green')[3-1-c][r]

    # Adjust adjacent faces
    temp_white_col = [colors['white'][r][2] for r in range(3)]
    temp_red_col = [colors['red'][r][2] for r in range(3)]
    temp_yellow_col = [colors['yellow'][2-r][2] for r in range(3)]
    temp_orange_col = [colors['orange'][r][0] for r in range(3)]

    for r in range(3):
        colors['white'][r][2] = temp_red_col[r]
        colors['red'][r][2] = temp_yellow_col[r]
        colors['yellow'][2-r][2] = temp_orange_col[r]
        colors['orange'][r][0] = temp_white_col[r]

    # Update the right face
    for r in range(3):
        for c in range(3):
            colors['green'][r][c] = temp_greenface[r][c]

'''
def rotate_face_90_clockwise(face):
    """Rotate a 3x3 face 90 degrees clockwise."""
    return [list(reversed(col)) for col in zip(*face)]


def rotate_face_90_counterclockwise(face):
    """Rotate a 3x3 face 90 degrees counterclockwise."""
    return rotate_face_90_clockwise(rotate_face_90_clockwise(rotate_face_90_clockwise(face)))


def ff(colors):
    """Rotate the front (red) face clockwise."""
    # Rotate the red face (front) clockwise
    colors['red'] = rotate_face_90_clockwise(colors['red'])

    # Temporary values for adjacent faces
    temp_yellow_row = colors['yellow'][2]  # Bottom row of yellow (top face)
    temp_green_col = [row[0] for row in colors['green']]  # First column of green (right face)
    temp_white_row = colors['white'][0]  # Top row of white (bottom face)
    temp_blue_col = [row[2] for row in colors['blue']]  # Third column of blue (left face)

    # Update adjacent faces
    for i in range(3):
        colors['green'][i][0] = temp_yellow_row[i]  # First column of green gets bottom row of yellow
    colors['white'][0] = temp_green_col[::-1]  # Top row of white gets reversed first column of green
    for i in range(3):
        colors['blue'][i][2] = temp_white_row[i]  # Third column of blue gets top row of white
    colors['yellow'][2] = temp_blue_col[::-1]  # Bottom 


def bf(colors):
    """Rotate the back (orange) face clockwise."""
    colors['orange'] = rotate_face_90_clockwise(colors['orange'])

    temp_yellow_row = colors['yellow'][0]
    temp_blue_col = [row[0] for row in colors['blue']]
    temp_white_row = colors['white'][2]
    temp_green_col = [row[2] for row in colors['green']]

    colors['yellow'][0] = temp_blue_col[::-1]
    for i in range(3):
        colors['blue'][i][0] = temp_white_row[i]
    colors['white'][2] = temp_green_col[::-1]
    for i in range(3):
        colors['green'][i][2] = temp_yellow_row[i]


def lf(colors):
    """Rotate the left (blue) face clockwise."""
    colors['blue'] = rotate_face_90_clockwise(colors['blue'])

    temp_yellow_col = [row[0] for row in colors['yellow']]
    temp_red_col = [row[0] for row in colors['red']]
    temp_white_col = [row[0] for row in colors['white']]
    temp_orange_col = [row[2] for row in colors['orange']]

    for i in range(3):
        colors['red'][i][0] = temp_yellow_col[i]
        colors['white'][i][0] = temp_red_col[i]
        colors['orange'][i][2] = temp_white_col[i]
        colors['yellow'][i][0] = temp_orange_col[2 - i]


def rf(colors):
    """Rotate the right (green) face clockwise."""
    colors['green'] = rotate_face_90_clockwise(colors['green'])

    temp_yellow_col = [row[2] for row in colors['yellow']]
    temp_orange_col = [row[0] for row in colors['orange']]
    temp_white_col = [row[2] for row in colors['white']]
    temp_red_col = [row[2] for row in colors['red']]

    for i in range(3):
        colors['red'][i][2] = temp_yellow_col[i]
        colors['white'][i][2] = temp_red_col[i]
        colors['orange'][i][0] = temp_white_col[i]
        colors['yellow'][i][2] = temp_orange_col[2 - i]


def tf(colors):
    """Rotate the top (yellow) face clockwise."""
    colors['yellow'] = rotate_face_90_clockwise(colors['yellow'])

    temp_red_row = colors['red'][0]
    temp_blue_row = colors['blue'][0]
    temp_orange_row = colors['orange'][0]
    temp_green_row = colors['green'][0]

    colors['red'][0] = temp_blue_row
    colors['green'][0] = temp_red_row
    colors['orange'][0] = temp_green_row
    colors['blue'][0] = temp_orange_row


def bof(colors):
    """Rotate the bottom (white) face clockwise."""
    colors['white'] = rotate_face_90_clockwise(colors['white'])

    temp_red_row = colors['red'][2]
    temp_green_row = colors['green'][2]
    temp_orange_row = colors['orange'][2]
    temp_blue_row = colors['blue'][2]

    colors['red'][2] = temp_green_row
    colors['green'][2] = temp_orange_row
    colors['orange'][2] = temp_blue_row
    colors['blue'][2] = temp_red_row


# COUNTER CLOCKWISE ROTATIONS

# Front face counterclockwise
def ff_ccw(colors):
    temp_redface = colors.get('red')
    temp_blueface = colors.get('blue')
    temp_greenface = colors.get('green')
    temp_yellowface = colors.get('yellow')
    temp_whiteface = colors.get('white')

    # Rotate the front face counterclockwise
    for r in range(3):
        for c in range(3):
            temp_redface[r][c] = colors.get('red')[c][3-1-r]

    # Adjust adjacent faces
    temp_yellow_row = [colors['yellow'][2][c] for c in range(3)]
    temp_blue_col = [colors['blue'][r][2] for r in range(3)]
    temp_white_row = [colors['white'][0][c] for c in range(3)]
    temp_green_col = [colors['green'][r][0] for r in range(3)]

    for c in range(3):
        colors['yellow'][2][c] = temp_blue_col[c]
        colors['blue'][c][2] = temp_white_row[2-c]
        colors['white'][0][c] = temp_green_col[c]
        colors['green'][c][0] = temp_yellow_row[2-c]

    # Update the front face
    for r in range(3):
        for c in range(3):
            colors['red'][r][c] = temp_redface[r][c]

def bf_ccw(colors):
    temp_orangeface = colors.get('orange')
    temp_blueface = colors.get('blue')
    temp_greenface = colors.get('green')
    temp_yellowface = colors.get('yellow')
    temp_whiteface = colors.get('white')

    # Rotate the back face counterclockwise
    for r in range(3):
        for c in range(3):
            temp_orangeface[r][c] = colors.get('orange')[c][3-1-r]

    # Adjust adjacent faces
    temp_yellow_row = [colors['yellow'][0][c] for c in range(3)]
    temp_blue_col = [colors['blue'][r][0] for r in range(3)]
    temp_white_row = [colors['white'][2][c] for c in range(3)]
    temp_green_col = [colors['green'][r][2] for r in range(3)]

    for c in range(3):
        colors['yellow'][0][c] = temp_green_col[c]
        colors['blue'][c][0] = temp_yellow_row[2-c]
        colors['white'][2][c] = temp_blue_col[c]
        colors['green'][c][2] = temp_white_row[2-c]

    # Update the back face
    for r in range(3):
        for c in range(3):
            colors['orange'][r][c] = temp_orangeface[r][c]

def rf_ccw(colors):
    temp_greenface = colors.get('green')
    temp_redface = colors.get('red')
    temp_yellowface = colors.get('yellow')
    temp_orangeface = colors.get('orange')
    temp_whiteface = colors.get('white')

    # Rotate the right face counterclockwise
    for r in range(3):
        for c in range(3):
            temp_greenface[r][c] = colors.get('green')[c][3-1-r]

    # Adjust adjacent faces
    temp_white_col = [colors['white'][r][2] for r in range(3)]
    temp_red_col = [colors['red'][r][2] for r in range(3)]
    temp_yellow_col = [colors['yellow'][2-r][2] for r in range(3)]
    temp_orange_col = [colors['orange'][r][0] for r in range(3)]

    for r in range(3):
        colors['white'][r][2] = temp_orange_col[r]
        colors['red'][r][2] = temp_white_col[r]
        colors['yellow'][2-r][2] = temp_red_col[r]
        colors['orange'][r][0] = temp_yellow_col[r]

    # Update the right face
    for r in range(3):
        for c in range(3):
            colors['green'][r][c] = temp_greenface[r][c]

def lf_ccw(colors):
    temp_blueface = colors.get('blue')
    temp_redface = colors.get('red')
    temp_yellowface = colors.get('yellow')
    temp_orangeface = colors.get('orange')
    temp_whiteface = colors.get('white')

    # Rotate the left face counterclockwise
    for r in range(3):
        for c in range(3):
            temp_blueface[r][c] = colors.get('blue')[c][3-1-r]

    # Adjust adjacent faces
    temp_white_col = [colors['white'][r][0] for r in range(3)]
    temp_red_col = [colors['red'][r][0] for r in range(3)]
    temp_yellow_col = [colors['yellow'][2-r][0] for r in range(3)]
    temp_orange_col = [colors['orange'][r][2] for r in range(3)]

    for r in range(3):
        colors['white'][r][0] = temp_red_col[r]
        colors['red'][r][0] = temp_yellow_col[2-r]
        colors['yellow'][2-r][0] = temp_orange_col[r]
        colors['orange'][r][2] = temp_white_col[r]

    # Update the left face
    for r in range(3):
        for c in range(3):
            colors['blue'][r][c] = temp_blueface[r][c]

def tf_ccw(colors):
    temp_yellowface = colors.get('yellow')
    temp_redface = colors.get('red')
    temp_greenface = colors.get('green')
    temp_orangeface = colors.get('orange')
    temp_blueface = colors.get('blue')

    # Rotate the top face counterclockwise
    for r in range(3):
        for c in range(3):
            temp_yellowface[r][c] = colors.get('yellow')[c][3-1-r]

    # Adjust adjacent faces
    temp_red_row = [colors['red'][0][c] for c in range(3)]
    temp_blue_row = [colors['blue'][0][c] for c in range(3)]
    temp_orange_row = [colors['orange'][0][c] for c in range(3)]
    temp_green_row = [colors['green'][0][c] for c in range(3)]

    for c in range(3):
        colors['red'][0][c] = temp_green_row[c]
        colors['green'][0][c] = temp_orange_row[c]
        colors['orange'][0][c] = temp_blue_row[c]
        colors['blue'][0][c] = temp_red_row[c]

    # Update the top face
    for r in range(3):
        for c in range(3):
            colors['yellow'][r][c] = temp_yellowface[r][c]

def bof_ccw(colors):
    temp_whiteface = colors.get('white')
    temp_redface = colors.get('red')
    temp_greenface = colors.get('green')
    temp_orangeface = colors.get('orange')
    temp_blueface = colors.get('blue')

    # Rotate the bottom face counterclockwise
    for r in range(3):
        for c in range(3):
            temp_whiteface[r][c] = colors.get('white')[c][3-1-r]

    # Adjust adjacent faces
    temp_red_row = [colors['red'][2][c] for c in range(3)]
    temp_blue_row = [colors['blue'][2][c] for c in range(3)]
    temp_orange_row = [colors['orange'][2][c] for c in range(3)]
    temp_green_row = [colors['green'][2][c] for c in range(3)]

    for c in range(3):
        colors['red'][2][c] = temp_blue_row[c]
        colors['green'][2][c] = temp_red_row[c]
        colors['orange'][2][c] = temp_green_row[c]
        colors['blue'][2][c] = temp_orange_row[c]

    # Update the bottom face
    for r in range(3):
        for c in range(3):
            colors['white'][r][c] = temp_whiteface[r][c]

if __name__ == '__main__':
    app.run(debug=True)
