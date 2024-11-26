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
import random


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
    # Rotate the yellow face (top face) clockwise
    colors['yellow'] = rotate_face_90_clockwise(colors['yellow'])

    # Store the top rows of adjacent faces
    temp_red_row = colors['red'][0]
    temp_blue_row = colors['blue'][0]
    temp_orange_row = colors['orange'][0]
    temp_green_row = colors['green'][0]

    # Update the top rows
    colors['red'][0] = temp_blue_row  # Top row of red gets top row of blue
    colors['blue'][0] = temp_orange_row  # Top row of blue gets top row of orange
    colors['orange'][0] = temp_green_row  # Top row of orange gets top row of green
    colors['green'][0] = temp_red_row  # Top row of green gets top row of red


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

def ff_ccw(colors):
    """Rotate the front (red) face counterclockwise."""
    colors['red'] = rotate_face_90_counterclockwise(colors['red'])

    temp_yellow_row = colors['yellow'][2]
    temp_blue_col = [row[2] for row in colors['blue']]
    temp_white_row = colors['white'][0]
    temp_green_col = [row[0] for row in colors['green']]

    colors['yellow'][2] = temp_green_col[::-1]
    for i in range(3):
        colors['blue'][i][2] = temp_white_row[i]
    colors['white'][0] = temp_blue_col[::-1]
    for i in range(3):
        colors['green'][i][0] = temp_yellow_row[i]


def bf_ccw(colors):
    """Rotate the back (orange) face counterclockwise."""
    colors['orange'] = rotate_face_90_counterclockwise(colors['orange'])

    temp_yellow_row = colors['yellow'][0]
    temp_green_col = [row[2] for row in colors['green']]
    temp_white_row = colors['white'][2]
    temp_blue_col = [row[0] for row in colors['blue']]

    colors['yellow'][0] = temp_green_col[::-1]
    for i in range(3):
        colors['green'][i][2] = temp_white_row[i]
    colors['white'][2] = temp_blue_col[::-1]
    for i in range(3):
        colors['blue'][i][0] = temp_yellow_row[i]


def lf_ccw(colors):
    """Rotate the left (blue) face counterclockwise."""
    colors['blue'] = rotate_face_90_counterclockwise(colors['blue'])

    temp_yellow_col = [row[0] for row in colors['yellow']]
    temp_orange_col = [row[2] for row in colors['orange']]
    temp_white_col = [row[0] for row in colors['white']]
    temp_red_col = [row[0] for row in colors['red']]

    for i in range(3):
        colors['yellow'][i][0] = temp_red_col[i]
        colors['orange'][i][2] = temp_yellow_col[2 - i]
        colors['white'][i][0] = temp_orange_col[i]
        colors['red'][i][0] = temp_white_col[i]


def rf_ccw(colors):
    """Rotate the right (green) face counterclockwise."""
    colors['green'] = rotate_face_90_counterclockwise(colors['green'])

    temp_yellow_col = [row[2] for row in colors['yellow']]
    temp_red_col = [row[2] for row in colors['red']]
    temp_white_col = [row[2] for row in colors['white']]
    temp_orange_col = [row[0] for row in colors['orange']]

    for i in range(3):
        colors['yellow'][i][2] = temp_red_col[i]
        colors['red'][i][2] = temp_white_col[i]
        colors['white'][i][2] = temp_orange_col[i]
        colors['orange'][i][0] = temp_yellow_col[2 - i]


def tf_ccw(colors):
    """Rotate the top (yellow) face counterclockwise."""
    # Rotate the yellow face (top face) counterclockwise
    colors['yellow'] = rotate_face_90_counterclockwise(colors['yellow'])

    # Store the top rows of adjacent faces
    temp_red_row = colors['red'][0]
    temp_green_row = colors['green'][0]
    temp_orange_row = colors['orange'][0]
    temp_blue_row = colors['blue'][0]

    # Update the top rows
    colors['red'][0] = temp_green_row  # Top row of red gets top row of green
    colors['green'][0] = temp_orange_row  # Top row of green gets top row of orange
    colors['orange'][0] = temp_blue_row  # Top row of orange gets top row of blue
    colors['blue'][0] = temp_red_row  # Top row of blue gets top row of red


def bof_ccw(colors):
    """Rotate the bottom (white) face counterclockwise."""
    colors['white'] = rotate_face_90_counterclockwise(colors['white'])

    temp_red_row = colors['red'][2]
    temp_blue_row = colors['blue'][2]
    temp_orange_row = colors['orange'][2]
    temp_green_row = colors['green'][2]

    colors['red'][2] = temp_blue_row
    colors['blue'][2] = temp_orange_row
    colors['orange'][2] = temp_green_row
    colors['green'][2] = temp_red_row

def create_scramble():
    """Generate a scramble sequence for a Rubik's Cube with random moves."""
    num_moves = random.randint(20, 60)  # Random number of moves between 20 and 60
    faces = ['F', 'B', 'L', 'R', 'U', 'D']  # Front, Back, Left, Right, Up, Down
    modifiers = ['', "'", '2']  # '', ' means clockwise, "' means counterclockwise, '2' means double turn
    
    scramble = []
    for _ in range(num_moves):
        move = random.choice(faces) + random.choice(modifiers)
        scramble.append(move)
    
    return ' '.join(scramble)


def apply_scramble(colors, scramble):
    """Apply a scramble sequence to the cube."""
    scramble_moves = scramble.split()
    for move in scramble_moves:
        if move == "F":
            ff(colors)
        elif move == "F'":
            ff_ccw(colors)
        elif move == "F2":
            ff(colors)
            ff(colors)
        elif move == "B":
            bf(colors)
        elif move == "B'":
            bf_ccw(colors)
        elif move == "B2":
            bf(colors)
            bf(colors)
        elif move == "L":
            lf(colors)
        elif move == "L'":
            lf_ccw(colors)
        elif move == "L2":
            lf(colors)
            lf(colors)
        elif move == "R":
            rf(colors)
        elif move == "R'":
            rf_ccw(colors)
        elif move == "R2":
            rf(colors)
            rf(colors)
        elif move == "U":
            tf(colors)
        elif move == "U'":
            tf_ccw(colors)
        elif move == "U2":
            tf(colors)
            tf(colors)
        elif move == "D":
            bof(colors)
        elif move == "D'":
            bof_ccw(colors)
        elif move == "D2":
            bof(colors)
            bof(colors)
    return colors

@app.route('/scramble',methods=['POST'])
def scramble():
    data = request.get_json()
    colors = data.get("colors")
    scramble = create_scramble()
    apply_scramble(colors,scramble)
    img_buf = render_rubiks_cube(colors)
    img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    return jsonify({
        'colors': colors
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
