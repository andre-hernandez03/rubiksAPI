import os
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import base64
from PIL import Image
import numpy as np
import cv2
import io
import random
import kociemba


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

'''
color_ranges = {
    'red': [(0, 100, 100), (10, 255, 255)],
    'green': [(40, 40, 40), (70, 255, 255)],
    'blue': [(90, 50, 50), (130, 255, 255)],
    'yellow': [(20, 100, 100), (30, 255, 255)],
    'orange': [(10, 100, 100), (20, 255, 255)]
}
'''

color_ranges = {
    'red': [(0, 120, 70), (10, 255, 255)],
    'green':[(35,100,70),(85,255,255)],
    'blue': [(90,100,70),(130,255,255)],
    'yellow':[(25,120,70),(35,255,255)],
    'orange': [(10,120,70),(25,255,255)]
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

# ROTATIONS

@app.route('/rot', methods=['POST']) 
def rotate():
    data = request.get_json()
    colors = data.get("colors")
    rot = data.get("rot")
    match rot:
        case "ff" | "F":
            ff(colors)
        case "bf" | "B":
            bf(colors)
        case "lf" | "L":
            lf(colors)
        case "rf" | "R":
            rf(colors)
        case "bof" | "D":
            bof(colors)
        case "tf" | "U":
            tf(colors)
        case "ffc" | "F'":
            ff_ccw(colors)
        case "bfc" | "B'":
            bf_ccw(colors)
        case "lfc" | "L'":
            lf_ccw(colors)
        case "rfc" | "R'":
            rf_ccw(colors)
        case "bofc" | "D'":
            bof_ccw(colors)
        case "tfc" | "U'":
            tf_ccw(colors)
        case __ :
            return
    return jsonify({
        'colors': colors
    }), 200


def rotate_face_90_counterclockwise(face):
    return rotate_face_90_clockwise(rotate_face_90_clockwise(rotate_face_90_clockwise(face)))

def rotate_face_90_clockwise(face):
    return [list(reversed(col)) for col in zip(*face)]

def ff(colors):
    colors['red'] = rotate_face_90_clockwise(colors['red'])
    
    temp_yellow_row = colors['yellow'][2]               # Yellow’s bottom row (adjacent to front)
    temp_green_col = [row[0] for row in colors['green']]  # Green’s left column (adjacent to front)
    temp_white_row = colors['white'][0]                   # White’s top row (adjacent to front)
    temp_blue_col = [row[2] for row in colors['blue']]    # Blue’s right column (adjacent to front)
    
    for i in range(3):
        colors['green'][i][0] = temp_yellow_row[i]        # Green left column <- Yellow bottom row
    colors['white'][0] = temp_green_col[::-1]             # White top row <- (reversed) Green left column
    for i in range(3):
        colors['blue'][i][2] = temp_white_row[i]          # Blue right column <- White top row
    colors['yellow'][2] = temp_blue_col[::-1]             # Yellow bottom row <- (reversed) Blue right column

def bf(colors):
    colors['orange'] = rotate_face_90_clockwise(colors['orange'])
    
    # Save adjacent edges.
    A = colors['yellow'][0].copy()              # yellow top row
    B = [row[2] for row in colors['green']]      # green right column
    C = colors['white'][2].copy()                # white bottom row
    D = [row[0] for row in colors['blue']]       # blue left column

    # Cycle the edges with swapped reversal:
    # Blue left column becomes reversed(A)
    for i in range(3):
        colors['blue'][i][0] = A[::-1][i]
    
    # White bottom row becomes D direct
    colors['white'][2] = D.copy()
    
    # Green right column becomes reversed(C)
    for i in range(3):
        colors['green'][i][2] = C[::-1][i]
    
    # Yellow top row becomes B direct
    colors['yellow'][0] = B.copy()

def rf(colors):
    colors['green'] = rotate_face_90_clockwise(colors['green'])
    
    A = [row[2] for row in colors['yellow']]  # yellow right column
    B = [row[0] for row in colors['orange']]    # orange left column
    C = [row[2] for row in colors['white']]     # white right column
    D = [row[2] for row in colors['red']]       # red right column

    # Cycle:
    # Orange left column becomes A (direct)
    for i in range(3):
        colors['orange'][i][0] = A[::-1][i]
    # White right column becomes reversed(original B)
    for i in range(3):
        colors['white'][i][2] = B[::-1][i]
    # Red right column becomes C (direct)
    for i in range(3):
        colors['red'][i][2] = C[i]
    # Yellow right column becomes reversed(original D)
    for i in range(3):
        colors['yellow'][i][2] = D[i]

def lf(colors):
    colors['blue'] = rotate_face_90_clockwise(colors['blue'])
    
    temp_up = [row[0] for row in colors['yellow']]      # Yellow’s left column
    temp_red = [row[0] for row in colors['red']]          # Red’s left column
    temp_down = [row[0] for row in colors['white']]       # White’s left column
    temp_orange = [row[2] for row in colors['orange']]    # Orange’s right column (adjacent to blue)
    
    for i in range(3):
        colors['red'][i][0] = temp_up[i]                  # Red left column <- Yellow left column
    for i in range(3):
        colors['white'][i][0] = temp_red[i]               # White left column <- Red left column
    for i in range(3):
        colors['orange'][i][2] = temp_down[::-1][i]       # Orange right column <- (reversed) White left column
    for i in range(3):
        colors['yellow'][i][0] = temp_orange[::-1][i]     # Yellow left column <- (reversed) Orange right column

def tf(colors):
    colors['yellow'] = rotate_face_90_clockwise(colors['yellow'])
    
    # Save copies of the top rows of the four adjacent faces.
    temp_red = colors['red'][0].copy()       # Front (red) top row
    temp_green = colors['green'][0].copy()     # Right (green) top row
    temp_orange = colors['orange'][0].copy()   # Back (orange) top row
    temp_blue = colors['blue'][0].copy()       # Left (blue) top row

    # Cycle the edges in the clockwise direction.
    colors['red'][0] = temp_green            # Front gets Right
    colors['green'][0] = temp_orange         # Right gets Back
    colors['orange'][0] = temp_blue          # Back gets Left
    colors['blue'][0] = temp_red             # Left gets Front


def bof(colors):
    colors['white'] = rotate_face_90_clockwise(colors['white'])
    
    temp = colors['red'][2].copy()
    colors['red'][2]    = colors['blue'][2].copy()
    colors['blue'][2]   = colors['orange'][2].copy()
    colors['orange'][2] = colors['green'][2].copy()
    colors['green'][2]  = temp


# COUNTER CLOCKWISE ROTATIONS

def ff_ccw(colors):
    """Rotate the front (red) face 90° counterclockwise."""
    for _ in range(3):
        ff(colors)

def bf_ccw(colors):
    """Rotate the back (orange) face 90° counterclockwise."""
    for _ in range(3):
        bf(colors)

def rf_ccw(colors):
    """Rotate the right (green) face 90° counterclockwise."""
    for _ in range(3):
        rf(colors)

def lf_ccw(colors):
    """Rotate the left (blue) face 90° counterclockwise."""
    for _ in range(3):
        lf(colors)

def tf_ccw(colors):
    """Rotate the top (yellow) face 90° counterclockwise."""
    for _ in range(3):
        tf(colors)

def bof_ccw(colors):
    """Rotate the bottom (white) face 90° counterclockwise."""
    for _ in range(3):
        bof(colors)


def create_scramble():
    """Generate a scramble sequence for a Rubik's Cube with random moves."""
    num_moves = random.randint(20, 30)  # Random number of moves between 20 and 60
    faces = ['F', 'B', 'L', 'R', 'U', 'D']  # Front, Back, Left, Right, Up, Down
    modifiers = ['', "'"]  # '', ' means clockwise, "' means counterclockwise
    
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
        elif move == "B":
            bf(colors)
        elif move == "B'":
            bf_ccw(colors)
        elif move == "L":
            lf(colors)
        elif move == "L'":
            lf_ccw(colors)
        elif move == "R":
            rf(colors)
        elif move == "R'":
            rf_ccw(colors)
        elif move == "U":
            tf(colors)
        elif move == "U'":
            tf_ccw(colors)
        elif move == "D":
            bof(colors)
        elif move == "D'":
            bof_ccw(colors)
    #print(scramble_moves)
    return colors

@app.route('/scramble',methods=['POST'])
def scramble():
    data = request.get_json()
    colors = data.get("colors")
    scramble = create_scramble()
    print(scramble)
    apply_scramble(colors,scramble)
    return jsonify({
        'colors': colors,
        'scramble': scramble
    }), 200

@app.route('/kociemba',methods=['POST'])
def k_algorithm():
    data = request.get_json()
    colors = data.get("colors")
    solution = kociemba.solve(colors)
    return jsonify({
        'solution': solution
    }), 200
    print(colors)

if __name__ == '__main__':
    app.run(debug=True)
