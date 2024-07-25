from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
#default open window
@app.route('/')
def index():
    return "Hello, World!"


def calculate_fwhm(x, y, peak):
    half_max = y[peak] / 2.0
    left_idx = np.where(y[:peak] < half_max)[0]
    right_idx = np.where(y[peak:] < half_max)[0] + peak
    
    if left_idx.size == 0:
        left_idx = 0
    else:
        left_idx = left_idx[-1]
        
    if right_idx.size == 0:
        right_idx = len(y) - 1
    else:
        right_idx = right_idx[0]
        
    fwhm = x[right_idx] - x[left_idx]
    return fwhm, x[left_idx], x[right_idx]

def identify_raman_band(x):
    if 1570 <= x <= 1590:
        return "G Band"
    elif 2600 <= x <= 2800:
        return "2D Band"
    else:
        return "<NA>"
#make the determine symmetery like the find-symmetery
def find_symmetry(new_left_x_value, original_x_value, new_right_x_value):
    if new_right_x_value is None or original_x_value is None:
        return float('inf')
    if abs(new_right_x_value - original_x_value) < 1e-10:
        return float('inf')    
    s = (new_left_x_value - original_x_value) / (new_right_x_value - original_x_value)
    s = round(s, 1)
    return abs(s)

def determine_symmetry(left_val, right_val, x):
    new_left_x_value = left_val
    new_right_x_value = right_val
    original_x_value = x
    return find_symmetry(new_left_x_value, original_x_value, new_right_x_value)


def determine_syminfo(symmetry):
    if 0.9 <= symmetry <= 1.1:
        return "Single Layer Graphene"
    else:
        return "Multi Layer Graphene"

@app.route('/detect-peaks', methods=['POST'])
def detect_peaks():
    file = request.files['file']
    df = pd.read_csv(file, header=None)
    
    wavelengths = df[0].to_numpy()
    intensities = df[1].to_numpy()

    peaks, _ = find_peaks(intensities, height=0)
    peak_data = []
    
    for i, peak in enumerate(peaks):
        fwhm, left_val, right_val = calculate_fwhm(wavelengths, intensities, peak)
        x = wavelengths[peak]
        symmetry = determine_symmetry(left_val, right_val, x)
        peak_data.append({
            "peak": i + 1,
            "x": float(x),
            "y": float(intensities[peak]),
            "right_val": float(right_val - x),
            "left_val": float(x - left_val),
            "raman_band": identify_raman_band(x),
            "xl": float(left_val),
            "xr": float(right_val),
            "symmetry": float(symmetry),
            "fwhm": float(fwhm),
            "syminfo": determine_syminfo(symmetry)
        })

    return jsonify(peak_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)