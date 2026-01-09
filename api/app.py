from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import signal
from flask_cors import CORS
import os
import io

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Hello, World!"

def smooth_data(x, y, window_length):
    if window_length < 3:
        window_length = 3
    if window_length % 2 == 0:
        window_length += 1
    return signal.savgol_filter(y, window_length, 2)

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
    """Broad band assignment based on shift ranges."""
    if 1330 <= x <= 1360: return "D Band (C)"
    if 1570 <= x <= 1590: return "G Band (C)"
    if 2680 <= x <= 2715: return "2D Band (Graphene)"
    if 2716 <= x <= 2760: return "2D Band (Graphite)"
    if 350 <= x <= 362: return "E¹₂g (WS₂)"
    if 415 <= x <= 425: return "A₁g (WS₂)"
    if 380 <= x <= 390: return "E¹₂g (MoS₂)"
    if 400 <= x <= 412: return "A₁g (MoS₂)"
    return "Unknown"

def find_symmetry(new_left_x_value, original_x_value, new_right_x_value):
    if new_right_x_value is None or original_x_value is None:
        return float('inf')
    if abs(new_right_x_value - original_x_value) < 1e-10:
        return float('inf')    
    s = (new_left_x_value - original_x_value) / (new_right_x_value - original_x_value)
    s = round(s, 1)
    return abs(s)

def determine_symmetry(left_val, right_val, x):
    return find_symmetry(left_val, x, right_val)

def determine_syminfo(symmetry):
    if 0.9 <= symmetry <= 1.1:
        return "Symmetric (likely Single Layer)"
    else:
        return "Asymmetric (likely Multi-layer)"

@app.route('/detect-peaks', methods=['POST'])
def detect_peaks():
    file = request.files['file']
    threshold = float(request.form.get('threshold', 0))
    
    # Read the CSV data from the file
    file_content = file.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(file_content), header=None)
    
    shifts = df[0].to_numpy()
    intensities = df[1].to_numpy()

    # Smooth the data
    window_length = max(3, int(len(intensities) * 0.02))
    smoothed_intensities = smooth_data(shifts, intensities, window_length)

    # Detect peaks based on threshold
    peaks, _ = find_peaks(smoothed_intensities, height=threshold)
    peak_data = []
    
    # Material Analysis Storage
    materials_detected = []
    
    # Helper to find peak in range
    def find_p(low, high):
        best = None
        m_int = -1
        for p in peaks:
            if low <= shifts[p] <= high:
                if smoothed_intensities[p] > m_int:
                    m_int = smoothed_intensities[p]
                    best = p
        return best

    # 1. Graphene / Graphite Analysis
    g_p = find_p(1570, 1590)
    if g_p is not None:
        g_pos = shifts[g_p]
        g_int = smoothed_intensities[g_p]
        strain = (g_pos - 1581) / -14.0
        
        # Check Graphite first
        two_d_graphite = find_p(2716, 2760)
        if two_d_graphite:
            ratio = smoothed_intensities[two_d_graphite] / g_int
            if ratio < 0.5:
                materials_detected.append({
                    "type": "Graphite", 
                    "strain": round(strain, 2),
                    "2d_pos": round(shifts[two_d_graphite], 1),
                    "ratio": round(ratio, 2)
                })
        
        # Check Graphene if not Graphite
        if not any(m["type"] == "Graphite" for m in materials_detected):
            two_d_graphene = find_p(2680, 2715)
            if two_d_graphene is not None:
                ratio = smoothed_intensities[two_d_graphene] / g_int
                layer = "Monolayer" if ratio > 2 else "Bilayer" if ratio > 1 else "Multilayer"
                materials_detected.append({
                    "type": "Graphene", 
                    "layer": layer, 
                    "strain": round(strain, 2), 
                    "i2d_ig": round(ratio, 2)
                })

    # 2. WS2 Analysis
    ws2_e = find_p(350, 362)
    ws2_a = find_p(415, 425)
    if ws2_e is not None and ws2_a is not None:
        e_pos = shifts[ws2_e]
        a_pos = shifts[ws2_a]
        delta = a_pos - e_pos
        strain = (e_pos - 356.5) / -0.66
        layer = "Monolayer" if delta < 65 else "Bulk"
        materials_detected.append({"type": "WS2", "layer": layer, "strain": round(strain, 2), "delta": round(delta, 1)})

    # 3. MoS2 Analysis
    mos2_e = find_p(380, 390)
    mos2_a = find_p(400, 412)
    if mos2_e is not None and mos2_a is not None:
        e_pos = shifts[mos2_e]
        a_pos = shifts[mos2_a]
        delta = a_pos - e_pos
        strain = (e_pos - 383) / -1.5
        layer = "Monolayer" if delta < 22 else "Bulk"
        materials_detected.append({"type": "MoS2", "layer": layer, "strain": round(strain, 2), "delta": round(delta, 1)})

    for i, peak in enumerate(peaks):
        fwhm, left_val, right_val = calculate_fwhm(shifts, smoothed_intensities, peak)
        x = shifts[peak]
        symmetry = determine_symmetry(left_val, right_val, x)
        peak_data.append({
            "peak": i + 1,
            "x": float(x),
            "y": float(smoothed_intensities[peak]),
            "symmetry": float(symmetry),
            "fwhm": float(fwhm),
            "raman_band": identify_raman_band(x),
            "sym_info": determine_syminfo(symmetry)
        })

    return jsonify({
        "peaks": peak_data,
        "analysis": materials_detected
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)