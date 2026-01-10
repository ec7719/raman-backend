from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import signal
import google.generativeai as genai
import datetime
from io import StringIO, BytesIO
import os 
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
# --- GEMINI SETUP ---
# Primary key from working React app, fallback from Streamlit
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def get_model(api_key):
    genai.configure(api_key=api_key)
    # Reverting to 1.5-flash because 2.5-flash is currently causing initialization crashes and server downtime.
    selected_model = 'gemini-2.5-flash'
    print(f"Server initializing with STABLE model: {selected_model}")
    return genai.GenerativeModel(selected_model)

model = get_model(GEMINI_API_KEY)
# --- UTILS ---
def smooth_data(y, window_length):
    if window_length < 3: window_length = 3
    if window_length % 2 == 0: window_length += 1
    return signal.savgol_filter(y, window_length, 2)

def calculate_fwhm(x, y, peak):
    half_max = y[peak] / 2.0
    left_idx = np.where(y[:peak] < half_max)[0]
    right_idx = np.where(y[peak:] < half_max)[0] + peak
    left_idx = left_idx[-1] if left_idx.size > 0 else 0
    right_idx = right_idx[0] if right_idx.size > 0 else len(y) - 1
    fwhm = x[right_idx] - x[left_idx]
    return fwhm, x[left_idx], x[right_idx]

def identify_raman_band(x, valid_materials=None):
    # valid_materials: list of strings, e.g. ["Carbon"], ["MoS2"], ["WS2"], or None (for all)
    
    if valid_materials is None or "Carbon" in valid_materials:
        if 1330 <= x <= 1360: return "D Band (C)"
        if 1570 <= x <= 1590: return "G Band (C)"
        if 2680 <= x <= 2715: return "2D Band (Graphene)"
        if 2716 <= x <= 2760: return "2D Band (Graphite)"
    
    if valid_materials is None or "WS2" in valid_materials:
        if 350 <= x <= 362: return "E¹₂g (WS₂)"
        if 415 <= x <= 425: return "A₁g (WS₂)"
    
    if valid_materials is None or "MoS2" in valid_materials:
        if 380 <= x <= 390: return "E¹₂g (MoS₂)"
        if 400 <= x <= 412: return "A₁g (MoS₂)"
        
    return "Unknown"

def determine_symmetry(left_val, right_val, x):
    if right_val is None or x is None or abs(right_val - x) < 1e-10:
        return float('inf')
    s = (left_val - x) / (right_val - x)
    return abs(round(s, 2))

def determine_syminfo(symmetry):
    if 0.9 <= symmetry <= 1.1:
        return "Symmetric (likely 1L)"
    else:
        return "Asymmetric (likely ML)"

@app.route('/')
def home():
    return "Raman Analysis API is Running"

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    threshold = float(request.form.get('threshold', 100.0))
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        df = pd.read_csv(file, header=None, sep=None, engine='python')
        shifts = df[0].to_numpy()
        intensities = df[1].to_numpy()
        
        # Fixed smoothing window as per user request (2%)
        win_size = max(3, int(len(intensities) * (2.0 / 100.0)))
        smoothed = smooth_data(intensities, win_size)
        peaks_idx, _ = find_peaks(smoothed, height=threshold)
        
        # Analysis logic
        def find_p(low, high):
            best = None; m_int = -1
            for p in peaks_idx:
                if low <= shifts[p] <= high:
                    if smoothed[p] > m_int:
                        m_int = smoothed[p]; best = p
            return best

        findings_to_share = []
        detected_materials = []

        # 1. Determine Dominant Material First
        # Check for Carbon (G Peak is the strongest indicator)
        g_p = find_p(1570, 1590)
        
        if g_p is not None:
             detected_materials = ["Carbon"]
        else:
            # If no Carbon, check for MoS2 / WS2 interaction
            # We can check specific peaks to see which is present
            ws2_e = find_p(350, 362)
            mos2_e = find_p(380, 390)
            
            if ws2_e is not None:
                detected_materials.append("WS2")
            if mos2_e is not None:
                detected_materials.append("MoS2")
        
        # If nothing specific detected, we might want to allow all or none. 
        # But per requirements, "no one materials peaks should be identified in another".
        # If detected_materials is empty, we leave it empty (Unknown).

        # 2. Generate Findings based ONLY on detected materials
        
        # Carbon Logic
        if "Carbon" in detected_materials:
            g_pos = shifts[g_p]; g_int = smoothed[g_p]
            strain = (g_pos - 1581) / -14.0
            dg_p = find_p(2716, 2760)
            gn_p = find_p(2680, 2715)
            
            if dg_p is not None and (smoothed[dg_p]/g_int < 0.5):
                f = f"Material: Graphite (Bulk Carbon). G-Peak: {g_pos:.1f}, 2D-Peak: {shifts[dg_p]:.1f} cm⁻¹. Ratio (I2D/IG): {smoothed[dg_p]/g_int:.2f}. Strain: {strain:.2f}%"
                findings_to_share.append(f)
            elif gn_p is not None:
                ratio = smoothed[gn_p] / g_int
                layer = "Monolayer" if ratio > 2 else "Bilayer" if ratio > 1 else "Multilayer"
                f = f"Material: Graphene. G-Peak: {g_pos:.1f}, 2D-Peak: {shifts[gn_p]:.1f} cm⁻¹. Layer: {layer} (I2D/IG: {ratio:.2f}). Strain: {strain:.2f}%"
                d_p = find_p(1330, 1360)
                if d_p is not None: f += f". Quality: Defects detected (ID/IG: {smoothed[d_p]/g_int:.2f}, D-Peak: {shifts[d_p]:.1f} cm⁻¹)"
                findings_to_share.append(f)

        # WS2 Logic
        if "WS2" in detected_materials:
            ws2_e = find_p(350, 362); ws2_a = find_p(415, 425)
            if ws2_e is not None and ws2_a is not None:
                e_pos, a_pos = shifts[ws2_e], shifts[ws2_a]
                delta = a_pos - e_pos
                f = f"Material: WS2. E2g: {e_pos:.1f}, A1g: {a_pos:.1f} cm-1. Layer: {'Monolayer' if delta < 65 else 'Bulk'} (Delta: {delta:.1f}). Strain: {(e_pos - 356.5) / -0.66:.2f}%"
                findings_to_share.append(f)

        # MoS2 Logic
        if "MoS2" in detected_materials:
            mos2_e = find_p(380, 390); mos2_a = find_p(400, 412)
            if mos2_e is not None and mos2_a is not None:
                e_pos, a_pos = shifts[mos2_e], shifts[mos2_a]
                delta = a_pos - e_pos
                f = f"Material: MoS2. E2g: {e_pos:.1f}, A1g: {a_pos:.1f} cm-1. Layer: {'Monolayer' if delta < 22 else 'Bulk'} (Delta: {delta:.1f}). Strain: {(e_pos - 383) / -1.5:.2f}%"
                findings_to_share.append(f)

        # Global Symmetry Calculation
        global_sym = "N/A"
        global_info = "No peaks detected"
        if len(peaks_idx) > 0:
            # Calculate based on the highest peak
            main_peak = peaks_idx[np.argmax(smoothed[peaks_idx])]
            fwhm, l_val, r_val = calculate_fwhm(shifts, smoothed, main_peak)
            sym_val = determine_symmetry(l_val, r_val, shifts[main_peak])
            if sym_val != float('inf'):
                global_sym = sym_val
                global_info = determine_syminfo(sym_val)

        # Peak Data (Removed Sym and Info from per-peak)
        p_list = []
        for i in peaks_idx:
            fwhm, l_val, r_val = calculate_fwhm(shifts, smoothed, i)
            p_list.append({
                "Shift": round(shifts[i], 1),
                "Intensity": int(smoothed[i]),
                "FWHM": round(fwhm, 1),
                "Band": identify_raman_band(shifts[i], valid_materials=detected_materials),
                "xl": l_val,
                "xr": r_val
            })

        return jsonify({
            "shifts": shifts.tolist(),
            "intensities": intensities.tolist(),
            "smoothed": smoothed.tolist(),
            "peaks": p_list,
            "findings": findings_to_share,
            "global_symmetry": global_sym,
            "global_info": global_info
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        user_input = data.get('message')
        history = data.get('history', [])
        findings = data.get('findings', [])
        peaks = data.get('peaks', [])

        findings_str = "; ".join([str(f) for f in findings]) if isinstance(findings, list) else str(findings)
        peaks_str = str(peaks)

        context = f"Scientific Findings: {findings_str}. Peak Data: {peaks_str}. User is asking about a Raman Spectroscopy spectrum."
        full_prompt = f"Context: {context}\n\nUser Question: {user_input}\n\nRespond as a Raman Spectroscopy expert. Be concise and scientifically accurate."
        
        try:
            response = model.generate_content(full_prompt)
            ai_reply = response.text
            return jsonify({"reply": ai_reply})
        except Exception as e:
            return jsonify({"error": f"Gemini API Error: {str(e)}"}), 500

    except Exception as e:
        print(f"CRITICAL ERROR in /chat: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route('/report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        findings = data.get('findings', [])
        peaks = data.get('peaks', [])
        history = data.get('history', [])
        global_sym = data.get('global_symmetry', 'N/A')
        global_info = data.get('global_info', 'N/A')
        
        findings_str = "; ".join([str(f) for f in findings])
        peaks_str = "\n".join([f"Peak {i+1}: Shift={p['Shift']}cm-1, Int={p['Intensity']}, FWHM={p['FWHM']}, Band={p['Band']}" for i, p in enumerate(peaks)])
        chat_str = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history])
        
        prompt = f"""
        Generate a Formal Technical Raman Spectroscopy Analysis Report.
        
        INPUT DATA:
        Scientific Findings: {findings_str}
        Detected Peaks:
        {peaks_str}
        Global Symmetry: {global_sym} ({global_info})
        
        USER INTERACTION SUMMARY:
        {chat_str}
        
        REQUIREMENTS:
        1. Start with a "Technical Executive Summary" summarizing the material and its state.
        2. Provide a "Spectral Feature Analysis" section discussing the observed peaks and the overall data symmetry.
        3. Include an "Interaction Synthesis" section that summarizes the key technical points discussed during the AI chat.
        4. Use a formal, objective, and expert tone.
        5. Describe the plot characteristics as a "Visual Data Representation Summary".
        
        Format the response in clear Markdown.
        """
        
        try:
            response = model.generate_content(prompt)
            formal_summary = response.text
        except Exception as e:
            formal_summary = f"Error generating automated summary: {str(e)}"

        report_content = f"""# RAMAN SPECTROSCOPY FORMAL ANALYSIS REPORT
Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{formal_summary}

---
## DATA APPENDIX

### Global Data Symmetry
- **Symmetry Ratio:** {global_sym}
- **Interpretation:** {global_info}

### Detected Peak Table
| Rank | Shift (cm⁻¹) | Intensity (a.u.) | FWHM | Assigned Band |
|------|--------------|-------------------|------|---------------|
"""
        for i, p in enumerate(peaks):
            report_content += f"| {i+1} | {p['Shift']} | {p['Intensity']} | {p['FWHM']} | {p['Band']} |\n"

        report_content += f"""
### Raw Scientific Findings
{chr(10).join(['- ' + f for f in findings]) if findings else 'No characteristic materials identified.'}

### Communication Log
"""
        for m in history:
            report_content += f"**{m['role'].upper()}**: {m['content']}\n\n"

        report_content += "\n---\n*Report generated by Raman-Data-Analysis System with Gemini AI Integration.*"
        
        return jsonify({"report": report_content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
