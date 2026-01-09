import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import signal
import plotly.graph_objects as go
from io import StringIO
import google.generativeai as genai
import datetime

# --- GEMINI SETUP ---
GEMINI_API_KEY = "AIzaSyBOW46Vs8PhbfEMt89P3kuUh5LhHrv7u4k"
genai.configure(api_key=GEMINI_API_KEY)
# Reverting to a known stable model; change to 'gemini-2.0-flash-exp' if latest features are needed.
model = genai.GenerativeModel('gemini-1.5-flash')

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gemini Raman Analyzer", layout="wide", page_icon="🔬")

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .material-card {
        padding: 1.2rem;
        border-radius: 0.8rem;
        border-left: 6px solid #00d2ff;
        background-color: #1a1c23;
        margin-bottom: 1rem;
        color: #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .chat-container {
        border: 1px solid #30363d;
        border-radius: 0.5rem;
        padding: 1rem;
        height: 400px;
        overflow-y: auto;
        background-color: #0d1117;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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

def identify_raman_band(x):
    if 1330 <= x <= 1360: return "D Band (C)"
    if 1570 <= x <= 1590: return "G Band (C)"
    if 2680 <= x <= 2715: return "2D Band (Graphene)"
    if 2716 <= x <= 2760: return "2D Band (Graphite)"
    if 350 <= x <= 362: return "E¹₂g (WS₂)"
    if 415 <= x <= 425: return "A₁g (WS₂)"
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

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 Settings & Upload")
    uploaded_file = st.file_uploader("Upload Data (CSV/TXT)", type=["csv", "txt"])
    
    st.divider()
    st.subheader("Analysis Parameters")
    threshold = st.number_input("Peak Threshold", value=100.0, step=10.0, help="Minimum intensity for peak detection.")
    smooth_win_pc = 2.0 # Fixed at 2% as requested previously
    
    st.divider()
    st.info("Compatible Materials: **Graphite, Graphene, WS₂, MoS₂**")
    
    if st.button("Reset Analysis"):
        st.session_state.chat_history = []
        st.rerun()

# --- MAIN UI ---
st.title("Interactive Raman Analyzer with Gemini AI")

# Initialize Chat State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    try:
        # Load Data
        df = pd.read_csv(uploaded_file, header=None, sep=None, engine='python')
        shifts = df[0].to_numpy()
        intensities = df[1].to_numpy()
        
        # Process Data
        win_size = max(3, int(len(intensities) * (smooth_win_pc / 100.0)))
        smoothed = smooth_data(intensities, win_size)
        peaks_idx, _ = find_peaks(smoothed, height=threshold)
        
        # --- INTERACTIVE PLOTLY CHART ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=shifts, y=intensities, mode='lines', name='Raw Data', line=dict(color='rgba(100,100,100,0.3)', width=1)))
        fig.add_trace(go.Scatter(x=shifts, y=smoothed, mode='lines', name='Smoothed', line=dict(color='#00d2ff', width=2)))
        
        if len(peaks_idx) > 0:
            fig.add_trace(go.Scatter(
                x=shifts[peaks_idx], y=smoothed[peaks_idx], mode='markers', name='Peaks',
                marker=dict(color='red', size=8, symbol='circle-open'),
                hovertemplate="Shift: %{x:.1f} cm⁻¹<br>Int: %{y:.0f}<extra></extra>"
            ))

        fig.update_layout(
            template="plotly_dark", xaxis_title="Raman Shift (cm⁻¹)", yaxis_title="Intensity (a.u.)",
            hovermode="x unified", height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- ANALYSIS & CHAT LAYOUT ---
        col_analysis, col_chat = st.columns([1, 1])
        
        findings_to_share = []
        
        with col_analysis:
            st.subheader("⚛️ Scientific Analysis")
            
            def find_p(low, high):
                best = None; m_int = -1
                for p in peaks_idx:
                    if low <= shifts[p] <= high:
                        if smoothed[p] > m_int:
                            m_int = smoothed[p]; best = p
                return best

            findings_md = []
            
            # Carbon logic
            g_p = find_p(1570, 1590)
            if g_p:
                g_pos = shifts[g_p]; g_int = smoothed[g_p]
                strain = (g_pos - 1581) / -14.0
                dg_p = find_p(2716, 2760)
                gn_p = find_p(2680, 2715)
                
                if dg_p and (smoothed[dg_p]/g_int < 0.5):
                    f = f"**Material: Graphite (Bulk Carbon)**\n- G-Peak: {g_pos:.1f}, 2D-Peak: {shifts[dg_p]:.1f} cm⁻¹\n- Ratio (I2D/IG): {smoothed[dg_p]/g_int:.2f}\n- Strain: {strain:.2f}%"
                    findings_md.append(f)
                elif gn_p:
                    ratio = smoothed[gn_p] / g_int
                    layer = "Monolayer" if ratio > 2 else "Bilayer" if ratio > 1 else "Multilayer"
                    f = f"**Material: Graphene**\n- G-Peak: {g_pos:.1f} cm⁻¹\n- Layer: {layer} (I2D/IG: {ratio:.2f})\n- Strain: {strain:.2f}%"
                    d_p = find_p(1330, 1360)
                    if d_p: f += f"\n- Quality: Defects detected (ID/IG: {smoothed[d_p]/g_int:.2f})"
                    findings_md.append(f)
            
            # WS2 & MoS2 logic
            ws2_e = find_p(350, 362); ws2_a = find_p(415, 425)
            if ws2_e and ws2_a:
                e_pos, a_pos = shifts[ws2_e], shifts[ws2_a]
                delta = a_pos - e_pos
                findings_md.append(f"**Material: WS₂**\n- E¹₂g: {e_pos:.1f}, A₁g: {a_pos:.1f} cm⁻¹\n- Layer: {'Monolayer' if delta < 65 else 'Bulk'} (Δ: {delta:.1f})\n- Strain: {(e_pos - 356.5) / -0.66:.2f}%")

            mos2_e = find_p(380, 390); mos2_a = find_p(400, 412)
            if mos2_e and mos2_a:
                e_pos, a_pos = shifts[mos2_e], shifts[mos2_a]
                delta = a_pos - e_pos
                findings_md.append(f"**Material: MoS₂**\n- E¹₂g: {e_pos:.1f}, A₁g: {a_pos:.1f} cm⁻¹\n- Layer: {'Monolayer' if delta < 22 else 'Bulk'} (Δ: {delta:.1f})\n- Strain: {(e_pos - 383) / -1.5:.2f}%")

            if findings_md:
                for f in findings_md:
                    st.markdown(f'<div class="material-card">{f}</div>', unsafe_allow_html=True)
                    findings_to_share.append(f)
            else:
                st.warning("No characteristic peaks detected for known materials.")

            # Peak Data Preview
            st.caption("Detailed Peak Map")
            p_list = []
            for i in peaks_idx:
                fwhm, l_val, r_val = calculate_fwhm(shifts, smoothed, i)
                sym = determine_symmetry(l_val, r_val, shifts[i])
                p_list.append({
                    "Shift": round(shifts[i], 1),
                    "Intensity": int(smoothed[i]),
                    "FWHM": round(fwhm, 1),
                    "Band": identify_raman_band(shifts[i]),
                    "Sym": sym if sym != float('inf') else "N/A",
                    "Info": determine_syminfo(sym) if sym != float('inf') else "N/A"
                })
            st.dataframe(pd.DataFrame(p_list), use_container_width=True, height=250)

        # --- GEMINI CHAT WINDOW ---
        with col_chat:
            st.subheader("💬 Chat with Gemini AI")
            
            chat_placeholder = st.empty()
            with chat_placeholder.container():
                for msg in st.session_state.chat_history:
                    role = "user" if msg["role"] == "user" else "assistant"
                    with st.chat_message(role):
                        st.write(msg["content"])

            user_input = st.chat_input("Ask about the spectrum (e.g., 'Is my graphene high quality?')")
            
            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.chat_message("user"): st.write(user_input)
                
                # Context preparation
                context = f"Scientific Findings: {'; '.join(findings_to_share)}. Peak Data: {p_list}. User is asking about a Raman Spectroscopy spectrum."
                
                try:
                    full_prompt = f"Context: {context}\n\nUser Question: {user_input}\n\nRespond as a Raman Spectroscopy expert. Be concise and scientifically accurate."
                    response = model.generate_content(full_prompt)
                    ai_reply = response.text
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
                    with st.chat_message("assistant"): st.write(ai_reply)
                except Exception as e:
                    st.error(f"Gemini Error: {e}")

        # --- DOWNLOAD REPORT ---
        st.divider()
        st.subheader("📄 Generate Final Report")
        col_rep1, col_rep2 = st.columns([2, 1])
        
        with col_rep1:
            st.info("The report combines scientific analysis, detected peaks, and your AI chat history.")
        
        with col_rep2:
            report_content = f"""# Raman Spectroscopy Analysis Report
Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Scientific Summary
{chr(10).join(['- ' + f.replace('**', '') for f in findings_to_share]) if findings_to_share else 'No characteristic materials identified.'}

## 2. Detected Peaks
{pd.DataFrame(p_list).to_markdown() if p_list else 'No peaks detected.'}

## 3. AI Insights & Discussion
{chr(10).join([f"**{m['role'].upper()}**: {m['content']}" for m in st.session_state.chat_history]) if st.session_state.chat_history else 'No chat history.'}

---
*Generated by Raman Analyzer with Google Gemini.*
"""
            st.download_button(
                label="📥 Download Complete Report",
                data=report_content,
                file_name=f"Raman_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

    except Exception as e:
        st.error(f"Error: {e}")
        st.error("Please ensure the CSV file has two columns: [Raman Shift, Intensity]")
else:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Raman_spectrum_of_graphene.png/800px-Raman_spectrum_of_graphene.png", caption="Example: Monolayer Graphene", width=500)
    st.info("👋 Welcome! Upload a .csv or .txt file in the sidebar to begin AI-powered analysis.")
