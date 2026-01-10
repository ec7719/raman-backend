
import numpy as np
from scipy import signal

# --- MOCKED FUNCTIONS FROM IMPLEMENTATION ---

def identify_raman_band(x, valid_materials=None):
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

def test_logic(peaks, shifts):
    """
    Simulates the logic flow implemented in the main files.
    """
    detected_materials = []
    
    def find_p(low, high):
        # Simplified find_p just checking existence in peaks list for this test
        for p_shift in peaks:
            if low <= p_shift <= high:
                return p_shift
        return None

    # 1. Determine Dominant Material
    g_p = find_p(1570, 1590)
    
    if g_p:
         detected_materials = ["Carbon"]
    else:
        ws2_e = find_p(350, 362)
        mos2_e = find_p(380, 390)
        if ws2_e: detected_materials.append("WS2")
        if mos2_e: detected_materials.append("MoS2")
        
    print(f"Detected Materials: {detected_materials}")
    
    # 2. Test Band ID for a specific "noise" peak
    # Let's say we have a peak at 383 (MoS2 E2g)
    test_peak = 383
    band_id = identify_raman_band(test_peak, valid_materials=detected_materials)
    print(f"Peak at {test_peak} identified as: '{band_id}'")
    
    return detected_materials, band_id

# --- TEST CASES ---

print("--- TEST 1: Pure Graphene (G + 2D) ---")
# Peaks: G (1580), 2D (2700)
test_logic([1580, 2700], None)
# Expect: ["Carbon"], Unknown (for the 383 check)

print("\n--- TEST 2: Graphene with MoS2 Noise ---")
# Peaks: G (1580), 2D (2700), MoS2 E2g (383 - noise)
test_logic([1580, 2700, 383], None)
# Expect: ["Carbon"], Unknown (Crucial test! Should NOT be MoS2)

print("\n--- TEST 3: Pure MoS2 ---")
# Peaks: E2g (383), A1g (405)
test_logic([383, 405], None)
# Expect: ["MoS2"], E¹₂g (MoS2)

print("\n--- TEST 4: Pure WS2 ---")
# Peaks: E2g (355), A1g (420)
test_logic([355, 420], None)
# Expect: ["WS2"], Unknown (because 383 is MoS2, not WS2)
