import sys
import os
import io
import numpy as np
import pandas as pd
from tkinter import Tk, Label, Button, Entry, filedialog, messagebox, ttk, Frame
from scipy.signal import find_peaks
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RamanPeakAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Raman Spectroscopy Analyzer")
        self.root.geometry("800x600")  # Larger window for more data
        self.root.configure(bg='black')

        # Create and place the threshold label and entry
        self.lblThreshold = Label(root, text="Peak Threshold:", bg='black', fg='cyan', font=('Arial', 12, 'bold'))
        self.lblThreshold.grid(row=0, column=0, padx=10, pady=10, sticky='e')
        
        self.leThreshold = Entry(root, font=('Arial', 12), bg='lightgray')
        self.leThreshold.insert(0, "100") # Default threshold
        self.leThreshold.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        # Create and place buttons
        self.btnLoad = Button(root, text="Load Data", command=self.load_data, bg='lightblue', font=('Arial', 12, 'bold'), width=15)
        self.btnLoad.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        
        self.btnAnalyze = Button(root, text="Analyze Data", command=self.analyze_data, bg='lightgreen', font=('Arial', 12, 'bold'), width=15)
        self.btnAnalyze.grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

        self.btnClear = Button(root, text="Clear Data", command=self.clear_data, bg='lightcoral', font=('Arial', 12, 'bold'), width=15)
        self.btnClear.grid(row=1, column=2, padx=10, pady=10, sticky='nsew')

        # Create and place the Treeview for displaying data
        cols = ("Shift", "Intensity", "FWHM", "Band", "Symmetry")
        self.tree = ttk.Treeview(root, columns=cols, show="headings", height=8)
        self.tree.heading("Shift", text="Raman Shift (cm⁻¹)")
        self.tree.heading("Intensity", text="Intensity (a.u.)")
        self.tree.heading("FWHM", text="FWHM (cm⁻¹)")
        self.tree.heading("Band", text="Assign Band")
        self.tree.heading("Symmetry", text="Symmetry")
        
        # Column formatting
        for col in cols:
            self.tree.column(col, width=120, anchor='center')
        
        self.tree.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

        # Create and place the plot frame
        self.plot_frame = Frame(root, bg='black')
        self.plot_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

        # Configure grid weights
        root.grid_rowconfigure(2, weight=1)
        root.grid_rowconfigure(3, weight=2)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_columnconfigure(2, weight=1)

        # Start fade effect
        self.fade_in()

    def on_hover(self, button, color):
        button.config(bg=color)

    def fade_in(self, alpha=0):
        """Simple fade-in effect for the window."""
        if alpha < 255:
            self.root.attributes('-alpha', alpha / 255)
            self.root.after(10, self.fade_in, alpha + 5)

    def load_data(self):
        file_path = filedialog.askopenfilename(
            title="Open Data File", 
            filetypes=[("Data Files", "*.csv *.txt"), ("CSV Files", "*.csv"), ("Text Files", "*.txt")]
        )
        if file_path:
            try:
                # Use sep=None and engine='python' for automatic delimiter detection
                self.df = pd.read_csv(file_path, header=None, sep=None, engine='python')
                self.raman_shifts = self.df[0].to_numpy()
                self.intensities = self.df[1].to_numpy()
                self.analyze_data()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def analyze_data(self):
        try:
            threshold_input = self.leThreshold.get()
            threshold = float(threshold_input) if threshold_input else 0

            window_length = max(3, int(len(self.intensities) * 0.02))
            self.smoothed_intensities = smooth_data(self.raman_shifts, self.intensities, window_length)
            
            # Use threshold in find_peaks height parameter
            self.peaks, _ = find_peaks(self.smoothed_intensities, height=threshold)
            
            self.update_table()
            self.plot_data()
            self.generate_report()
        except ValueError as ve:
            messagebox.showerror("Error", f"Invalid input for threshold: {str(ve)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze data: {str(e)}")

    def generate_report(self):
        """Perform comprehensive material, layer thickness, and strain analysis."""
        findings = []
        
        # Search for key peaks with distinct ranges
        g_peak = self.find_peak_in_range(1570, 1590)
        d_peak = self.find_peak_in_range(1330, 1360)
        two_d_peak_graphene = self.find_peak_in_range(2680, 2715)
        two_d_peak_graphite = self.find_peak_in_range(2716, 2760)
        
        ws2_e_peak = self.find_peak_in_range(350, 362)
        ws2_a_peak = self.find_peak_in_range(415, 425)
        
        mos2_e_peak = self.find_peak_in_range(380, 390)
        mos2_a_peak = self.find_peak_in_range(400, 412)

        # Graphene / Graphite Analysis
        if g_peak is not None:
            g_pos = self.raman_shifts[g_peak]
            g_int = self.smoothed_intensities[g_peak]
            strain_g = (g_pos - 1581) / -14.0
            
            # Determine if Graphite or Graphene based on 2D peak position and ratio
            if two_d_peak_graphite is not None:
                two_d_int = self.smoothed_intensities[two_d_peak_graphite]
                ratio = two_d_int / g_int
                if ratio < 0.5:
                    findings.append(f"Material: Graphite (Bulk Carbon)\n- G-Peak: {g_pos:.1f} cm⁻¹\n- 2D-Peak: {self.raman_shifts[two_d_peak_graphite]:.1f} cm⁻¹\n- Ratio (I2D/IG): {ratio:.2f}\n- Strain: {strain_g:.2f}%")
            
            if two_d_peak_graphene is not None and not any("Graphite" in f for f in findings):
                ratio = self.intensities[two_d_peak_graphene] / g_int
                if ratio > 2: layer = "Monolayer"
                elif ratio > 1: layer = "Bilayer"
                else: layer = "Multilayer"
                
                res = f"Material: Graphene\n- G-Peak: {g_pos:.1f} cm⁻¹\n- Layer: {layer} (I2D/IG: {ratio:.2f})\n- Strain: {strain_g:.2f}%"
                if d_peak is not None:
                    res += f"\n- Quality: Defects detected (ID/IG: {self.intensities[d_peak]/g_int:.2f})"
                findings.append(res)

        # WS2 Analysis
        if ws2_e_peak is not None and ws2_a_peak is not None:
            e_pos = self.raman_shifts[ws2_e_peak]
            a_pos = self.raman_shifts[ws2_a_peak]
            delta = a_pos - e_pos
            strain_ws2 = (e_pos - 356.5) / -0.66
            layer = "Monolayer" if delta < 65 else "Bulk/Multilayer"
            findings.append(f"Material: WS₂\n- E¹₂g: {e_pos:.1f}, A₁g: {a_pos:.1f} cm⁻¹\n- Layer: {layer} (Δ: {delta:.1f} cm⁻¹)\n- Strain: {strain_ws2:.2f}%")

        # MoS2 Analysis
        if mos2_e_peak is not None and mos2_a_peak is not None:
            e_pos = self.raman_shifts[mos2_e_peak]
            a_pos = self.raman_shifts[mos2_a_peak]
            delta = a_pos - e_pos
            strain_mos2 = (e_pos - 383) / -1.5
            layer = "Monolayer" if delta < 22 else "Bulk/Multilayer"
            findings.append(f"Material: MoS₂\n- E¹₂g: {e_pos:.1f}, A₁g: {a_pos:.1f} cm⁻¹\n- Layer: {layer} (Δ: {delta:.1f} cm⁻¹)\n- Strain: {strain_mos2:.2f}%")

        if findings:
            report_text = "\n\n---\n\n".join(findings)
            messagebox.showinfo("Analysis Report", report_text)
        else:
            messagebox.showwarning("Analysis Report", "No characteristic materials recognized.")

    def find_peak_in_range(self, low, high):
        """Helper to find the strongest peak index within a specific raman shift range."""
        best_peak = None
        max_int = -1
        for p in self.peaks:
            shift = self.raman_shifts[p]
            if low <= shift <= high:
                if self.smoothed_intensities[p] > max_int:
                    max_int = self.smoothed_intensities[p]
                    best_peak = p
        return best_peak

    def plot_data(self):
        """Plot the smoothed intensities against raman shifts in the Tkinter window."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.raman_shifts, self.smoothed_intensities, label='Smoothed Data', color='#1f77b4', linewidth=1.5)
        ax.scatter(self.raman_shifts[self.peaks], self.smoothed_intensities[self.peaks], color='red', s=30, label='Peaks', zorder=5)
        
        ax.set_title('Raman Spectroscopy Analysis', fontsize=14, pad=15)
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
        ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def update_table(self):
        """Update the Treeview to show shift, intensity, FWHM, band, and symmetry."""
        for item in self.tree.get_children():
            self.tree.delete(item)

        for i, peak in enumerate(self.peaks):
            fwhm, left_val, right_val = calculate_fwhm(self.raman_shifts, self.smoothed_intensities, peak)
            shift = self.raman_shifts[peak]
            band = identify_raman_band(shift)
            symmetry = determine_symmetry(left_val, right_val, shift)
            
            self.tree.insert("", "end", values=(
                f"{shift:.1f}", 
                f"{self.smoothed_intensities[peak]:.0f}", 
                f"{fwhm:.1f}", 
                band,
                f"{symmetry:.2f}" if symmetry != float('inf') else "N/A"
            ))

    def clear_data(self):
        """Clear the input fields and reset any data."""
        self.leThreshold.delete(0, 'end')
        self.tree.delete(*self.tree.get_children())
        self.intensities = None
        self.raman_shifts = None
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        messagebox.showinfo("Info", "Data cleared successfully!")

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

def determine_symmetry(left_val, right_val, x):
    if right_val is None or x is None:
        return float('inf')
    if abs(right_val - x) < 1e-10:
        return float('inf')
    s = (left_val - x) / (right_val - x)
    s = round(s, 1)
    return abs(s)

def determine_syminfo(symmetry):
    if 0.9 <= symmetry <= 1.1:
        return "Single Layer Graphene"
    else:
        return "Multi Layer Graphene"

if __name__ == "__main__":
    root = Tk()
    app = RamanPeakAnalyzer(root)
    root.mainloop()
