import sys
import os
import io
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5 import uic
from scipy.signal import find_peaks
from scipy import signal

class RamanPeakAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('raman_peak_analyzer.ui', self)
        self.setWindowTitle("Raman Peak Analyzer")

        self.btnLoadData.clicked.connect(self.load_data)
        self.btnAnalyze.clicked.connect(self.analyze_data)

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Data File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    file_content = file.read()
                    self.df = pd.read_csv(io.StringIO(file_content), header=None)
                    self.wavelengths = self.df[0].to_numpy()
                    self.intensities = self.df[1].to_numpy()
                    self.update_table()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def analyze_data(self):
        try:
            threshold = float(self.leThreshold.text())
            self.intensities = np.where(self.intensities < threshold, 0, self.intensities)
            window_length = max(3, int(len(self.intensities) * 0.02))
            self.smoothed_intensities = smooth_data(self.wavelengths, self.intensities, window_length)
            self.peaks, _ = find_peaks(self.smoothed_intensities, height=0)
            self.update_table()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to analyze data: {str(e)}")

    def update_table(self):
        self.tblResults.setRowCount(len(self.peaks))
        for i, peak in enumerate(self.peaks):
            fwhm, left_val, right_val = calculate_fwhm(self.wavelengths, self.smoothed_intensities, peak)
            x = self.wavelengths[peak]
            symmetry = determine_symmetry(left_val, right_val, x)
            self.tblResults.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.tblResults.setItem(i, 1, QTableWidgetItem(f"{x:.2f}"))
            self.tblResults.setItem(i, 2, QTableWidgetItem(f"{self.smoothed_intensities[peak]:.2f}"))
            self.tblResults.setItem(i, 3, QTableWidgetItem(f"{right_val - x:.2f}"))
            self.tblResults.setItem(i, 4, QTableWidgetItem(f"{x - left_val:.2f}"))
            self.tblResults.setItem(i, 5, QTableWidgetItem(identify_raman_band(x)))
            self.tblResults.setItem(i, 6, QTableWidgetItem(f"{left_val:.2f}"))
            self.tblResults.setItem(i, 7, QTableWidgetItem(f"{right_val:.2f}"))
            self.tblResults.setItem(i, 8, QTableWidgetItem(f"{symmetry:.2f}"))
            self.tblResults.setItem(i, 9, QTableWidgetItem(f"{fwhm:.2f}"))
            self.tblResults.setItem(i, 10, QTableWidgetItem(determine_syminfo(symmetry)))

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
    if 1570 <= x <= 1590:
        return "G Band"
    elif 2600 <= x <= 2800:
        return "2D Band"
    else:
        return "<NA>"

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
    app = QApplication(sys.argv)
    window = RamanPeakAnalyzer()
    window.show()
    sys.exit(app.exec_())