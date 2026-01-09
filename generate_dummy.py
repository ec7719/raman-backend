import numpy as np
import pandas as pd

def generate_dummy_data():
    x = np.linspace(0, 3000, 3000)
    y = np.random.normal(0, 10, 3000)
    
    # Graphene peaks (Monolayer)
    # G-peak 1581, 2D-peak 2700 (I2D/IG approx 2.5)
    y += 500 * np.exp(-((x - 1581)**2) / 400)
    y += 1250 * np.exp(-((x - 2700)**2) / 900)
    
    # WS2 peaks (Monolayer)
    # E12g 356.5, A1g 418.5 (Delta = 62)
    y += 800 * np.exp(-((x - 356.5)**2) / 100)
    y += 600 * np.exp(-((x - 418.5)**2) / 100)
    
    # MoS2 peaks (Bulk)
    # E12g 383, A1g 408 (Delta = 25)
    y += 700 * np.exp(-((x - 383)**2) / 100)
    y += 900 * np.exp(-((x - 408)**2) / 100)
    
    df = pd.DataFrame({0: x, 1: y})
    df.to_csv('raman_dummy_data.csv', index=False, header=False)
    print("Dummy data generated: raman_dummy_data.csv")

if __name__ == "__main__":
    generate_dummy_data()
