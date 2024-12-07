import pandas as pd
import matplotlib.pyplot as plt
import csv

# File paths to your CSVs
sparsity_file = 'trained_model_duo_gate_diff/train_sparsities.csv'
gate_stats_file = 'trained_model_duo_gate_diff/train_gate_stats.csv'

# Read Sparsities from CSV
epochs_sparsity = []
sparsity_values = []
with open(sparsity_file, mode='r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        epochs_sparsity.append(int(row[0]))
        sparsity_values.append(float(row[1]))

# Read Gate Stats (Mean and Std) from CSV
epochs_gate = []
gate_means = []
gate_stds = []
with open(gate_stats_file, mode='r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        epochs_gate.append(int(row[0]))
        gate_means.append(float(row[1]))
        gate_stds.append(float(row[2]))

# Create a plot for Sparsity, Gate Mean, and Gate Std
plt.figure(figsize=(12, 8))

# Plot Sparsity
plt.subplot(3, 1, 1)
plt.plot(epochs_sparsity, sparsity_values, label="Sparsity", color="blue", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Sparsity")
plt.title("Train Sparsity Over Epochs")
plt.grid(True)
plt.legend()

# Plot Gate Mean
plt.subplot(3, 1, 2)
plt.plot(epochs_gate, gate_means, label="Gate Mean", color="green", marker="x")
plt.xlabel("Epoch")
plt.ylabel("Gate Mean")
plt.title("Train Gate Mean Over Epochs")
plt.grid(True)
plt.legend()

# Plot Gate Std
plt.subplot(3, 1, 3)
plt.plot(epochs_gate, gate_stds, label="Gate Std", color="red", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Gate Std")
plt.title("Train Gate Std Over Epochs")
plt.grid(True)
plt.legend()

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the figure
plt.savefig("./trained_model_duo_gate_diff/train_gate_sparsity_plot.png")
plt.show()