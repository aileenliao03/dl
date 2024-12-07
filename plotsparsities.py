import pandas as pd
import matplotlib.pyplot as plt
import csv

# File paths to your CSVs
sparsity_file_duo_gate = 'trained_model_duo_gate_diff/train_sparsities.csv'
sparsity_file_no_mask = 'trained_model_no_mask_duo_gate_diff/train_sparsities.csv'

# Read Sparsities for trained_model_duo_gate_diff
epochs_sparsity_duo_gate = []
sparsity_values_duo_gate = []
with open(sparsity_file_duo_gate, mode='r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        epochs_sparsity_duo_gate.append(int(row[0]))
        sparsity_values_duo_gate.append(float(row[1]))

# Read Sparsities for trained_model_no_mask_duo_gate_diff
epochs_sparsity_no_mask = []
sparsity_values_no_mask = []
with open(sparsity_file_no_mask, mode='r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        epochs_sparsity_no_mask.append(int(row[0]))
        sparsity_values_no_mask.append(float(row[1]))

# Create a plot for Sparsity comparison
plt.figure(figsize=(12, 6))

# Plot Sparsity for trained_model_duo_gate_diff
plt.plot(epochs_sparsity_duo_gate, sparsity_values_duo_gate, label="With Gated Attention", color="blue", marker="o")

# Plot Sparsity for trained_model_no_mask_duo_gate_diff
plt.plot(epochs_sparsity_no_mask, sparsity_values_no_mask, label="No Mask", color="orange", marker="x")

# Adding labels, title, and legend
plt.xlabel("Epoch")
plt.ylabel("Sparsity")
plt.title("Comparison of Sparsity Across Epochs")
plt.grid(True)
plt.legend()

# Save the figure
plt.tight_layout()
plt.savefig("./trained_model_comparison_sparsity_plot.png")
plt.show()
