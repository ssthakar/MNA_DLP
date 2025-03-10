import matplotlib.pyplot as plt
import numpy as np
import re


# Extract data from the provided text
def extract_data_from_log(log_text):
    pattern = r"Iteration : (\d+), Loss: ([\d.]+)"
    matches = re.findall(pattern, log_text)

    iterations = []
    losses = []

    for match in matches:
        iterations.append(int(match[0]))
        losses.append(float(match[1]))

    return iterations, losses


# Read the log data from file (or you can paste it directly here)
with open("loss.text", "r") as f:
    log_text = f.read()

# Extract iterations and losses
iterations, losses = extract_data_from_log(log_text)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, losses, marker="o", linestyle="-", color="b")

# Add annotations
plt.title("Loss vs. Iteration")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# Use logarithmic scale for y-axis due to large range of loss values
plt.yscale("log")

# No learning rate change point annotation

# Customize the plot for better readability
plt.tight_layout()

# Save the plot
plt.savefig("loss_history.png")

# Show the plot
plt.show()

print(
    f"Lowest loss value: {min(losses)} at iteration {iterations[losses.index(min(losses))]}"
)
