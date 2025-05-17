import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlarena.utils.plot_utils import plot_box_scatter

# Set random seed for reproducibility
np.random.seed(42)

# Create a large synthetic dataset
n_points = 10000  # Large number of points
n_categories = 5  # Number of categories

# Generate data
categories = [f"Category {i+1}" for i in range(n_categories)]
data = {
    "category": np.random.choice(categories, size=n_points),
    "value": np.random.normal(loc=100, scale=15, size=n_points),
}

# Add two types of point_hue for testing
# 1. Binary numerical hue (e.g., like a threshold or binary classifier output)
data["binary_hue"] = np.random.choice([0, 1], size=n_points, p=[0.7, 0.3])

# 2. Categorical hue (e.g., like different sources or types)
data["categorical_hue"] = np.random.choice(
    ["Type A", "Type B", "Type C"], size=n_points, p=[0.5, 0.3, 0.2]
)

# Create DataFrame
df = pd.DataFrame(data)

# Test with binary numerical hue
plt.figure(1)
fig1, ax1 = plot_box_scatter(
    data=df,
    x="category",
    y="value",
    point_hue="binary_hue",
    title="Box Plot with Binary Numerical Hue",
)

# Test with categorical hue
plt.figure(2)
fig2, ax2 = plot_box_scatter(
    data=df,
    x="category",
    y="value",
    point_hue="categorical_hue",
    title="Box Plot with Categorical Hue",
)

plt.show()
