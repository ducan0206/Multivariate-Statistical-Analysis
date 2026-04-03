import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .utils import mean

def distribution(df, column_name):
    """
    Plots the distribution of a column using a histogram and KDE.
    Does not call plt.show(), allowing use in subplots.
    """
    sns.histplot(df[column_name], kde=True, color="steelblue")
    plt.title(f"Distribution of {column_name}", fontsize=10, fontweight="bold")
    plt.xlabel(column_name, fontsize=8)
    plt.ylabel("Frequency", fontsize=8)
    plt.grid(True, alpha=0.3)

def dot_diagram(df, column_name):
    """
    Plots a dot diagram (1D scatter plot) for a single variable.
    Does not call plt.show(), allowing use in subplots.
    """
    plt.scatter(df[column_name], np.zeros_like(df[column_name]), alpha=0.5, color="salmon", s=20)
    plt.title(f"Dot Diagram of {column_name}", fontsize=10, fontweight="bold")
    plt.xlabel(column_name, fontsize=8)
    plt.yticks([])  # Hide y-axis for 1D plot
    plt.grid(True, axis='x', alpha=0.3)

def scatter_plot(df, col1, col2):
    """
    Plots a scatter plot of two variables and highlights the mean vector.
    """
    x = df[col1]
    y = df[col2]
    x_bar = mean(x)
    y_bar = mean(y)

    plt.scatter(x, y, alpha=0.4, s=25, color="steelblue", label="Data points")
    
    # Highlight Mean Vector
    plt.scatter([x_bar], [y_bar], s=200, color="red", zorder=5, marker="*",
               label=f"Mean vector (μ₁={x_bar:.2f}, μ₂={y_bar:.2f})")
    
    # Draw dashed lines for mean
    plt.axvline(x_bar, color="red", lw=1, ls="--", alpha=0.6)
    plt.axhline(y_bar, color="red", lw=1, ls="--", alpha=0.6)
    
    plt.xlabel(col1.capitalize(), fontsize=12)
    plt.ylabel(col2.capitalize(), fontsize=12)
    plt.title(f"Scatter Plot: {col1} vs {col2}", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
