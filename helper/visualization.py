import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .utils import mean

def distribution(df, column_name):
    """
    V? bi?u ?? phân ph?i c?a m?t c?t d? li?u k?t h?p Histogram và ???ng KDE.

    Input:
        - df: DataFrame ch?a d? li?u.
        - column_name: Tên c?t c?n v? phân ph?i.
    Output:
        - Hi?n th? bi?u ?? phân ph?i v?i Seaborn.
    """
    # S? d?ng histplot ?? v? ??ng th?i c?t t?n su?t và ???ng cong m?t ?? (KDE)
    sns.histplot(df[column_name], kde=True, color="steelblue")
    plt.title(f"Phân ph?i c?a {column_name}", fontsize=10, fontweight="bold")
    plt.xlabel(column_name, fontsize=8)
    plt.ylabel("T?n su?t (Frequency)", fontsize=8)
    plt.grid(True, alpha=0.3)

def dot_diagram(df, column_name):
    """
    V? bi?u ?? ch?m (Dot Diagram) 1 chi?u ?? quan sát m?t ?? t?p trung c?a d? li?u.

    Input:
        - df: DataFrame ch?a d? li?u.
        - column_name: Tên c?t c?n v?.
    Output:
        - Hi?n th? bi?u ?? scatter plot 1D (tr?c Y c? ??nh b?ng 0).
    """
    
    # T?o m?ng s? 0 có cùng kích th??c ?? tr?i d? li?u trên m?t ???ng th?ng ngang
    plt.scatter(df[column_name], np.zeros_like(df[column_name]), alpha=0.5, color="salmon", s=1)
    plt.title(f"Bi?u ?? ch?m c?a {column_name}", fontsize=10, fontweight="bold")
    plt.xlabel(column_name, fontsize=8)
    plt.yticks([])  # ?n tr?c Y vì ?ây là bi?u ?? 1 chi?u
    plt.grid(True, axis='x', alpha=0.3)

def lln_convergence(ax, sizes, errors, column_name, color="steelblue"):
    """
    V? ?? th? h?i t? theo Lu?t s? l?n (LLN) - sai s? t??ng ??i gi?m khi kích th??c m?u t?ng.

    Input:
        - ax: ??i t??ng tr?c (axes) c?a matplotlib ?? v? lên.
        - sizes: Danh sách các kích th??c m?u (n).
        - errors: Danh sách sai s? t??ng ??i t??ng ?ng.
        - column_name: Tên bi?n ?ang xét.
        - color: Màu s?c c?a ???ng v?.
    Output:
        - V? bi?u ?? h?i t? lên ax ???c cung c?p.
    """
    ax.plot(sizes, errors, color=color, lw=1.2, alpha=0.85, label=column_name)
    ax.axhline(0, color='black', lw=0.8) # ???ng tham chi?u sai s? b?ng 0
    ax.fill_between(sizes, errors, alpha=0.15, color=color) # Tô màu vùng d??i ???ng cong
    ax.set_title(f'LLN: {column_name}', fontsize=10, fontweight="bold")
    ax.set_xlabel('Kích th??c m?u ($n$)', fontsize=8)
    ax.set_ylabel('Sai s? t??ng ??i', fontsize=8)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

    # Chú thích giá tr? t?i ?i?m n=200 ?? quan sát t?c ?? h?i t? ban ??u
    idx = np.searchsorted(sizes, 200)
    if idx < len(sizes):
        ax.annotate(f'$n=200$: {errors[idx]:.4f}', 
                    xy=(sizes[idx], errors[idx]), 
                    xytext=(15, 15),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=color),
                    fontsize=8)
        
def box_plot(df, x_col, y_col):
    """
    V? bi?u ?? h?p (Box Plot) ?? so sánh phân ph?i c?a m?t bi?n ??nh l??ng theo các m?c ch?t l??ng.

    Input:
        - df: DataFrame ch?a d? li?u.
        - x_col: Tên bi?n ??nh danh (th??ng là 'quality') ?? phân nhóm trên tr?c X.
        - y_col: Tên bi?n ??nh l??ng c?n quan sát phân ph?i trên tr?c Y.
    Output:
        - Hi?n th? bi?u ?? Box Plot v?i Seaborn.
    """
    # V? bi?u ?? boxplot ?? quan sát trung v?, t? phân v? và các ?i?m ngo?i l? theo t?ng nhóm
    sns.boxplot(x=x_col, y=y_col, data=df, palette="coolwarm", hue=x_col, legend=False)
    plt.title(f"Bi?u ?? h?p: {y_col} theo {x_col}", fontsize=12, fontweight="bold")
    plt.xlabel(x_col.capitalize(), fontsize=10)
    plt.ylabel(y_col.capitalize(), fontsize=10)
    plt.grid(True, axis='y', alpha=0.3)

def contribution_bar(values, labels, title, color="steelblue", ax=None):
    # Bi?u ?? thanh % ?óng góp c?a t?ng bi?n vào kho?ng cách
    pct = np.array(values) / np.sum(values) * 100
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels[::-1], pct[::-1], color=color, alpha=0.75, edgecolor="white")
    ax.set_xlabel("?óng góp (%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for bar, val in zip(bars, pct[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, pct.max() * 1.2)
    return ax