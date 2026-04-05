import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .utils import mean

def distribution(df, column_name):
    """
    Vẽ biểu đồ phân phối của một cột dữ liệu kết hợp Histogram và đường KDE.

    Input:
        - df: DataFrame chứa dữ liệu.
        - column_name: Tên cột cần vẽ phân phối.
    Output:
        - Hiển thị biểu đồ phân phối với Seaborn.
    """
    # Sử dụng histplot để vẽ đồng thời cột tần suất và đường cong mật độ (KDE)
    sns.histplot(df[column_name], kde=True, color="steelblue")
    plt.title(f"Phân phối của {column_name}", fontsize=10, fontweight="bold")
    plt.xlabel(column_name, fontsize=8)
    plt.ylabel("Tần suất (Frequency)", fontsize=8)
    plt.grid(True, alpha=0.3)

def dot_diagram(df, column_name):
    """
    Vẽ biểu đồ chấm (Dot Diagram) 1 chiều để quan sát mật độ tập trung của dữ liệu.

    Input:
        - df: DataFrame chứa dữ liệu.
        - column_name: Tên cột cần vẽ.
    Output:
        - Hiển thị biểu đồ scatter plot 1D (trục Y cố định bằng 0).
    """
    # Tạo mảng số 0 có cùng kích thước để trải dữ liệu trên một đường thẳng ngang
    plt.scatter(df[column_name], np.zeros_like(df[column_name]), alpha=0.5, color="salmon", s=1)
    plt.title(f"Biểu đồ chấm của {column_name}", fontsize=10, fontweight="bold")
    plt.xlabel(column_name, fontsize=8)
    plt.yticks([])  # Ẩn trục Y vì đây là biểu đồ 1 chiều
    plt.grid(True, axis='x', alpha=0.3)

def lln_convergence(ax, sizes, errors, column_name, color="steelblue"):
    """
    Vẽ đồ thị hội tụ theo Luật số lớn (LLN) - sai số tương đối giảm khi kích thước mẫu tăng.

    Input:
        - ax: Đối tượng trục (axes) của matplotlib để vẽ lên.
        - sizes: Danh sách các kích thước mẫu (n).
        - errors: Danh sách sai số tương đối tương ứng.
        - column_name: Tên biến đang xét.
        - color: Màu sắc của đường vẽ.
    Output:
        - Vẽ biểu đồ hội tụ lên ax được cung cấp.
    """
    ax.plot(sizes, errors, color=color, lw=1.2, alpha=0.85, label=column_name)
    ax.axhline(0, color='black', lw=0.8) # Đường tham chiếu sai số bằng 0
    ax.fill_between(sizes, errors, alpha=0.15, color=color) # Tô màu vùng dưới đường cong
    ax.set_title(f'LLN: {column_name}', fontsize=10, fontweight="bold")
    ax.set_xlabel('Kích thước mẫu ($n$)', fontsize=8)
    ax.set_ylabel('Sai số tương đối', fontsize=8)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

    # Chú thích giá trị tại điểm n=200 để quan sát tốc độ hội tụ ban đầu
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
    Vẽ biểu đồ hộp (Box Plot) để so sánh phân phối của một biến định lượng theo các mức chất lượng.

    Input:
        - df: DataFrame chứa dữ liệu.
        - x_col: Tên biến định danh (thường là 'quality') để phân nhóm trên trục X.
        - y_col: Tên biến định lượng cần quan sát phân phối trên trục Y.
    Output:
        - Hiển thị biểu đồ Box Plot với Seaborn.
    """
    # Vẽ biểu đồ boxplot để quan sát trung vị, tứ phân vị và các điểm ngoại lệ theo từng nhóm
    sns.boxplot(x=x_col, y=y_col, data=df, palette="coolwarm", hue=x_col, legend=False)
    plt.title(f"Biểu đồ hộp: {y_col} theo {x_col}", fontsize=12, fontweight="bold")
    plt.xlabel(x_col.capitalize(), fontsize=10)
    plt.ylabel(y_col.capitalize(), fontsize=10)
    plt.grid(True, axis='y', alpha=0.3)
