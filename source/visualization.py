import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
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

def contribution_bar(values, labels, title, color="steelblue", ax=None):
    """
    Vẽ biểu đồ thanh ngang thể hiện phần trăm đóng góp của từng biến vào một giá trị tổng (ví dụ: khoảng cách).

    Input:
        - values: Danh sách các giá trị đóng góp của từng biến.
        - labels: Danh sách tên (nhãn) của các biến tương ứng.
        - title: Tiêu đề của biểu đồ.
        - color: Màu sắc của các thanh biểu đồ.
        - ax: Đối tượng trục (axes) của matplotlib để vẽ lên (tùy chọn).
    Output:
        - Đối tượng trục ax sau khi đã vẽ biểu đồ.
    """
    # Biểu đồ thanh % đóng góp của từng biến vào khoảng cách
    pct = np.array(values) / np.sum(values) * 100
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels[::-1], pct[::-1], color=color, alpha=0.75, edgecolor="white")
    ax.set_xlabel("Đóng góp (%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for bar, val in zip(bars, pct[::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, pct.max() * 1.2)
    return ax

def plot_lln_convergence_grid(all_trials_errors, sizes, title="Luật số lớn (LLN): Sự hội tụ của trung bình mẫu về trung bình quần thể"):
    """
    Vẽ lưới biểu đồ hội tụ LLN cho tất cả các biến.

    Input:
        - all_trials_errors: Dictionary chứa ma trận sai số từ simulate_lln.
        - sizes: Mảng các kích thước mẫu n.
        - title: Tiêu đề tổng quát của biểu đồ.
    """
    columns = list(all_trials_errors.keys())
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        errors_matrix = np.array(all_trials_errors[col])
        mean_errors = np.mean(errors_matrix, axis=0)
        std_errors = np.std(errors_matrix, axis=0)
        
        axes[i].plot(sizes, mean_errors, color='blue', lw=2, label='Mean Error')
        axes[i].fill_between(sizes, mean_errors - 1.96*std_errors, mean_errors + 1.96*std_errors, 
                             color='blue', alpha=0.2, label='95% Confidence Band')
        
        axes[i].set_title(f"LLN Convergence: {col}")
        axes[i].set_xlabel("Sample Size (n)")
        axes[i].set_ylabel("Relative Error")
        axes[i].set_ylim(0, max(0.2, np.max(mean_errors)*1.5))
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.show()

def plot_clt_convergence_grid(clt_results, sample_sizes, pop_means, pop_stds, title="Định lý giới hạn trung tâm (CLT): Sự hội tụ của trung bình mẫu về phân phối chuẩn"):
    """
    Vẽ lưới biểu đồ hội tụ CLT cho tất cả các biến.

    Input:
        - clt_results: Dictionary kết quả từ simulate_clt.
        - sample_sizes: Danh sách kích thước mẫu n.
        - pop_means: Dictionary trung bình quần thể.
        - pop_stds: Dictionary độ lệch chuẩn quần thể.
        - title: Tiêu đề tổng quát của biểu đồ.
    """
    columns = list(clt_results.keys())
    fig, axes = plt.subplots(len(columns), len(sample_sizes), figsize=(25, 4 * len(columns)), sharey=False)

    for i, col in enumerate(columns):
        mu = pop_means[col]
        sigma = pop_stds[col]
        
        for j, n in enumerate(sample_sizes):
            ax = axes[i, j]
            data = clt_results[col][n]
            
            # Vẽ histogram thực nghiệm
            sns.histplot(data, ax=ax, color='lightgreen', stat='density')
            # Vẽ đường KDE
            sns.kdeplot(data, ax=ax, color='red', linewidth=2)
            
            # Vẽ đường chuẩn lý thuyết
            x_range = np.linspace(min(data), max(data), 100)
            theory_std = sigma / np.sqrt(n)
            ax.plot(x_range, stats.norm.pdf(x_range, mu, theory_std), color='black', lw=2)
            
            if i == 0:
                ax.set_title(f"n = {n}", fontsize=16, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f"{col}\nDensity", fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel("")
                
    plt.tight_layout()
    plt.suptitle(title, fontsize=22, fontweight='bold', y=1.01)
    plt.show()