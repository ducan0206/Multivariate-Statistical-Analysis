# Multivariate Statistical Analysis

Dự án thực hiện phân tích thống kê đa biến trên tập dữ liệu [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) (Red Wine) để thực hành các khái niệm thống kê cơ bản.

## 📂 Cấu trúc thư mục

```text
Multivariate-Statistical-Analysis/
├── data/                               # Chứa tập dữ liệu (CSV)
│   └── winequality-red.csv             # Dataset chính cho phân tích
├── notebooks/                          # Jupyter Notebooks cho từng giai đoạn
│   ├── 01_descriptive_stats.ipynb      # Các đại lượng thống kê cơ bản
│   ├── 02_clt_lln_proof.ipynb          # Mô phỏng LLN và CLT
│   └── 03_statistical_distance.ipynb   # Khoảng cách Euclid và Thống kê
├── source/                             # Mã nguồn Python tái sử dụng
│   ├── __init__.py
│   ├── utils.py                        # Các hàm tính toán thống kê (mean, var, cov, corr, dist)
│   └── visualization.py                # Các hàm vẽ biểu đồ (LLN, CLT, distribution)
├── requirements.txt                    # Danh sách các thư viện cần thiết
└── README.md                           # Hướng dẫn dự án
```

---

## 🚀 Hướng dẫn cài đặt

### 1. Cài đặt môi trường
Khuyên dùng Python 3.10 trở lên. Nên sử dụng môi trường ảo (venv hoặc conda).

**Sử dụng venv:**
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Linux/macOS)
source venv/bin/activate
# Kích hoạt môi trường (Windows)
# venv\Scripts\activate
```

### 2. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 3. Chạy Notebook
Có thể mở các file trong thư mục `notebooks/` bằng VSCode (có cài Jupyter extension) hoặc chạy lệnh:
```bash
jupyter notebook
```
Sau đó truy cập vào trình duyệt và chọn file cần chạy. Hoặc vô từng file và ấn `Run all`

---

## 📝 Phân công nhiệm vụ

| Thành viên | Nhiệm vụ chính |
| :--- | :--- |
| **23120109 - Lê Đức An** | - Xây dựng module `source/utils.py`.<br>- Thực hiện `02_clt_lln_proof.ipynb` (CLT) và `03_statistical_distance.ipynb`. |
| **23122014 - Hoàng Minh Trung** | - Xây dựng module `source/visualization.py`.<br>- Thực hiện `01_descriptive_stats.ipynb` và `02_clt_lln_proof.ipynb` (LLN).<br>- Viết báo cáo tổng hợp và hoàn thiện `README.md`. |

---

## 📊 Tóm tắt kết quả

- **Thống kê mô tả:**
    - Dataset gồm 1,599 mẫu rượu vang đỏ với chất lượng trung bình 5.64/10.
    - Xác định được `alcohol` (r=0.48) và `volatile acidity` (r=-0.39) là hai yếu tố quan trọng nhất ảnh hưởng đến chất lượng rượu.
    - Hầu hết các biến có phân phối lệch, ngoại trừ `pH` và `density` phân phối gần chuẩn.
- **Luật số lớn (LLN):** Đã mô phỏng sự hội tụ của trung bình mẫu ($\bar{x}$) về trung bình quần thể ($\mu$) thông qua trung bình tích lũy (Cumulative Mean) với 100 lần thử nghiệm.
- **Định lý giới hạn trung tâm (CLT):** Chứng minh phân phối của $\bar{x}$ tiến về phân phối chuẩn $\mathcal{N}(\mu, \sigma^2/n)$ khi kích thước mẫu $n$ tăng, kể cả với các biến có phân phối lệch mạnh.
- **Khoảng cách thống kê:** Phân tích sự khác biệt giữa các điểm dữ liệu bằng khoảng cách Euclid và khoảng cách thống kê (chuẩn hóa theo phương sai).

---

## 🛠 Thư viện sử dụng
- `numpy`, `pandas`: Xử lý dữ liệu và tính toán vector.
- `matplotlib`, `seaborn`: Trực quan hóa dữ liệu.
- `scipy.stats`: Kiểm định thống kê (Shapiro-Wilk) và các hàm xác suất.
- `IPython.display`: Hiển thị Markdown trong Notebook.
