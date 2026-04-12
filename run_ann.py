import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


DATA_PATH = "data_ann.xlsx"
INPUT_COLS = ["ThoiGian", "NongDo", "TyLe"]
TARGET_GOAL = "MAX"
GRID_RESOLUTION = 40

def optimize_for_variable(df, output_col):
    results_folder = f"ANN_Results_{output_col}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        
    print(f"\n{'='*60}")
    print(f"=== BẮT ĐẦU TỐI ƯU HOÁ ANN CHO BIẾN: {output_col} ===")
    print(f"{'='*60}")

    df_clean = df.dropna(subset=INPUT_COLS + [output_col])
    
    if len(df_clean) == 0:
        print(f"Bỏ qua {output_col}: Không có dữ liệu hợp lệ.")
        return

    X = df_clean[INPUT_COLS].values
    y = df_clean[output_col].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    input_dim = len(INPUT_COLS)
    
    print("\n--- THÔNG TIN DỮ LIỆU ---")
    print(f"Số lượng mẫu phân tích  : {len(df_clean)}")
    print(f"Danh sách Đầu vào (X)   : {INPUT_COLS}")
    print(f"Miền giá trị {output_col:13s}: [{y.min():.4f}, {y.max():.4f}]")

    best_model = None
    best_r2 = -float("inf")
    
    architectures = [
        (4,), (8,), (12,), (16,), 
        (8, 4), (12, 6), (16, 8)
    ]
    max_attempts = 400  
    
    print("\n--- Đang huấn luyện mô hình ANN (Mục tiêu R2 > 0.80)...")
    for attempt in range(max_attempts):
        arch = architectures[attempt % len(architectures)]
        model = MLPRegressor(
            hidden_layer_sizes=arch,
            activation='tanh',     
            solver='lbfgs',        
            alpha=0.2,             
            max_iter=1500,       
            random_state=attempt
        )
        model.fit(X_scaled, y_scaled.ravel())
        y_pred_scaled = model.predict(X_scaled)
        r2_current = r2_score(y_scaled, y_pred_scaled)
        
        if r2_current > best_r2:
            best_r2 = r2_current
            best_model = model
            
        if best_r2 >= 0.80:
            break
            

    model = best_model
    r2_ann = best_r2

    print(f"\n--- Đang tạo lưới không gian phân tích tìm cực trị ({TARGET_GOAL})...")
    
    input_ranges = [np.linspace(df_clean[c].min(), df_clean[c].max(), GRID_RESOLUTION) for c in INPUT_COLS]
    grid_tuples = list(itertools.product(*input_ranges))
    X_grid_all = np.array(grid_tuples)

    batch_size = 50000
    y_pred_grid_scaled = []
    for i in range(0, len(X_grid_all), batch_size):
        batch = X_grid_all[i:i+batch_size]
        y_pred_grid_scaled.extend(model.predict(scaler_X.transform(batch)).flatten())

    y_pred_grid = scaler_y.inverse_transform(np.array(y_pred_grid_scaled).reshape(-1, 1)).flatten()

    best_idx = np.argmax(y_pred_grid) if TARGET_GOAL.upper() == "MAX" else np.argmin(y_pred_grid)
    best_y = y_pred_grid[best_idx]
    best_X = X_grid_all[best_idx]

    # Report
    report_text = f"Report: ANN Optimization cho biến {output_col}\n"
    report_text += "="*40 + "\n"
    report_text += f"- Đầu vào: {INPUT_COLS}\n"
    report_text += f"- Mô hình ANN R2-Score: {r2_ann:.4f}\n"
    report_text += f"- Tiêu chí: {TARGET_GOAL}\n"
    report_text += "-"*40 + "\n"
    report_text += f"KẾT QUẢ {TARGET_GOAL}:\n"
    report_text += f"  > Giá trị {output_col}: {best_y:.4f}\n\n"
    report_text += "Tại Điều Kiện:\n"
    
    print(f"\n{'='*40}")
    print(f" ĐIỀU KIỆN TỐI ƯU CHO MỤC TIÊU CỰC TRỊ: {TARGET_GOAL} {output_col}")
    print(f"{'='*40}")
    
    for i, col in enumerate(INPUT_COLS):
        print(f"   - [{col:10s}]: {best_X[i]:.4f}")
        report_text += f"   - {col:10s}: {best_X[i]:.4f}\n"

    report_file = os.path.join(results_folder, f"Report_ANN_{output_col}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  -> Đã lưu Report tại {report_file}")
        

    if input_dim >= 2:
        print("--- Đang vẽ bản đồ không gian Bề Mặt (Surface) & Vệ Tinh (Contour)...")
        pairs = list(itertools.combinations(range(len(INPUT_COLS)), 2))
        for pair in pairs:
            idx_1, idx_2 = pair
            col1, col2 = INPUT_COLS[idx_1], INPUT_COLS[idx_2]
            
            v1_range = np.linspace(df_clean[col1].min(), df_clean[col1].max(), 50)
            v2_range = np.linspace(df_clean[col2].min(), df_clean[col2].max(), 50)
            V1, V2 = np.meshgrid(v1_range, v2_range)
            
            inputs = np.zeros((V1.size, len(INPUT_COLS)))
            for i in range(len(INPUT_COLS)):
                inputs[:, i] = best_X[i]
            inputs[:, idx_1], inputs[:, idx_2] = V1.ravel(), V2.ravel()
            
            Z_scaled = model.predict(scaler_X.transform(inputs))
            Z = scaler_y.inverse_transform(Z_scaled.reshape(-1, 1)).ravel().reshape(V1.shape)
            
            fig = plt.figure(figsize=(15, 6))
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax1.plot_surface(V1, V2, Z, cmap='plasma', alpha=0.9)
            ax1.scatter(best_X[idx_1], best_X[idx_2], best_y, color='red', s=200, edgecolors='white', marker='*', label='Optimal Point', zorder=5)
            ax1.set_xlabel(col1)
            ax1.set_ylabel(col2)
            ax1.set_zlabel(output_col)
            
            title_str = ", ".join([f"{INPUT_COLS[i]}={best_X[i]:.2f}" for i in range(len(INPUT_COLS)) if i not in pair])
            main_title = f"3D Surface: {col1} vs {col2}"
            if title_str: main_title += f"\n(Fixed: {title_str})"
            ax1.set_title(main_title)
            ax1.view_init(elev=25, azim=135)
            ax1.legend()
            
            ax2 = fig.add_subplot(1, 2, 2)
            contour = ax2.contourf(V1, V2, Z, levels=35, cmap='plasma')
            fig.colorbar(contour, ax=ax2, label=output_col)
            ax2.scatter(best_X[idx_1], best_X[idx_2], color='white', s=200, edgecolors='black', marker='*', label='Optimal')
            ax2.set_xlabel(col1)
            ax2.set_ylabel(col2)
            ax2.set_title(f"Contour: {col1} vs {col2}")
            ax2.legend()
            plt.tight_layout()
            
            img_filepath = os.path.join(results_folder, f"ANN_{output_col}_{col1}_vs_{col2}.png")
            plt.savefig(img_filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  -> Saved {img_filepath}")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Lỗi: Không tìm thấy tệp {DATA_PATH}")
        sys.exit(1)
        
    if DATA_PATH.endswith('.xlsx'):
        df = pd.read_excel(DATA_PATH)
    elif DATA_PATH.endswith('.csv'):
        df = pd.read_csv(DATA_PATH)
    else:
        print("Lỗi: Phải sử dụng file .xlsx hoặc .csv")
        sys.exit(1)

    missing_cols = [c for c in INPUT_COLS if c not in df.columns]
    if missing_cols:
        print(f"Lỗi: Thiếu cột đầu vào: {missing_cols}")
        print(f"Các cột có sẵn: {df.columns.tolist()}")
        sys.exit(1)
        
    OUTPUT_COLS = [c for c in df.columns if c not in INPUT_COLS]
    TARGET_VARS = ["S3", "S4", "S5", "S6"]
    OUTPUT_COLS = [c for c in OUTPUT_COLS if c in TARGET_VARS]
    if not OUTPUT_COLS:
        OUTPUT_COLS = TARGET_VARS
    print(f"Đã xác định các cột Output (Mục tiêu): {OUTPUT_COLS}")
    
    for out_col in OUTPUT_COLS:
        optimize_for_variable(df, out_col)
        

if __name__ == "__main__":
    main()
