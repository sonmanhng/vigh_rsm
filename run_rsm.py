import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    data_path = '/Users/sonn/Sonn/Workspace/Projects/draft/rsm-clengan/data.xlsx'
    df = pd.read_excel(data_path)
    df = df.dropna()

    X_cols = ['ThoiGian', 'NongDo', 'DM/DL']
    Y_cols = ['H', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    
    # Lấy thông số (min, max, center) của biến điều kiện
    factor_info = df[X_cols].describe().loc[['min', 'max', 'mean', '50%']]
    
    # Đổi tên cột DM/DL thành DM_DL để formula không bị lỗi do dấu '/'
    df = df.rename(columns={'DM/DL': 'DM_DL'})

    for y_col in Y_cols:
        print(f"Processing: {y_col}")
        
        # Tạo folder riêng
        folder = os.path.join(os.path.dirname(data_path), f'RSM_{y_col}')
        os.makedirs(folder, exist_ok=True)
        
        # Xây dựng mô hình RSM (Full Quadratic - Đa thức bậc 2)
        formula = f"{y_col} ~ ThoiGian + NongDo + DM_DL + I(ThoiGian**2) + I(NongDo**2) + I(DM_DL**2) + ThoiGian:NongDo + ThoiGian:DM_DL + NongDo:DM_DL"
        
        model = smf.ols(formula, data=df).fit()
        
        summary_file = os.path.join(folder, f'Report_{y_col}.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Report: {y_col}\n\n")
            f.write("1. Design Space:\n")
            f.write(factor_info.to_string() + "\n\n")
            
            f.write("2. RSM:\n")
            f.write(model.summary().as_text() + "\n\n")
            
            f.write("Goodness-of-Fit:\n")
            f.write(f"R-squared: {model.rsquared:.4f}\n")
            f.write(f"Adj. R-squared: {model.rsquared_adj:.4f}\n")
            f.write(f"F-statistic: {model.fvalue:.4f}\n")
            f.write(f"P-value: {model.f_pvalue:.4e}\n")

        # Vẽ 3 đồ thị Response Surface 3D (Bề mặt đáp ứng)
        # 1. ThoiGian vs NongDo (Giữ DM_DL ở center)
        plot_surface(model, 'ThoiGian', 'NongDo', 'DM_DL', df['DM_DL'].mean(), df, y_col, folder)
        # 2. ThoiGian vs DM_DL (Giữ NongDo ở center)
        plot_surface(model, 'ThoiGian', 'DM_DL', 'NongDo', df['NongDo'].mean(), df, y_col, folder)
        # 3. NongDo vs DM_DL (Giữ ThoiGian ở center)
        plot_surface(model, 'NongDo', 'DM_DL', 'ThoiGian', df['ThoiGian'].mean(), df, y_col, folder)
        
        # Tính điểm cực đại & cực tiểu tối ưu (Tối ưu cục bộ trong khoảng khảo sát - Grid Search)
        # Chia 50 khoảng trên từng trục
        grid_tg = np.linspace(df['ThoiGian'].min(), df['ThoiGian'].max(), 50)
        grid_nd = np.linspace(df['NongDo'].min(), df['NongDo'].max(), 50)
        grid_dmdl = np.linspace(df['DM_DL'].min(), df['DM_DL'].max(), 50)
        
        TG, ND, DMDL = np.meshgrid(grid_tg, grid_nd, grid_dmdl)
        grid_df = pd.DataFrame({
            'ThoiGian': TG.ravel(),
            'NongDo': ND.ravel(),
            'DM_DL': DMDL.ravel()
        })
        grid_df['pred'] = model.predict(grid_df)
        
        max_idx = grid_df['pred'].idxmax()
        min_idx = grid_df['pred'].idxmin()
        
        max_val = grid_df.loc[max_idx, 'pred']
        min_val = grid_df.loc[min_idx, 'pred']
        
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write("\n3. Grid Search:\n\n")
            
            f.write(f"Max {y_col}:\n")
            f.write(f"Value: {max_val:.4f}\n")
            f.write(f"ThoiGian: {grid_df.loc[max_idx, 'ThoiGian']:.2f}\n")
            f.write(f"NongDo: {grid_df.loc[max_idx, 'NongDo']:.2f}\n")
            f.write(f"DM_DL: {grid_df.loc[max_idx, 'DM_DL']:.2f}\n\n")
            
            f.write(f"Min {y_col}:\n")
            f.write(f"Value: {min_val:.4f}\n")
            f.write(f"ThoiGian: {grid_df.loc[min_idx, 'ThoiGian']:.2f}\n")
            f.write(f"NongDo: {grid_df.loc[min_idx, 'NongDo']:.2f}\n")
            f.write(f"DM_DL: {grid_df.loc[min_idx, 'DM_DL']:.2f}\n")
            
        print(f"  -> Generated {summary_file}")

    print("Done!")

def plot_surface(model, var1, var2, fixed_var, fixed_val, df, y_col, folder):
    v1_range = np.linspace(df[var1].min(), df[var1].max(), 50)
    v2_range = np.linspace(df[var2].min(), df[var2].max(), 50)
    V1, V2 = np.meshgrid(v1_range, v2_range)
    
    pred_df = pd.DataFrame({
        var1: V1.ravel(),
        var2: V2.ravel(),
        fixed_var: fixed_val
    })
    
    Z = model.predict(pred_df).values.reshape(V1.shape)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(V1, V2, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    ax.set_xlabel(var1)
    ax.set_ylabel(var2)
    
    # Nhan cho y chua ky tu dac biet nhu %
    ax.set_zlabel(y_col)
    
    ax.set_title(f'RSM 3D {y_col} (center point = {fixed_val:.2f})')
    
    file_name = f'Surface3D_{var1}_vs_{var2}.png'
    plt.savefig(os.path.join(folder, file_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {file_name}")

if __name__ == '__main__':
    main()
