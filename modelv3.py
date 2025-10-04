import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import re
import os
import joblib

warnings.filterwarnings('ignore')

# --- Các hàm cốt lõi ---
class Logger(object):
    def __init__(self, filename="model_training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def clean_feature_names(df):
    """Làm sạch tên cột để tương thích với LightGBM."""
    new_cols = {}
    for col in df.columns:
        new_col = re.sub(r'[^A-Za-z0-9_]+', '_', col)
        new_cols[col] = new_col
    df = df.rename(columns=new_cols)
    return df

def evaluate_all_models(X, y, models, output_dir):
    """
    Đánh giá toàn diện các mô hình, in kết quả và tạo các biểu đồ so sánh riêng biệt.
    """
    print("\n--- A. Bắt đầu Đánh giá Hiệu suất Mô hình bằng Cross-Validation ---\n")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for name, model in models.items():
        print(f"--- Đang đánh giá model: {name} ---")
        accuracies, aucs, f1s = [], [], []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                aucs.append(roc_auc_score(y_test, y_pred_proba))
            else:
                aucs.append(0.5)
            
            accuracies.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, average='weighted'))
        
        print("   Classification Report (từ fold cuối cùng để tham khảo):")
        print(classification_report(y_test, model.predict(X_test)))
        
        results.append({
            'model': name, 
            'Accuracy': np.mean(accuracies), 
            'AUC': np.mean(aucs), 
            'F1-score': np.mean(f1s)
        })

    results_df = pd.DataFrame(results)
    print("\n--- Bảng Tổng hợp Kết quả Cross-Validation ---")
    print(results_df.sort_values(by='AUC', ascending=False))

    # *** TÍNH NĂNG MỚI: Tạo và lưu 3 biểu đồ so sánh riêng biệt ***
    print("\n   - Đang tạo các biểu đồ so sánh hiệu suất...")
    
    metrics_to_plot = ['Accuracy', 'AUC', 'F1-score']
    for metric in metrics_to_plot:
        # Sắp xếp dataframe theo từng metric để biểu đồ đẹp hơn
        sorted_df = results_df.sort_values(by=metric, ascending=False)
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=metric, y='model', data=sorted_df, palette='viridis')
        
        plt.xlabel(f'Điểm {metric} Trung bình (Cross-Validation)')
        plt.ylabel('Mô hình')
        plt.title(f'So sánh Hiệu suất {metric} của các Mô hình')
        plt.xlim(0.4, max(0.8, sorted_df[metric].max() + 0.05))
        
        for index, value in enumerate(sorted_df[metric]):
            plt.text(value, index, f'{value:.4f}', va='center')
            
        chart_path = os.path.join(output_dir, f'performance_comparison_{metric}.png')
        plt.tight_layout()
        plt.savefig(chart_path)
        print(f"   -> Đã lưu biểu đồ so sánh '{metric}' vào: '{chart_path}'")

    print("-" * 50)
    
    return results_df

def train_and_save_all_models(X, y, models, output_dir):
    """
    Huấn luyện và lưu tất cả các mô hình trên toàn bộ dữ liệu vào một thư mục chỉ định.
    """
    print(f"\n--- B. Bắt đầu Huấn luyện và Lưu trữ các Mô hình vào thư mục '{output_dir}' ---\n")
    for name, model in models.items():
        try:
            model.fit(X, y)
            model_filename = os.path.join(output_dir, f'model_{name}.joblib')
            joblib.dump(model, model_filename)
            print(f"   -> Đã huấn luyện và lưu mô hình: {model_filename}")
        except Exception as e:
            print(f"   -> LỖI khi huấn luyện model '{name}': {e}")
    print("\n--- Hoàn thành Huấn luyện và Lưu trữ ---")

def main_pipeline(filepath, output_dir):
    """
    Hàm chính điều phối toàn bộ quy trình huấn luyện.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Tải dữ liệu thành công: {df.shape[0]} dòng, {df.shape[1]} cột.")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file '{filepath}'.")
        return

    target_col = 'dep_dep_weekly'
    df.dropna(subset=[target_col], inplace=True)
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))
    y = df[target_col]

    id_cols = ['pid', 'date', 'folder_id']
    potential_leakage_cols = [col for col in df.columns if 'dep_' in col or 'BDI2' in col or 'phq4' in col or 'pss4' in col or 'feel_' in col or '_POST' in col or '_PRE' in col]
    cols_to_drop = id_cols + potential_leakage_cols
    df_features = df.drop(columns=cols_to_drop, errors='ignore')
    
    selected_cols = [col for col in df_features.columns if '_norm_allday' in col or '_norm:allday' in col]
    X = df_features[selected_cols]
    X = clean_feature_names(X)
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    fs_model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = SelectFromModel(fs_model, threshold='median')
    selector.fit(X_imputed, y)
    X_selected_features = X_imputed.columns[selector.get_support()]
    X_final = X_imputed[X_selected_features]
    
    joblib.dump(imputer, os.path.join(output_dir, 'imputer.joblib'))
    joblib.dump(selector, os.path.join(output_dir, 'selector.joblib'))
    joblib.dump(X.columns, os.path.join(output_dir, 'initial_feature_list.joblib'))
    print(f"Lựa chọn được {len(X_selected_features)} đặc trưng. Đã lưu các công cụ tiền xử lý vào '{output_dir}'.")

    models = {
        "Logistic_Regression": LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced', C=0.1),
        "Random_Forest": RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced', max_depth=10),
        "LightGBM": lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbosity=-1),
        "SVC": SVC(random_state=42, probability=True, class_weight='balanced'),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
        "KNeighbors": KNeighborsClassifier(n_neighbors=5)
    }

    # BƯỚC A: Đánh giá hiệu suất, tạo biểu đồ và LƯU LẠI KẾT QUẢ
    performance_df = evaluate_all_models(X_final, y, models, output_dir)
    # joblib.dump(performance_df, os.path.join(output_dir, 'model_performance_scores.joblib'))
    print(f"   -> Đã lưu bảng hiệu suất mô hình vào '{output_dir}'.")

    # BƯỚC B: Huấn luyện và lưu mô hình
    train_and_save_all_models(X_final, y, models, output_dir)
    
    return models.keys()

if __name__ == '__main__':
    OUTPUT_DIRECTORY = 'trained_models_package'
    log_filename = 'model_training_log.txt'
    sys.stdout = Logger(log_filename)
    
    print(f"*** Bắt đầu phiên làm việc. Toàn bộ output sẽ được lưu vào file '{log_filename}' ***\n")

    trained_model_names = main_pipeline('./dataset2/final_dataset_wide.csv', OUTPUT_DIRECTORY)
