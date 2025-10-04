import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re

# --- 1. Cấu hình & Các hàm Tải mô hình ---

# Đường dẫn đến thư mục chứa tất cả các mô hình và công cụ đã lưu
MODEL_PACKAGE_DIR = 'trained_models_package'

# Từ điển định nghĩa các câu hỏi và cấu hình cho thanh trượt trên giao diện
INPUT_QUESTIONS = {
    # key: (Nhãn hiển thị, min, max, default_value, step)
    'thoi_gian_ngu': ("Bạn ngủ khoảng bao nhiêu tiếng đêm qua?", 0.0, 16.0, 7.5, 0.5),
    'thoi_gian_su_dung_dt': ("Thời gian sử dụng điện thoại của bạn hôm nay (giờ)?", 0.0, 24.0, 5.0, 0.5),
    'thoi_gian_o_nha_percent': ("Tỷ lệ thời gian bạn ở nhà hôm nay (%)?", 0.0, 1.0, 0.7, 0.05),
    'so_dia_diem_moi': ("Hôm nay bạn đã đến bao nhiêu địa điểm mới?", 0, 20, 3, 1),
    'so_buoc_chan': ("Tổng số bước chân của bạn hôm nay?", 0, 30000, 5000, 500),
    'thoi_gian_cuoc_goi': ("Tổng thời gian gọi điện của bạn hôm nay (phút)?", 0, 300, 30, 5)
}

def get_interpretation(probability):
    """Diễn giải xác suất dự đoán."""
    if probability < 0.4:
        return "Nguy cơ trầm cảm thấp."
    elif probability < 0.6:
        return "Có một vài dấu hiệu, cần chú ý theo dõi. Nên duy trì thói quen sinh hoạt đều đặn và vận động nhẹ nhàng, bổ sung thêm vitamin để nâng cao sức khỏe tinh thần. "
    else:
        return "Nguy cơ trầm cảm cao, nên tìm kiếm tư vấn chuyên môn."

@st.cache_resource
def load_artifacts(package_dir):
    """Tải tất cả các mô hình và công cụ tiền xử lý từ thư mục đã đóng gói."""
    artifacts = {}
    if not os.path.isdir(package_dir):
        st.error(f"Lỗi: Không tìm thấy thư mục '{package_dir}'.")
        return None

    try:
        # Tải các công cụ tiền xử lý
        artifacts['imputer'] = joblib.load(os.path.join(package_dir, 'imputer.joblib'))
        artifacts['selector'] = joblib.load(os.path.join(package_dir, 'selector.joblib'))
        artifacts['initial_features'] = joblib.load(os.path.join(package_dir, 'initial_feature_list.joblib'))
        
        # Tải tất cả các mô hình
        models = {}
        model_files = [f for f in os.listdir(package_dir) if f.startswith('model_') and f.endswith('.joblib')]
        for f in model_files:
            model_name = f.replace('model_', '').replace('.joblib', '').replace('_', ' ')
            models[model_name] = joblib.load(os.path.join(package_dir, f))
        
        artifacts['models'] = models
        st.success(f"Đã tải thành công {len(models)} mô hình và các công cụ.")
        return artifacts

    except FileNotFoundError as e:
        st.error(f"Lỗi: Không tìm thấy tệp '{e.filename}'. Hãy chắc chắn đã chạy script huấn luyện.")
        return None
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None

def predict_from_user_input(user_input, artifacts):
    """Nhận đầu vào từ người dùng, xử lý và đưa ra dự đoán từ tất cả các mô hình."""
    all_predictions = {}
    
    initial_features = artifacts['initial_features']
    imputer = artifacts['imputer']
    selector = artifacts['selector']
    models = artifacts['models']

    feature_mapping = {
        'thoi_gian_ngu': [c for c in initial_features if 'f_slp' in c],
        'thoi_gian_su_dung_dt': [c for c in initial_features if 'f_screen' in c],
        'thoi_gian_o_nha_percent': [c for c in initial_features if 'f_loc' in c and 'timeathome' in c],
        'so_dia_diem_moi': [c for c in initial_features if 'f_loc' in c and ('siglocsvisited' in c or 'numberofsignificantplaces' in c)],
        'so_buoc_chan': [c for c in initial_features if 'f_steps' in c],
        'thoi_gian_cuoc_goi': [c for c in initial_features if 'f_call' in c]
    }
    
    input_df = pd.DataFrame(columns=initial_features)
    input_df.loc[0] = 0.0
    for key, value in user_input.items():
        if key in feature_mapping:
            if key == 'thoi_gian_cuoc_goi': value = value / 60.0
            input_df.loc[0, feature_mapping[key]] = float(value)
            
    input_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
    input_selected = selector.transform(input_imputed)

    for name, model in models.items():
        try:
            prediction_proba = model.predict_proba(input_selected)[:, 1]
            probability = prediction_proba[0]
            interpretation = get_interpretation(probability)
            all_predictions[name] = {'probability': probability, 'interpretation': interpretation}
        except Exception as e:
            st.warning(f"Lỗi khi dự đoán với model '{name}': {e}")
            all_predictions[name] = {'probability': -1, 'interpretation': "Lỗi dự đoán."}

    return all_predictions

def main():
    st.set_page_config(page_title="Dự đoán Trầm cảm", layout="wide", initial_sidebar_state="expanded")
    
    st.title("Ứng dụng Hỗ trợ Sàng lọc Trầm cảm")
    st.markdown("""
    Chào mừng bạn đến với công cụ hỗ trợ sàng lọc trầm cảm dựa trên Học máy. 
    Công cụ này phân tích các thông tin về hành vi hàng ngày để đưa ra đánh giá tham khảo.

    **Lưu ý quan trọng:** Kết quả của ứng dụng này **KHÔNG** thay thế cho chẩn đoán y tế chuyên nghiệp. 
    Nó chỉ mang tính chất tham khảo và nâng cao nhận thức.
    """)
    
    artifacts = load_artifacts(MODEL_PACKAGE_DIR)
    if not artifacts:
        st.stop()

    with st.sidebar.form(key='user_input_form'):
        st.header("Thông tin hàng ngày của bạn")
        user_input_values = {}
        for key, (label, min_val, max_val, default_val, step) in INPUT_QUESTIONS.items():
            user_input_values[key] = st.slider(
                label=label, min_value=min_val, max_value=max_val, 
                value=default_val, step=step, key=key
            )
        submitted = st.form_submit_button("Dự đoán", use_container_width=True, type="primary")

    if submitted:
        st.header("Kết quả Dự đoán")
        with st.spinner('Đang phân tích và dự đoán từ các mô hình...'):
            all_results = predict_from_user_input(user_input_values, artifacts)
        
        if all_results:
            # --- HIỂN THỊ KẾT QUẢ CHÍNH TỪ RANDOM FOREST ---
            main_model_name = "Random Forest"
            if main_model_name in all_results and all_results[main_model_name]['probability'] != -1:
                main_result = all_results[main_model_name]
                st.metric(label=f"Dự đoán Chính (từ mô hình {main_model_name})", value=f"{main_result['probability']:.1%}")
                st.info(f"**Diễn giải:** {main_result['interpretation']}")
            else:
                st.warning("Không thể lấy kết quả từ mô hình chính (Random Forest).")

            st.divider()
            
            # --- HIỂN THỊ KẾT QUẢ CHI TIẾT CỦA TẤT CẢ CÁC MÔ HÌNH ĐỂ SO SÁNH ---
            st.markdown("##### Kết quả chi tiết từ các mô hình")
            sorted_model_names = sorted(all_results.keys())
            
            if sorted_model_names:
                cols = st.columns(len(sorted_model_names))
                for i, model_name in enumerate(sorted_model_names):
                    with cols[i]:
                        result = all_results[model_name]
                        if result['probability'] != -1:
                            st.metric(label=model_name, value=f"{result['probability']:.1%}")
                            st.caption(result['interpretation'])
                        else:
                            st.metric(label=model_name, value="Lỗi")
                            st.caption(result['interpretation'])
            else:
                st.write("Không có mô hình nào để hiển thị.")
        else:
            st.error("Không thể thực hiện dự đoán. Vui lòng kiểm tra lại.")

if __name__ == '__main__':
    main()
