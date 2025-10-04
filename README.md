Hướng dẫn Triển khai Ứng dụng Dự đoán Trầm cảm
Đây là tài liệu hướng dẫn để triển khai và chạy ứng dụng Streamlit dự đoán trầm cảm.

Yêu cầu
Python 3.9+

Một thư mục trained_models_package chứa tất cả các file .joblib đã được huấn luyện từ script mới nhất.

Cấu trúc Thư mục
Để ứng dụng hoạt động chính xác, hãy đảm bảo cấu trúc thư mục của bạn như sau:

/your_deployment_folder/
|
|-- trained_models_package/
|   |-- imputer.joblib
|   |-- selector.joblib
|   |-- initial_feature_list.joblib
|   |-- model_performance.joblib  <-- File mới chứa hiệu suất
|   |-- model_Random_Forest.joblib
|   |-- ... (và các mô hình khác)
|
|-- app.py             (File ứng dụng Streamlit, đã cập nhật)
|-- requirements.txt   (File chứa các thư viện cần thiết)
|-- train_model.py     (File huấn luyện mô hình, đã cập nhật)

Các bước Triển khai
Bước 1: Chạy lại script huấn luyện

Vì chúng ta cần tạo ra file model_performance.joblib mới, bạn cần chạy lại script huấn luyện một lần cuối.

Bước 2: Cài đặt các thư viện cần thiết

Mở terminal hoặc command prompt, di chuyển đến thư mục gốc của dự án (your_deployment_folder) và chạy lệnh sau (nếu chưa chạy):

pip install -r requirements.txt

Bước 3: Chạy ứng dụng Streamlit

Sau khi cài đặt thành công, chạy lệnh sau để khởi động ứng dụng web:

streamlit run app.py

Cách sử dụng Ứng dụng
Sử dụng các thanh trượt ở thanh bên (sidebar) bên trái để nhập thông tin.

Nhấn nút "Dự đoán".

Một Dự đoán Tổng hợp (Theo Trọng số Hiệu suất) sẽ được hiển thị nổi bật ở trên cùng. Đây là kết quả tham khảo chính, được tính toán bằng cách ưu tiên các mô hình có độ chính xác cao hơn.

Kết quả chi tiết từ từng mô hình riêng lẻ sẽ được hiển thị bên dưới để so sánh.