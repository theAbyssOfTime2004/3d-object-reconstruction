# filepath: scripts/01_prepare_raw_data.py
import os
import shutil
import zipfile
from huggingface_hub import hf_hub_download, login
from dotenv import load_dotenv

def download_and_organize_shapenet_category(category_id="03001627", base_data_dir="data"):
    """
    Tải file zip cho một category của ShapeNet, giải nén và tổ chức lại.
    """
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Lỗi: HF_TOKEN không được tìm thấy trong biến môi trường.")
        return
    login(hf_token)

    # --- Tải file ---
    download_target_dir = os.path.join(base_data_dir, "temp_downloads") # Thư mục tải tạm
    os.makedirs(download_target_dir, exist_ok=True)

    print(f"Đang tải {category_id}.zip...")
    try:
        file_path = hf_hub_download(
            repo_id="ShapeNet/ShapeNetCore",
            filename=f"{category_id}.zip",
            repo_type="dataset",
            local_dir=download_target_dir,
            local_dir_use_symlinks=False
        )
        print(f"Đã tải xong: {file_path}")
    except Exception as e:
        print(f"Lỗi khi tải file: {e}")
        return

    # --- Giải nén ---
    extraction_temp_dir = os.path.join(base_data_dir, "temp_extraction", category_id)
    os.makedirs(extraction_temp_dir, exist_ok=True)
    print(f"Đang giải nén {file_path} vào {extraction_temp_dir}...")
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_temp_dir)
        print("Giải nén hoàn tất.")
    except Exception as e:
        print(f"Lỗi khi giải nén: {e}")
        # Dọn dẹp thư mục tải tạm nếu giải nén lỗi
        if os.path.exists(download_target_dir):
            shutil.rmtree(download_target_dir)
        return

    # --- Tổ chức lại thư mục ---
    # Thư mục đích cho dữ liệu thô
    target_raw_data_dir_for_category = os.path.join(base_data_dir, 'shapenet_raw_data', category_id)
    os.makedirs(target_raw_data_dir_for_category, exist_ok=True)

    # Xác định thư mục nguồn chứa các model_id (có thể bị lồng)
    potential_nested_path = os.path.join(extraction_temp_dir, category_id)
    source_model_ids_path = None

    if os.path.isdir(potential_nested_path) and any(os.path.isdir(os.path.join(potential_nested_path, item)) for item in os.listdir(potential_nested_path)):
        source_model_ids_path = potential_nested_path
    elif os.path.isdir(extraction_temp_dir) and any(os.path.isdir(os.path.join(extraction_temp_dir, item)) for item in os.listdir(extraction_temp_dir)):
        source_model_ids_path = extraction_temp_dir
    
    if not source_model_ids_path:
        print(f"Lỗi: Không tìm thấy thư mục chứa model IDs trong {extraction_temp_dir}")
        # Dọn dẹp
        if os.path.exists(download_target_dir): shutil.rmtree(download_target_dir)
        if os.path.exists(os.path.join(base_data_dir, "temp_extraction")): shutil.rmtree(os.path.join(base_data_dir, "temp_extraction"))
        return

    print(f"Đang di chuyển model IDs từ {source_model_ids_path} đến {target_raw_data_dir_for_category}...")
    for item_name in os.listdir(source_model_ids_path):
        source_item = os.path.join(source_model_ids_path, item_name)
        target_item = os.path.join(target_raw_data_dir_for_category, item_name)
        if os.path.isdir(source_item): # Chỉ di chuyển thư mục model_id
            if not os.path.exists(target_item):
                shutil.move(source_item, target_item)
            # else: # Xử lý nếu thư mục đích đã tồn tại (ví dụ: ghi đè, bỏ qua)
            #     print(f"Cảnh báo: {target_item} đã tồn tại. Bỏ qua.")
    print("Hoàn tất tổ chức dữ liệu thô.")

    # --- Dọn dẹp thư mục tạm ---
    print("Đang dọn dẹp thư mục tạm...")
    if os.path.exists(download_target_dir):
        shutil.rmtree(download_target_dir)
    if os.path.exists(os.path.join(base_data_dir, "temp_extraction")):
        shutil.rmtree(os.path.join(base_data_dir, "temp_extraction"))
    print("Dọn dẹp hoàn tất.")

if __name__ == "__main__":
    # Ví dụ sử dụng:
    download_and_organize_shapenet_category(category_id="03001627", base_data_dir="data")
    # Bạn có thể thêm các category khác nếu cần
    # download_and_organize_shapenet_category(category_id="ANOTHER_ID", base_data_dir="data")