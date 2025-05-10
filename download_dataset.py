import kagglehub
import os
import pandas as pd
from tqdm import tqdm
import shutil

# Download latest version
path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

print("Path to dataset files:", path)

train_csv = os.path.join(path, "Train.csv")
test_csv = os.path.join(path, "Test.csv")
train_dir = os.path.join(path, "Train")
test_dir = os.path.join(path, "Test")

output_train = "dataset/Train"
output_test = "dataset/Test"

# Bước 3: Tạo thư mục theo nhãn
def prepare_label_dirs(base_dir, labels):
    for label in labels:
        os.makedirs(os.path.join(base_dir, str(label)), exist_ok=True)

# Bước 4: Copy ảnh
def copy_images(df, src_dir, dest_dir):
    labels = df['ClassId'].unique()
    prepare_label_dirs(dest_dir, labels)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['Filename'] if 'Filename' in df.columns else row['Path']
        class_id = row['ClassId']
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dest_dir, str(class_id), os.path.basename(filename))
        if os.path.exists(src_path):  # Kiểm tra nếu ảnh tồn tại
            shutil.copy(src_path, dst_path)
        else:
            print(f"⚠️ File không tồn tại: {src_path}")

# Bước 5: Đọc CSV
df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

# Bước 6: Copy ảnh
copy_images(df_train, path, output_train)
copy_images(df_test, path, output_test)

print("✅ Dataset đã được chia vào thư mục dataset/train và dataset/test.")
shutil.copy(train_csv, output_train)
shutil.copy(test_csv, output_test)