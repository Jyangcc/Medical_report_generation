import os
import numpy as np
import pydicom
from collections import Counter
import cv2
from tqdm import tqdm

dir_path = 'Kang_Ning_General_Hospital/'
child_dir = ['20230721_1st', '20230728_2nd', '20230804_3rd']
file_name_list = ['I0000000.dcm', 'I0000001.dcm', 'I0000002.dcm', 'I0000003.dcm']

# Read not full files
not_full_files = []
with open('docs/not_full_files.txt', 'r') as f:
    for line in f:
        not_full_files.append(line.strip())

# Collect all sizes
all_sizes = []
size_counter = Counter()

print("=== 步驟1: 掃描所有影像尺寸 ===")
for child in child_dir:
    full_child_path = os.path.join(dir_path, child)
    files = os.listdir(full_child_path)
    
    for file in tqdm(files, desc=f"Processing {child}"):
        full_file_path = os.path.join(full_child_path, file)
        
        if full_file_path in not_full_files:
            continue
            
        if os.path.isdir(full_file_path):
            for file_name in file_name_list:
                dicom_path = os.path.join(full_file_path, file_name)
                try:
                    ds = pydicom.dcmread(dicom_path)
                    pixel_array = ds.pixel_array
                    size = pixel_array.shape
                    all_sizes.append((dicom_path, size, file_name))
                    size_counter[size] += 1
                except Exception as e:
                    print(f"Error reading {dicom_path}: {str(e)}")

# 統計結果
print("\n=== 影像尺寸統計 ===")
for size, count in size_counter.most_common():
    print(f"尺寸 {size}: {count} 張影像 ({count/len(all_sizes)*100:.2f}%)")

# 按檔案名稱分類統計
print("\n=== 按檔案名稱分類 ===")
for fname in file_name_list:
    fname_sizes = [size for path, size, fn in all_sizes if fn == fname]
    fname_counter = Counter(fname_sizes)
    print(f"\n{fname}:")
    for size, count in fname_counter.most_common():
        print(f"  {size}: {count} 張")

# 決定目標尺寸(使用最常見的尺寸或固定尺寸)
target_size = size_counter.most_common(1)[0][0]
print(f"\n=== 建議目標尺寸: {target_size} ===")

# 選項: 也可以使用固定尺寸,例如 (224, 224) 或 (512, 512)
# target_size = (512, 512)  # 取消註解以使用固定尺寸

print("\n=== 步驟2: 統一影像尺寸並儲存 ===")
output_dir = 'preprocessed_images/'
os.makedirs(output_dir, exist_ok=True)

for child in child_dir:
    child_output_dir = os.path.join(output_dir, child)
    os.makedirs(child_output_dir, exist_ok=True)
    
    full_child_path = os.path.join(dir_path, child)
    files = os.listdir(full_child_path)
    
    for file in tqdm(files, desc=f"Resizing {child}"):
        full_file_path = os.path.join(full_child_path, file)
        
        if full_file_path in not_full_files:
            continue
            
        if os.path.isdir(full_file_path):
            case_output_dir = os.path.join(child_output_dir, file)
            os.makedirs(case_output_dir, exist_ok=True)
            
            for file_name in file_name_list:
                dicom_path = os.path.join(full_file_path, file_name)
                try:
                    ds = pydicom.dcmread(dicom_path)
                    pixel_array = ds.pixel_array
                    
                    # 正規化到 0-255
                    pixel_array = ((pixel_array - pixel_array.min()) / 
                                   (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
                    
                    # Resize
                    if pixel_array.shape != target_size:
                        resized = cv2.resize(pixel_array, 
                                           (target_size[1], target_size[0]), 
                                           interpolation=cv2.INTER_LANCZOS4)
                    else:
                        resized = pixel_array
                    
                    # 儲存為 numpy array
                    output_path = os.path.join(case_output_dir, 
                                              file_name.replace('.dcm', '.npy'))
                    np.save(output_path, resized)
                    
                except Exception as e:
                    print(f"Error processing {dicom_path}: {str(e)}")

print("\n預處理完成!")
