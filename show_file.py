import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np

# 確保 pydicom 可以處理 JPEG 2000
from pydicom.pixel_data_handlers import apply_modality_lut

PATH = 'Kang_Ning_General_Hospital/20230721_1st/MAMO_DEID_20230721_-00004'
file_name_list = ['I0000000.dcm', 'I0000001.dcm','I0000002.dcm', 'I0000003.dcm']

# 設置圖形顯示
fig, axes = plt.subplots(1, len(file_name_list), figsize=(20, 4))
if len(file_name_list) == 1:
    axes = [axes]

# 顯示每個檔案
for idx, file_name in enumerate(file_name_list):
    try:
        full_file_path = os.path.join(PATH, file_name)
        ds = pydicom.dcmread(full_file_path)
        
        # 取得像素資料
        pixel_array = ds.pixel_array
        
        # 應用 DICOM 的視窗調整（如果有的話）
        if hasattr(ds, 'WindowWidth') and hasattr(ds, 'WindowCenter'):
            # 應用窗寬窗位
            pixel_array = apply_modality_lut(pixel_array, ds)
            
        # 顯示影像
        axes[idx].imshow(pixel_array, cmap='gray')
        axes[idx].set_title(f'{file_name}')
        # axes[idx].axis('off')
        
        # 印出一些基本資訊
        print(f"\n檔案: {file_name}")
        print(f"  影像尺寸: {pixel_array.shape}")
        print(f"  傳輸語法: {ds.file_meta.TransferSyntaxUID.name if hasattr(ds.file_meta, 'TransferSyntaxUID') else 'Unknown'}")
        if hasattr(ds, 'PatientName'):
            print(f"  病患姓名: {ds.PatientName}")
        if hasattr(ds, 'Modality'):
            print(f"  檢查類型: {ds.Modality}")
            
    except Exception as e:
        print(f"處理 {file_name} 時發生錯誤: {str(e)}")
        axes[idx].text(0.5, 0.5, f'Error loading\n{file_name}', 
                      horizontalalignment='center', 
                      verticalalignment='center',
                      transform=axes[idx].transAxes)
        axes[idx].axis('off')

plt.tight_layout()
plt.show()

# 處理 I0000004.dcm 時發生錯誤: unsupported operand type(s) for *: 'float' and 'NoneType' 他好像是不一樣的資訊