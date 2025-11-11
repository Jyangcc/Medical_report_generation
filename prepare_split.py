import os
import pickle
import json
import random

FEATURES_V2_FILE = 'mammography_features_v2.pkl'
TEST_SET_RATIO = 0.2 # 20% 的資料作為測試集

def split_dataset():
    print(f"🔄 載入 V2 特徵檔案: {FEATURES_V2_FILE} 以取得所有 case ID...")
    
    if not os.path.exists(FEATURES_V2_FILE):
        print(f"❌ 錯誤: 找不到 {FEATURES_V2_FILE}")
        print("請先執行 'run_extraction.py' 來生成這個檔案。")
        return

    with open(FEATURES_V2_FILE, 'rb') as f:
        features_dict = pickle.load(f)
        
    all_case_ids = list(features_dict.keys())
    
    # 隨機打亂
    random.seed(42) # 使用固定的 seed 確保每次切分都一樣
    random.shuffle(all_case_ids)
    
    # 計算切分點
    total_cases = len(all_case_ids)
    test_size = int(total_cases * TEST_SET_RATIO)
    train_size = total_cases - test_size
    
    # 切分
    train_ids = all_case_ids[:train_size]
    test_ids = all_case_ids[train_size:]
    
    print(f"✅ 資料切分完成:")
    print(f"  - 總案例數: {total_cases}")
    print(f"  - 訓練集 (80%): {len(train_ids)} 個案例")
    print(f"  - 測試集 (20%): {len(test_ids)} 個案例")
    
    # 儲存到 JSON
    with open('train_cases.json', 'w') as f:
        json.dump(train_ids, f, indent=2)
    print(f"💾 訓練集 ID 已儲存到: train_cases.json")
    
    with open('test_cases.json', 'w') as f:
        json.dump(test_ids, f, indent=2)
    print(f"💾 測試集 ID 已儲存到: test_cases.json")

if __name__ == "__main__":
    split_dataset()