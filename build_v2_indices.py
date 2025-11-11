import numpy as np
import pickle
import faiss
import json
import os
from typing import List, Dict, Any

FEATURES_V2_FILE = 'mammography_features_v2.pkl'
TRAIN_CASES_FILE = 'train_cases.json'

def load_features_v2(features_file='mammography_features_v2.pkl'):
    print(f"🔄 載入 V2 特徵檔案: {features_file}")
    with open(features_file, 'rb') as f:
        features_dict = pickle.load(f)
    print(f"✅ 載入完成，共 {len(features_dict)} 個案例")
    return features_dict

def load_train_ids(train_file=TRAIN_CASES_FILE):
    if not os.path.exists(train_file):
        print(f"❌ 錯誤: 找不到訓練集檔案 {train_file}")
        print("請先執行 'prepare_split.py'")
        return None
    
    print(f"🔄 載入訓練集 ID: {train_file}")
    with open(train_file, 'r') as f:
        train_ids = json.load(f)
    print(f"✅ 成功載入 {len(train_ids)} 個訓練集 ID")
    return set(train_ids) # 使用 set 加速查找

def build_global_index(features_dict: Dict[str, Any], train_ids: set, feature_type: str):
    """
    [修改版]
    只使用 'train_ids' 中的案例來建立「全域特徵」索引
    """
    print(f"\n🔨 建立「全域」索引 (僅訓練集): {feature_type}")
    
    features_list = []
    case_ids_map = [] 
    
    # ！！！關鍵修改！！！
    # 只遍歷 features_dict 中，ID 在 train_ids 裡的案例
    for case_id in train_ids:
        if case_id not in features_dict:
            continue
            
        case_data = features_dict[case_id]
        global_features = case_data.get('features', {})
        
        if feature_type in global_features:
            features_list.append(global_features[feature_type])
            case_ids_map.append(case_id)

    if not features_list:
        print(f"❌ 沒有找到任何 {feature_type} 特徵，無法建立索引")
        return

    features_array = np.array(features_list).astype('float32')
    feature_dim = features_array.shape[1]
    
    index = faiss.IndexFlatIP(feature_dim)
    index.add(features_array)
    
    print(f"✅ {feature_type} 索引建立完成 (共 {index.ntotal} 個*訓練*向量)")
    
    # 儲存索引 (我們使用相同的檔名，覆蓋掉舊的 "作弊" 索引)
    index_file = f"faiss_global_{feature_type}.index"
    metadata_file = f"faiss_global_{feature_type}_map.pkl"
    
    faiss.write_index(index, index_file)
    with open(metadata_file, 'wb') as f:
        pickle.dump(case_ids_map, f)
        
    print(f"💾 索引已儲存: {index_file}, {metadata_file}")

def build_lesion_index(features_dict: Dict[str, Any], train_ids: set):
    """
    [修改版]
    只使用 'train_ids' 中的案例來建立「病灶特徵」索引
    """
    print(f"\n🔨 建立「病灶 (ROI)」索引 (僅訓練集)...")
    
    lesion_features = []
    lesion_metadata_map = [] 
    views_to_check = ['RCC', 'LCC', 'RMLO', 'LMLO']
    
    # ！！！關鍵修改！！！
    for case_id in train_ids:
        if case_id not in features_dict:
            continue
            
        case_data = features_dict[case_id]
        case_features = case_data.get('features', {})
        for view in views_to_check:
            if view in case_features:
                view_data = case_features[view]
                for lesion in view_data.get('lesions', []):
                    lesion_features.append(lesion['roi_feature'])
                    lesion_metadata_map.append({
                        'case_id': case_id,
                        'view': view,
                        'bbox': lesion['bbox'],
                        'conf': lesion['conf']
                    })

    if not lesion_features:
        print("⚠️ 警告: 訓練集中沒有找到任何病灶，病灶索引將是空的")
        return

    features_array = np.array(lesion_features).astype('float32')
    feature_dim = features_array.shape[1]
    
    index = faiss.IndexFlatIP(feature_dim)
    index.add(features_array)
    
    print(f"✅ 病灶索引建立完成 (共 {index.ntotal} 個*訓練*病灶向量)")
    
    index_file = f"faiss_lesion_roi.index"
    metadata_file = f"faiss_lesion_roi_map.pkl"
    
    faiss.write_index(index, index_file)
    with open(metadata_file, 'wb') as f:
        pickle.dump(lesion_metadata_map, f)
        
    print(f"💾 索引已儲存: {index_file}, {metadata_file}")

if __name__ == "__main__":
    
    try:
        import faiss
    except ImportError:
        print("❌ 錯誤: 找不到 'faiss-cpu'。請安裝: pip install faiss-cpu")
        exit()

    # 1. 載入完整的 V2 特徵
    features_data = load_features_v2()
    
    # 2. 載入訓練集 ID
    train_case_ids = load_train_ids()
    
    if train_case_ids:
        # 3. 建立所有「全域」索引 (僅訓練集)
        build_global_index(features_data, train_case_ids, 'avg_all_global')
        build_global_index(features_data, train_case_ids, 'avg_right_global')
        build_global_index(features_data, train_case_ids, 'avg_left_global')
        
        # 4. 建立「病灶」索引 (僅訓練集)
        build_lesion_index(features_data, train_case_ids)
        
        print("\n🎉 階段二 (修正版) 完成！所有索引庫均已根據訓練集重建。")