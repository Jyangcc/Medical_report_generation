from detection_and_feature_extractor import ImageFeatureExtractor
import os

# ==================== V3 特徵提取 (API 版本) ====================
if __name__ == "__main__":
    
    # 1. 初始化特徵提取器 (V3)
    #    不再需要傳入 detector_model_path
    #    它會自動從環境變數讀取 ROBOFLOW_API_KEY
    try:
        extractor = ImageFeatureExtractor()
        
        # 2. 批次提取所有案例的特徵
        features_dict = extractor.batch_extract_all_cases(
            preprocessed_dir='preprocessed_images/',
            output_file='mammography_features_v2.pkl' # 儲存為 V2 版本
        )
        
        # 3. 查看結果 (同前)
        print("\n" + "="*60)
        print("📊 V2 (API) 特徵提取統計:")
        print(f"總案例數: {len(features_dict)}")
        
        if len(features_dict) > 0:
            # 試著找一個有病灶的案例來顯示
            found_case_with_lesion = None
            for case_id, data in features_dict.items():
                if data['total_lesions_found'] > 0:
                    found_case_with_lesion = case_id
                    break
            
            if not found_case_with_lesion:
                found_case_with_lesion = list(features_dict.keys())[0]

            case_data = features_dict[found_case_with_lesion]
            case_features = case_data['features']
            
            print(f"\n範例案例: {found_case_with_lesion}")
            print(f"來源: {case_data['source_dir']}")
            print(f"總共檢測到的病灶數: {case_data['total_lesions_found']}")
            
            if 'LCC' in case_features:
                lcc_data = case_features['LCC']
                print(f"\nLCC 視圖資訊:")
                print(f"  - Global Feature shape: {lcc_data['global_feature'].shape}")
                print(f"  - Lesions found in LCC: {len(lcc_data['lesions'])}")
                if lcc_data['lesions']:
                    print(f"    - 範例病灶 1 BBox: {lcc_data['lesions'][0]['bbox']}")
                    print(f"    - 範例病灶 1 Conf: {lcc_data['lesions'][0]['conf']:.3f}")
                    print(f"    - 範例病灶 1 ROI Feature shape: {lcc_data['lesions'][0]['roi_feature'].shape}")
        
        print("="*60)

    except ValueError as e:
        print(f"\n❌ 執行失敗: {e}")
        print("請確保你已經設定了 ROBOFLOW_API_KEY 環境變數。")