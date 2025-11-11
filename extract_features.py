import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pickle
import open_clip

class ImageFeatureExtractor:
    """
    使用 BiomedCLIP 提取乳房 X 光影像的特徵向量
    支援個別影像和整體平均特徵
    """
    
    def __init__(self, model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        """初始化模型"""
        print("🔄 載入 BiomedCLIP 模型...")
        self.device = torch.device("cpu")
        
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型載入完成 (device: {self.device})")
    
    def extract_single_image(self, image_array):
        """提取單張影像的特徵向量"""
        if len(image_array.shape) == 3:
            image_array = image_array.squeeze()
        
        image_array = image_array.astype(np.uint8)
        image_pil = Image.fromarray(image_array).convert('RGB')
        
        image_tensor = self.preprocess_val(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().squeeze()
    
    def extract_case_features(self, case_dir):
        """
        提取一個案例的所有影像特徵
        
        Returns:
            features_dict: {
                'RCC': feature,
                'LCC': feature,
                'RMLO': feature,
                'LMLO': feature,
                'avg_all': 4張平均,
                'avg_right': RCC+RMLO平均,
                'avg_left': LCC+LMLO平均
            }
        """
        # 定義影像名稱對應
        view_mapping = {
            'I0000000.npy': 'RCC',   # Right CC
            'I0000001.npy': 'LCC',   # Left CC
            'I0000002.npy': 'RMLO',  # Right MLO
            'I0000003.npy': 'LMLO'   # Left MLO
        }
        
        features = {}
        
        for fname, view_name in view_mapping.items():
            file_path = os.path.join(case_dir, fname)
            
            if not os.path.exists(file_path):
                continue
            
            try:
                image_array = np.load(file_path)
                feature = self.extract_single_image(image_array)
                features[view_name] = feature
            except Exception as e:
                print(f"⚠️  處理 {file_path} 時出錯: {str(e)}")
                continue
        
        if len(features) == 0:
            raise ValueError(f"案例 {case_dir} 沒有有效的影像!")
        
        # 計算不同組合的平均特徵
        result = features.copy()
        
        # 全部平均
        all_features = list(features.values())
        result['avg_all'] = np.mean(all_features, axis=0)
        result['avg_all'] = result['avg_all'] / np.linalg.norm(result['avg_all'])
        
        # 右乳平均 (RCC + RMLO)
        if 'RCC' in features and 'RMLO' in features:
            result['avg_right'] = np.mean([features['RCC'], features['RMLO']], axis=0)
            result['avg_right'] = result['avg_right'] / np.linalg.norm(result['avg_right'])
        
        # 左乳平均 (LCC + LMLO)
        if 'LCC' in features and 'LMLO' in features:
            result['avg_left'] = np.mean([features['LCC'], features['LMLO']], axis=0)
            result['avg_left'] = result['avg_left'] / np.linalg.norm(result['avg_left'])
        
        return result
    
    def batch_extract_all_cases(self, preprocessed_dir='preprocessed_images/', 
                                output_file='mammography_features.pkl'):
        """批次處理所有案例"""
        features_dict = {}
        child_dirs = ['20230721_1st', '20230728_2nd', '20230804_3rd']
        
        print("\n🚀 開始批次提取特徵...")
        
        total_success = 0
        total_failed = 0
        
        for child in child_dirs:
            child_path = os.path.join(preprocessed_dir, child)
            
            if not os.path.exists(child_path):
                print(f"⚠️  找不到資料夾: {child_path}")
                continue
            
            cases = sorted([c for c in os.listdir(child_path) if os.path.isdir(os.path.join(child_path, c))])
            
            for case in tqdm(cases, desc=f"提取 {child} 特徵"):
                case_dir = os.path.join(child_path, case)
                
                try:
                    # 提取特徵（包含個別和平均）
                    case_features = self.extract_case_features(case_dir)
                    
                    # 儲存
                    features_dict[case] = {
                        'features': case_features,
                        'source_dir': child,
                        'num_views': len([k for k in case_features.keys() if k in ['RCC', 'LCC', 'RMLO', 'LMLO']])
                    }
                    
                    total_success += 1
                    
                except Exception as e:
                    print(f"❌ 提取 {case} 特徵失敗: {str(e)}")
                    total_failed += 1
        
        # 儲存特徵
        print(f"\n💾 儲存特徵到 {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(features_dict, f)
        
        print(f"\n{'='*60}")
        print(f"✅ 完成！")
        print(f"成功: {total_success} 個案例")
        print(f"失敗: {total_failed} 個案例")
        print(f"{'='*60}")
        
        return features_dict


# ==================== 使用範例 ====================

if __name__ == "__main__":
    # 1. 初始化特徵提取器
    extractor = ImageFeatureExtractor()
    
    # 2. 批次提取所有案例的特徵
    features_dict = extractor.batch_extract_all_cases(
        preprocessed_dir='preprocessed_images/',
        output_file='mammography_features.pkl'
    )
    
    # 3. 查看結果
    print("\n" + "="*60)
    print("📊 特徵提取統計:")
    print(f"總案例數: {len(features_dict)}")
    
    if len(features_dict) > 0:
        first_case_id = list(features_dict.keys())[0]
        first_case_data = features_dict[first_case_id]
        first_features = first_case_data['features']
        
        print(f"\n範例案例: {first_case_id}")
        print(f"影像數量: {first_case_data['num_views']}")
        print(f"\n可用特徵類型:")
        for key in first_features.keys():
            print(f"  - {key}: {first_features[key].shape}")
        
        print(f"\n特徵向量範例 (avg_all 前10維):")
        print(f"  {first_features['avg_all'][:10]}")
        
        # 統計
        from collections import Counter
        source_counter = Counter([info['source_dir'] for info in features_dict.values()])
        print(f"\n各時間點案例數:")
        for source, count in sorted(source_counter.items()):
            print(f"  {source}: {count} 個案例")
    
    print("="*60)