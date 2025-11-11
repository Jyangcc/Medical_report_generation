import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pickle
import open_clip
from typing import List, Dict, Any
import tempfile # 處理暫存檔案


from inference_sdk import InferenceHTTPClient

class LesionDetector:
    """
    [V3 版 - API 呼叫]
    使用 Roboflow HTTP API 進行病灶檢測
    """
    def __init__(self):
        """
        初始化 Roboflow API 客戶端
        """
        print(f"🔄 初始化 Roboflow API 客戶端...")
        
        # ！！！從環境變數讀取 API Key！！！
        # 絕對不要把 Key 寫死在這裡
        self.api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not self.api_key:
            print("="*80)
            print("❌ 錯誤：找不到 ROBOFLOW_API_KEY 環境變數")
            print("請先設定: export ROBOFLOW_API_KEY='your_roboflow_key_here'")
            print("="*80)
            raise ValueError("ROBOFLOW_API_KEY not set")
            
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=self.api_key
        )
        
        # 這是你找到的論文所使用的模型 ID
        self.model_id = "breast-cancer-jtuaz/1"
        
        print(f"✅ Roboflow API 客戶端初始化完成 (模型: {self.model_id})")

    def detect(self, image_pil: Image.Image) -> List[Dict[str, Any]]:
        """
        對單張 PIL 影像進行病灶檢測 (透過 API)
        
        Returns:
            lesions: 檢測到的病灶列表，包含 'bbox' (x1, y1, x2, y2) 和 'conf'
        """
        
        # 1. 將 PIL 影像儲存到一個暫存檔案
        # InferenceHTTPClient.infer() 需要一個檔案路徑
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as temp_file:
            # 確保影像是 RGB (如果它是灰階 'L' 的話)
            image_rgb = image_pil.convert("RGB")
            image_rgb.save(temp_file.name, format="JPEG")
            
            # 2. 呼叫 Roboflow API
            try:
                result = self.client.infer(temp_file.name, model_id=self.model_id)
            except Exception as e:
                print(f"❌ Roboflow API 呼叫失敗: {e}")
                return [] # 返回空列表

        # 3. 解析 API 回傳的 JSON 結果
        lesions = []
        img_w, img_h = image_pil.size # 取得原始影像尺寸

        for pred in result.get('predictions', []):
            # ... (取得 x_center, y_center, width, height 的代碼不變) ...
            x_center = pred['x']
            y_center = pred['y']
            width = pred['width']
            height = pred['height']
            confidence = pred['confidence']

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # ！！！新增：計算相對位置 (0.0 ~ 1.0)！！！
            rel_x = x_center / img_w
            rel_y = y_center / img_h

            lesions.append({
                'bbox': [x1, y1, x2, y2],
                'conf': confidence,
                'rel_center': (rel_x, rel_y) # <--- 加入這個新資訊
            })

        return lesions

class ImageFeatureExtractor:
    """
    [V3 版 - API 檢測]
    使用 BiomedCLIP 提取「全域特徵」和「局部病灶特徵」
    """
    
    def __init__(self, 
                 clip_model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        """初始化模型"""
        
        # 1. ！！！初始化 API 版本的病灶檢測器！！！
        #    (不再需要 detector_model_path)
        self.detector = LesionDetector()
        
        # 2. 初始化 CLIP 特徵提取器 (這部分不變)
        print("🔄 載入 BiomedCLIP 模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(clip_model_name)
        
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
        print(f"✅ BiomedCLIP 模型載入完成 (device: {self.device})")
    
    def extract_clip_feature(self, image_pil: Image.Image) -> np.ndarray:
        """提取單張 PIL 影像 (全圖或裁剪) 的特徵向量 (不變)"""
        
        image_tensor = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().squeeze()
    
    def _convert_npy_to_pil(self, image_array: np.ndarray) -> Image.Image:
        """將 NPY 陣列轉換為適合檢測和提取的 PIL 影像 (不變)"""
        if len(image_array.shape) == 3:
            image_array = image_array.squeeze()
        
        if image_array.dtype != np.uint8:
            image_array = (image_array / image_array.max() * 255).astype(np.uint8)
        
        # 轉換為 PIL (YOLO 需要 RGB, CLIP 也需要 RGB)
        return Image.fromarray(image_array).convert('RGB')

    
    def extract_case_features(self, case_dir: str) -> Dict[str, Any]:
        """
        [V3 版] 提取一個案例的所有影像特徵 (全域 + 局部) (不變)
        """
        view_mapping = {
            'I0000000.npy': 'RCC',   # Right CC
            'I0000001.npy': 'LCC',   # Left CC
            'I0000002.npy': 'RMLO',  # Right MLO
            'I0000003.npy': 'LMLO'   # Left MLO
        }
        
        features = {}
        global_features_cache = {} # 用於計算平均
        
        for fname, view_name in view_mapping.items():
            file_path = os.path.join(case_dir, fname)
            
            if not os.path.exists(file_path):
                continue
            
            try:
                # 1. 載入並轉換影像
                image_array = np.load(file_path)
                image_pil = self._convert_npy_to_pil(image_array)
                
                # 2. 提取全域特徵 (用於密度 RAG)
                global_feature = self.extract_clip_feature(image_pil)
                global_features_cache[view_name] = global_feature
                
                # 3. 檢測病灶 (!!!現在會呼叫 API!!!)
                detected_lesions = self.detector.detect(image_pil)
                
                processed_lesions = []
                # 4. 提取每個病灶的局部特徵 (用於病灶 RAG)
                for lesion in detected_lesions:
                    bbox = lesion['bbox']
                    
                    # 裁剪病灶區域
                    img_crop_pil = image_pil.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                    
                    # 提取 ROI 特徵
                    roi_feature = self.extract_clip_feature(img_crop_pil)
                    
                    processed_lesions.append({
                        'bbox': bbox, # 已經是 list
                        'conf': lesion['conf'],
                        'roi_feature': roi_feature
                    })
                
                # 5. 儲存該影像的所有資訊
                features[view_name] = {
                    'global_feature': global_feature,
                    'lesions': processed_lesions
                }

            except Exception as e:
                print(f"⚠️  處理 {file_path} 時出錯: {str(e)}")
                continue
        
        if len(features) == 0:
            raise ValueError(f"案例 {case_dir} 沒有有效的影像!")
        
        # 6. 計算不同組合的平均「全域」特徵 (不變)
        result = features.copy()
        
        all_global = list(global_features_cache.values())
        if all_global:
            avg_all = np.mean(all_global, axis=0)
            result['avg_all_global'] = avg_all / np.linalg.norm(avg_all)
        
        if 'RCC' in global_features_cache and 'RMLO' in global_features_cache:
            avg_right = np.mean([global_features_cache['RCC'], global_features_cache['RMLO']], axis=0)
            result['avg_right_global'] = avg_right / np.linalg.norm(avg_right)
        
        if 'LCC' in global_features_cache and 'LMLO' in global_features_cache:
            avg_left = np.mean([global_features_cache['LCC'], global_features_cache['LMLO']], axis=0)
            result['avg_left_global'] = avg_left / np.linalg.norm(avg_left)
            
        return result
    
    # 
    # batch_extract_all_cases() (在 V2 中) 保持不變
    #
    def batch_extract_all_cases(self, preprocessed_dir='preprocessed_images/', 
                                output_file='mammography_features_v2.pkl'):
        """[V3 版] 批次處理所有案例 (邏輯不變)"""
        features_dict = {}
        child_dirs = ['20230721_1st', '20230728_2nd', '20230804_3rd']
        
        print("\n🚀 開始 V3 批次提取特徵 (API 檢測 + 局部特徵)...")
        
        total_success = 0
        total_failed = 0
        
        for child in child_dirs:
            child_path = os.path.join(preprocessed_dir, child)
            
            if not os.path.exists(child_path):
                print(f"⚠️  找不到資料夾: {child_path}")
                continue
            
            cases = sorted([c for c in os.listdir(child_path) if os.path.isdir(os.path.join(child_path, c))])
            
            # ！！！DEBUGGING: 先只跑 5 個案例測試 API！！！
            # print("--- ⚠️  警告: 正在以 5 個案例進行 API 測試 ---")
            # cases = cases[:5] 
            
            for case in tqdm(cases, desc=f"提取 {child} 特徵"):
                case_dir = os.path.join(child_path, case)
                
                try:
                    case_features = self.extract_case_features(case_dir)
                    
                    # 計算這個案例總共找到多少病灶
                    total_lesions = 0
                    for view in ['RCC', 'LCC', 'RMLO', 'LMLO']:
                        if view in case_features:
                            total_lesions += len(case_features[view]['lesions'])
                    
                    features_dict[case] = {
                        'features': case_features,
                        'source_dir': child,
                        'total_lesions_found': total_lesions
                    }
                    total_success += 1
                    
                except Exception as e:
                    print(f"❌ 提取 {case} 特徵失敗: {str(e)}")
                    total_failed += 1
        
        # 儲存特徵
        print(f"\n💾 儲存 V2 特徵到 {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(features_dict, f)
        
        print(f"\n{'='*60}")
        print(f"✅ V2 特徵提取完成！")
        print(f"成功: {total_success} 個案例")
        print(f"失敗: {total_failed} 個案例")
        print(f"{'='*60}")
        
        return features_dict