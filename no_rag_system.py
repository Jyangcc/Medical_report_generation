import os
import numpy as np
import base64
import io
import re
import glob
from typing import List, Dict, Any, Optional
from PIL import Image
from anthropic import Anthropic

# 導入必要的模組
from detection_and_feature_extractor import LesionDetector

class MammographyNoRAGSystem:
    """
    [對照組] 無 RAG 的純 VLM 報告生成系統
    特點：
    1. 不載入 FAISS 索引。
    2. 不進行密度或病灶檢索。
    3. 僅依賴 VLM 本身的知識和我們提供的影像(含病灶裁剪)。
    """
    def __init__(self):
        print("="*80)
        print("🚀 初始化 No-RAG 對照組系統...")
        
        # 1. 初始化 Anthropic API
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY 未設定")
        self.client = Anthropic(api_key=self.anthropic_api_key)
        
        # 2. 初始化病灶檢測器 (API 版本)
        # 我們仍然需要它來"看"到細微的病灶並裁剪給 VLM
        self.detector = LesionDetector()
        
        print("✅ No-RAG 系統初始化完成 (僅使用 VLM + 檢測器)")

    def _image_to_base64(self, image_pil: Image.Image) -> str:
        """[V3 修正版] 將 PIL 影像轉換為 Base64 (使用 JPEG 壓縮以符合 5MB 限制)"""
        byte_arr = io.BytesIO()
        image_pil = image_pil.convert('RGB')
        # 使用 JPEG 並設定品質為 90，平衡畫質與檔案大小
        image_pil.save(byte_arr, format='JPEG', quality=90)
        encoded_string = base64.b64encode(byte_arr.getvalue()).decode('utf-8')
        
        # 如果還是太大 (超過 ~5MB)，降低品質重試
        if len(encoded_string) * 0.75 > 5 * 1024 * 1024:
             byte_arr = io.BytesIO()
             image_pil.save(byte_arr, format='JPEG', quality=75)
             encoded_string = base64.b64encode(byte_arr.getvalue()).decode('utf-8')
             
        return encoded_string

    def _load_case_images(self, case_id: str) -> Dict[str, Image.Image]:
        """搜尋並載入案例的 4 張原始影像 (回傳 PIL 格式)"""
        # 使用 glob 找出案例所在的子目錄 (例如 20230721_1st)
        case_paths = glob.glob(os.path.join('preprocessed_images', '*', case_id))
        if not case_paths:
            print(f"❌ 錯誤: 找不到案例 {case_id} 的影像資料夾")
            return {}
        
        case_dir = case_paths[0]
        image_views = {
            'RCC': 'I0000000.npy', 'LCC': 'I0000001.npy',
            'RMLO': 'I0000002.npy', 'LMLO': 'I0000003.npy'
        }
        
        pil_images = {}
        for view_name, file_name in image_views.items():
            file_path = os.path.join(case_dir, file_name)
            if os.path.exists(file_path):
                try:
                    image_array = np.load(file_path)
                    # 轉換為 PIL RGB (VLM 需要)
                    if len(image_array.shape) == 3:
                        image_array = image_array.squeeze()
                    if image_array.dtype != np.uint8:
                        image_array = (image_array / image_array.max() * 255).astype(np.uint8)
                    pil_images[view_name] = Image.fromarray(image_array).convert('RGB')
                except Exception as e:
                    print(f"⚠️  讀取影像 {file_path} 失敗: {str(e)}")
        return pil_images

    def _generate_no_rag_prompt(self, case_id: str, detected_lesions_count: int) -> str:
        """生成不包含任何參考案例的 Prompt"""
        
        lesion_instruction = ""
        if detected_lesions_count > 0:
            lesion_instruction = f"My detector has identified {detected_lesions_count} potential lesion(s), shown in the 'Cropped Lesion' images. Please carefully evaluate these regions."
        else:
            lesion_instruction = "My detector did NOT find any obvious lesions. Please double-check the full images to confirm if it's truly negative."

        return f"""You are an expert radiologist. Analyze the provided mammography images for patient {case_id}.

**CRITICAL INSTRUCTIONS (NO-RAG MODE):**
1.  **SOLE SOURCE OF TRUTH:** You have NO external reference cases. You must rely **ONLY** on your medical knowledge to analyze the provided images (Full Views + Cropped Lesions).
2.  **DETECTED LESIONS:** {lesion_instruction}
3.  **REPORTING:** If you see a suspicious finding, describe its location, size, and shape, and assign BI-RADS 0. If the breasts are clear, assign BI-RADS 1.

**MANDATORY FORMAT:**

<REPORT_TEXT>
**Bilateral screening mammograms**

1. [Describe breast density]
2. [Describe any findings or state 'No suspicious masses, calcifications, or architectural distortion.']
</REPORT_TEXT>

<BI_RADS_CATEGORY>
[Single digit ONLY: 0, 1, 2...]
</BI_RADS_CATEGORY>

<COMPARISON>
Not specified/Unknown
</COMPARISON>
"""

    def run_no_rag_evaluation(self, query_case_id: str) -> str:
        """
        執行 No-RAG 評估流程
        """
        print(f"🚀 [No-RAG] 開始評估: {query_case_id}")
        
        # 1. 載入影像
        full_pil_images = self._load_case_images(query_case_id)
        if not full_pil_images:
            return "Error: Images not found"

        # 2. 準備多模態訊息 (含即時檢測)
        content_list = []
        detected_lesions_count = 0
        
        for view in ['RCC', 'LCC', 'RMLO', 'LMLO']:
            if view in full_pil_images:
                pil_img = full_pil_images[view]
                
                # a. 加入全幅影像 (使用修正後的 JPEG Base64)
                content_list.append({"type": "text", "text": f"--- Full-View Image: {view} ---"})
                content_list.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": self._image_to_base64(pil_img)}
                })
                
                # b. 即時檢測並加入裁剪影像
                # 注意：這裡會呼叫 Roboflow API，所以還是需要時間和成本
                try:
                    lesions = self.detector.detect(pil_img)
                    for lesion in lesions:
                        detected_lesions_count += 1
                        bbox = lesion['bbox']
                        cropped_pil = pil_img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                        
                        content_list.append({"type": "text", "text": f"--- Cropped Lesion #{detected_lesions_count} (from {view}, Conf: {lesion['conf']:.2f}) ---"})
                        content_list.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": self._image_to_base64(cropped_pil)}
                        })
                except Exception as e:
                    print(f"⚠️  {view} 檢測失敗: {e}")

        print(f"✅ 影像準備完成 (檢測到 {detected_lesions_count} 個病灶)")

        # 3. 建構 Prompt
        system_prompt = self._generate_no_rag_prompt(query_case_id, detected_lesions_count)
        content_list.append({"type": "text", "text": "Generate the report now based solely on these images."})

        # 4. 呼叫 Claude API
        print(f"🤖 呼叫 Claude API (No-RAG)...")
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929", 
                max_tokens=1024, # No-RAG 報告通常較短
                temperature=0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": content_list}]
            )
            return message.content[0].text
        except Exception as e:
            print(f"❌ API 呼叫失敗: {e}")
            raise