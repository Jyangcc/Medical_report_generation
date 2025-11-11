import os
from anthropic import Anthropic
import base64
from typing import List, Dict
import numpy as np
from PIL import Image
import re

# 導入之前寫的模組
from Get_Report import MammographyDataLoader, MammographyReport
from faiss_retrieval import MammographyRetrievalSystem
from extract_features import ImageFeatureExtractor

class MammographyRAGSystem:
    """
    完整的乳房 X 光 RAG 報告生成系統
    """
    
    def __init__(self, 
                 anthropic_api_key=None,
                 features_file='mammography_features.pkl',
                 faiss_index='mammography_faiss.index',
                 faiss_metadata='mammography_metadata.pkl',
                 reports_dir='Kang_Ning_General_Hospital/'):
        """
        初始化 RAG 系統
        
        Args:
            anthropic_api_key: Claude API key (如果不提供，從環境變數讀取)
        """
        # 1. 初始化 Claude API
        if anthropic_api_key is None:
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not anthropic_api_key:
            print("⚠️  請設定 ANTHROPIC_API_KEY 環境變數")
            print("   export ANTHROPIC_API_KEY='your-api-key'")
        
        self.client = Anthropic(api_key=anthropic_api_key)
        
        # 2. 載入檢索系統
        print("🔄 初始化檢索系統...")
        self.retrieval_system = MammographyRetrievalSystem(features_file)
        self.retrieval_system.load_index(faiss_index, faiss_metadata)
        
        # 3. 載入報告資料
        print("🔄 載入報告資料...")
        self.report_loader = MammographyDataLoader(reports_dir)
        self.reports = self.report_loader.load_all_reports()
        
        # 建立 case_id 到 report 的映射
        self.reports_dict = {report.case_id: report for report in self.reports}
        
        print("✅ RAG 系統初始化完成！")

    def _image_to_base64(self, image_array: np.ndarray) -> str:
        """將 NPY 影像陣列轉換為 Base64 字串 (PNG 格式)"""
        if len(image_array.shape) == 3:
            image_array = image_array.squeeze()
        
        # 影像正規化到 0-255 (BiomedCLIP 特徵提取已將影像轉為 uint8)
        # 假設你的 .npy 已經是 0-255 或類似的灰度圖
        if image_array.dtype != np.uint8:
            image_array = (image_array / image_array.max() * 255).astype(np.uint8)
            
        image_pil = Image.fromarray(image_array, mode='L') # 假設是灰階
        
        # 轉換為 PNG 格式並 Base64 編碼
        import io
        byte_arr = io.BytesIO()
        image_pil.save(byte_arr, format='PNG')
        encoded_string = base64.b64encode(byte_arr.getvalue()).decode('utf-8')
        return encoded_string
    
    def _get_query_images(self, query_case_id: str) -> Dict[str, str]:
        """讀取查詢案例的 4 張影像並轉換為 Base64"""
        
        # 你的影像路徑結構：preprocessed_images/{child_dir}/{case_id}/{I000000X.npy}
        # 我們需要找到 {child_dir}。從 features_dict 取得。
        case_data = self.retrieval_system.features_dict.get(query_case_id)
        if not case_data:
            print(f"❌ 查詢案例 {query_case_id} 找不到特徵數據，無法讀取影像。")
            return {}
            
        source_dir = case_data['source_dir'] # 例如 '20230721_1st'
        
        image_views = {
            'RCC': 'I0000000.npy', 'LCC': 'I0000001.npy',
            'RMLO': 'I0000002.npy', 'LMLO': 'I0000003.npy'
        }
        
        base_dir = 'preprocessed_images' # 假設這是你的影像根目錄
        base64_images = {}
        
        for view_name, file_name in image_views.items():
            file_path = os.path.join(base_dir, source_dir, query_case_id, file_name)
            if os.path.exists(file_path):
                try:
                    image_array = np.load(file_path)
                    base64_images[view_name] = self._image_to_base64(image_array)
                except Exception as e:
                    print(f"⚠️  讀取影像 {file_path} 失敗: {str(e)}")
        
        return base64_images
        
    def generate_prompt(self, similar_cases: Dict[str, List[Dict]], 
                       query_case_id: str) -> str:
        """
        [教授修改版]
        生成給 LLM 的 System Prompt，強制執行影像優先的分析流程。
        """
        newline = '\n'
        
        # 1. (新) 建立一個精簡的參考案例庫 (僅供風格參考)
        # 我們不希望模型從 RAG 中複製病灶描述，因為 RAG 是錯的。
        reference_text = "\n--- Reference Cases for Style and Terminology (DO NOT COPY FINDINGS) ---\n"
        
        # 只取 "Overall" 案例作為風格參考，避免 L/R 案例的錯誤病灶描述汙染結果
        for i, case in enumerate(similar_cases.get('all', []), 1):
            report = self.reports_dict.get(case['case_id'])
            report_text = report.raw_text if report else "Report not found"
            
            # (新) 我們只提取報告的 "結構" 和 "密度" 描述，過濾掉關鍵病灶
            if report:
                density_match = re.search(r'1\..+\.', report_text) # 抓第一點
                birads_match = re.search(r'BI-RADS Category[^\.]+\.', report_text) # 抓 BI-RADS
                density = density_match.group(0) if density_match else "[Density description]"
                birads = birads_match.group(0) if birads_match else "[BI-RADS description]"
                reference_text += f"\n- Ref {i} Style: {density} ... {birads}\n"
            else:
                reference_text += f"\n- Ref {i} (Style reference only)\n"

        
        # 2. (新) 重新設計 Prompt，採用「兩階段思考」
        prompt = f"""You are an expert radiologist. Your task is to analyze the provided 4 mammography images (RCC, LCC, RMLO, LMLO) for patient {query_case_id}.

**CRITICAL INSTRUCTIONS:**
Your analysis MUST follow this two-step process:

**Step 1: Image-First Analysis (Internal Monologue)**
First, meticulously examine the 4 images provided. Pay extremely close attention to:
- Breast density.
- Any asymmetries, masses, calcifications, or architectural distortions.
- Note the location (e.g., Lt. breast, UOQ, posterior 1/3) and characteristics (size, shape, margins) of ANY findings.

**Step 2: Report Generation (Final Output)**
After your visual analysis is complete, generate the final report.
- Your report **MUST** be based **100% on your visual findings** from Step 1.
- The reference cases provided below are **ONLY** for understanding the desired *formatting, terminology, and style*.
- **DO NOT** copy findings (like mass locations or sizes) or the BI-RADS category from the reference cases.
- If you see a suspicious finding in the images, you MUST report it, even if all reference cases are negative.
- If you see *no* findings, report it as BI-RADS 1.
- If you see a finding that requires further evaluation (e.g., a new mass), report it as BI-RADS 0.

---
{reference_text}
---

**MANDATORY FORMAT (Fill this based on your Step 1 Image Analysis):**

**Bilateral screening mammograms**

1. [Your description of breast density based on the images]
2. [Your description of additional findings (if any)]
3. [Your description of key findings (if any), including location, size, and characteristics]

**BI-RADS Category**: [Your category based on images, e.g., 0, 1, 2]

**Comparison**: [State 'Not specified/Unknown' unless comparison info is provided]

---
**ACTION: Perform Step 1 (Analyze Images) and then Step 2 (Generate Report).**
"""
        return prompt
    
    def generate_report(self, 
                       query_case_id: str = None,
                       k: int = 2, # 每個特徵類型只取 k 個，避免重複太多
                       model: str = "claude-sonnet-4-5-20250929", # 使用通用別名，假設已修復
                       temperature: float = 0.3) -> Dict:
        """
        生成報告 (執行分乳檢索)
        ... (略去參數說明)
        """
        
        if query_case_id is None:
             raise ValueError("需要提供 query_case_id")
             
        # 1. 執行三次檢索：整體、右乳、左乳
        print(f"\n🔍 檢索與 {query_case_id} 相似的案例 (分乳)...")
        
        # 總體相似案例
        all_cases = self.retrieval_system.search_by_case_id(
            query_case_id, k=k, feature_type='avg_all'
        )
        
        # 右乳相似案例
        right_cases = self.retrieval_system.search_by_case_id(
            query_case_id, k=k, feature_type='avg_right'
        )
        
        # 左乳相似案例
        left_cases = self.retrieval_system.search_by_case_id(
            query_case_id, k=k, feature_type='avg_left'
        )

        # 彙整相似案例 (使用集合去重)
        unique_case_ids = set()
        for case in all_cases + right_cases + left_cases:
            unique_case_ids.add(case['case_id'])
            
        # 重新組成列表 (這裡可以優化排序，但先以去重為主)
        # 為了傳輸給 Prompt，我們只需要一個包含所有信息的列表
        # 這裡我們只傳入 right_cases 和 left_cases，總體案例可能導致混亂
        
        # 建立一個結構化的相似案例字典
        similar_cases = {
            'all': all_cases,
            'right': right_cases,
            'left': left_cases,
        }
        
        print(f"✅ 找到 {len(unique_case_ids)} 個獨立相似案例")
        
        # 2. 載入並轉換查詢影像
        query_images_base64 = self._get_query_images(query_case_id)
        
        # 3. 生成 Prompt (使用結構化的相似案例)
        # (新) 這裡的 system_prompt 是我們的主要指令
        system_prompt = self.generate_prompt(similar_cases, query_case_id)
        
        # 4. 組織 messages 傳給 Claude API (包含影像)
        print(f"\n🤖 呼叫 Claude API 生成報告...")
        
        content_list = []
        
        # b. 加入所有 Base64 影像
        for view, base64_data in query_images_base64.items():
            print(f"   - 加入影像: {view}")
            content_list.append({
                "type": "image",
                "source": { ... } # (同你原來的)
            })
            
        # c. (新) 加入一個簡短的、觸發動作的文字
        content_list.append({
            "type": "text",
            "text": "Please analyze these images and generate the mammography report based on my instructions."
        })
        
        try:
            # (新) 這裡的 system_prompt 是我們的主要指令
            system_prompt = self.generate_prompt(similar_cases, query_case_id)
            
            # (新) 組織 messages，影像和一個簡短的觸發詞
            content_list = []
            
            # b. 加入所有 Base64 影像
            for view, base64_data in query_images_base64.items():
                print(f"   - 加入影像: {view}")
                content_list.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_data
                    }
                })
                
            # c. (新) 加入一個簡短的、觸發動作的文字
            content_list.append({
                "type": "text",
                "text": "Please analyze these images and generate the mammography report based on my instructions."
            })

            # --- 確保你的 API 呼叫是這樣的 ---
            message = self.client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=temperature,
                system=system_prompt, # <-- 這裡必須是 system_prompt (str)
                messages=[
                    {
                        "role": "user",
                        "content": content_list # <-- 這裡必須是 content_list (list)
                    }
                ]
            )
            # --- 檢查結束 ---

            generated_report = message.content[0].text
            
            print("✅ 報告生成完成！")
            
            # 4. 整理結果
            result = {
                'query_case_id': query_case_id,
                'generated_report': generated_report,
                'similar_cases': similar_cases,
                'prompt': system_prompt, # <--- 確保這裡也是 system_prompt
                'model': model,
                'api_usage': {
                    'input_tokens': message.usage.input_tokens,
                    'output_tokens': message.usage.output_tokens
                }
            }
            
            return result
            
        except Exception as e:
            print(f"❌ API 呼叫失敗: {str(e)}")
            raise
    
    def evaluate_report(self, query_case_id: str, k: int = 3):
        """
        評估模式：生成報告並與真實報告對比
        
        Args:
            query_case_id: 查詢案例 ID
            k: 檢索相似案例數量
        """
        # 1. 生成報告
        result = self.generate_report(query_case_id=query_case_id, k=k)
        
        # 2. 取得真實報告
        possible_ids = [
            query_case_id,
            query_case_id.replace('MAMO_DEID_', ''),
            f"MAMO_DEID_{query_case_id}"
        ]
        
        ground_truth_report = None
        for pid in possible_ids:
            if pid in self.reports_dict:
                ground_truth_report = self.reports_dict[pid].raw_text
                break
        
        # 3. 顯示結果
        print("\n" + "="*80)
        print("評估結果")
        print("="*80)
        
        print(f"\n📋 真實報告 ({query_case_id}):")
        print("-" * 80)
        print(ground_truth_report if ground_truth_report else "找不到真實報告")
        
        print(f"\n🤖 生成報告:")
        print("-" * 80)
        print(result['generated_report'])
        
        print(f"\n📊 相似案例:")
        print("-" * 80)
        
        # 取得結構化的字典
        similar_cases_dict = result['similar_cases']
        
        print("  --- 右乳相似 (Right Breast) ---")
        if similar_cases_dict.get('right'):
            for case in similar_cases_dict['right']:
                print(f"    - {case['case_id']} (相似度: {case['similarity']:.3f})")
        else:
            print("    (無)")

        print("\n  --- 左乳相似 (Left Breast) ---")
        if similar_cases_dict.get('left'):
            for case in similar_cases_dict['left']:
                print(f"    - {case['case_id']} (相似度: {case['similarity']:.3f})")
        else:
            print("    (無)")
            
        print("\n  --- 整體相似 (Overall) ---")
        if similar_cases_dict.get('all'):
            for case in similar_cases_dict['all']:
                print(f"    - {case['case_id']} (相似度: {case['similarity']:.3f})")
        else:
            print("    (無)")
        print(f"\n💰 API 使用:")
        print("-" * 80)
        print(f"  Input tokens: {result['api_usage']['input_tokens']}")
        print(f"  Output tokens: {result['api_usage']['output_tokens']}")
        print(f"  估計成本: ${(result['api_usage']['input_tokens'] * 0.003 + result['api_usage']['output_tokens'] * 0.015) / 1000:.4f}")
        
        print("="*80)
        
        return result


# ==================== 使用範例 ====================

if __name__ == "__main__":
    # 設定你的 API key（或在終端機執行：export ANTHROPIC_API_KEY='your-key'）
    # os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    
    print("="*80)
    print("乳房 X 光 RAG 報告生成系統")
    print("="*80)
    
    # 1. 初始化系統
    rag_system = MammographyRAGSystem(
        features_file='mammography_features.pkl',
        faiss_index='mammography_faiss.index',
        faiss_metadata='mammography_metadata.pkl',
        reports_dir='Kang_Ning_General_Hospital/'
    )
    
    # 2. 測試：用已有案例生成報告
    test_case_id = "MAMO_DEID_20230721_-00011"
    
    result = rag_system.evaluate_report(
        query_case_id=test_case_id,
        k=3  # 檢索 3 個最相似案例
    )
    
    # 3. 儲存結果
    import json
    with open('generated_report_sample.json', 'w', encoding='utf-8') as f:
        # 準備一個字典來儲存分類後的相似案例
        cleaned_similar_cases = {}
        
        # 迭代字典的鍵 (key) 和值 (case_list)
        for key, case_list in result['similar_cases'].items():
            cleaned_similar_cases[key] = [
                {
                    'case_id': c['case_id'],
                    'similarity': float(c['similarity'])
                } for c in case_list # 迭代 'all', 'right', 'left' 各自的列表
            ]

        json.dump({
            'query_case_id': result['query_case_id'],
            'generated_report': result['generated_report'],
            'similar_cases': cleaned_similar_cases # <--- 儲存這個新的、乾淨的字典
        }, f, indent=2, ensure_ascii=False)
    
    print("\n💾 結果已儲存到 generated_report_sample.json")