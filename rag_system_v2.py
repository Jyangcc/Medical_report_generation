import os
import numpy as np
import pickle
import faiss
import base64
from typing import List, Dict, Any
from PIL import Image
import re
import io

# 導入我們現有的模組
from Get_Report import MammographyDataLoader, MammographyReport
from detection_and_feature_extractor import ImageFeatureExtractor # 導入 V3 (API) 版本的提取器
from anthropic import Anthropic

# ==============================================================================
# 階段 2.1: V2 檢索系統 (FAISS 索引客戶端)
# ==============================================================================

class MammographyRetrievalSystemV2:
    """
    [V2 版]
    載入並管理所有 V2 索引 (全域 + 病灶)
    """
    def __init__(self, features_v2_file='mammography_features_v2.pkl'):
        print("🔄 初始化 V2 檢索系統...")
        
        # 1. 載入 V2 特徵 (我們需要它來讀取影像)
        print(f"🔄 載入 V2 特徵資料: {features_v2_file}")
        with open(features_v2_file, 'rb') as f:
            self.features_dict_v2 = pickle.load(f)
        
        # 2. 載入全域索引
        self.global_indices = {}
        self.global_maps = {}
        for feature_type in ['avg_all_global', 'avg_right_global', 'avg_left_global']:
            index_file = f"faiss_global_{feature_type}.index"
            map_file = f"faiss_global_{feature_type}_map.pkl"
            if os.path.exists(index_file) and os.path.exists(map_file):
                print(f"  - 載入全域索引: {index_file}")
                self.global_indices[feature_type] = faiss.read_index(index_file)
                with open(map_file, 'rb') as f:
                    self.global_maps[feature_type] = pickle.load(f)
            else:
                print(f"⚠️  警告: 找不到全域索引 {index_file}，將無法進行密度 RAG")
                
        # 3. 載入病灶 (ROI) 索引
        self.lesion_index = None
        self.lesion_map = None
        lesion_index_file = "faiss_lesion_roi.index"
        lesion_map_file = "faiss_lesion_roi_map.pkl"
        if os.path.exists(lesion_index_file) and os.path.exists(lesion_map_file):
            print(f"  - 載入病灶索引: {lesion_index_file} (共 {faiss.read_index(lesion_index_file).ntotal} 個病灶)")
            self.lesion_index = faiss.read_index(lesion_index_file)
            with open(lesion_map_file, 'rb') as f:
                self.lesion_map = pickle.load(f)
        else:
            print("⚠️  警告: 找不到病灶索引，將無法進行病灶 RAG")
        
        print("✅ V2 檢索系統載入完成")

    def search_global(self, query_feature: np.ndarray, k: int, feature_type: str = 'avg_all_global') -> List[Dict[str, Any]]:
        """
        搜尋「全域」索引 (用於密度/風格)
        """
        if feature_type not in self.global_indices:
            print(f"❌ 錯誤: 全域索引 {feature_type} 未載入")
            return []
            
        index = self.global_indices[feature_type]
        id_map = self.global_maps[feature_type]
        
        # 準備查詢向量
        query_feature = query_feature.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_feature) # 確保正規化
        
        similarities, indices = index.search(query_feature, k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1: continue # FAISS 可能返回 -1
            results.append({
                'case_id': id_map[idx],
                'similarity': float(sim)
            })
        return results

    def search_lesion(self, query_roi_feature: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """
        搜尋「病灶」索引 (用於病灶描述)
        """
        if self.lesion_index is None:
            print("❌ 錯誤: 病灶索引未載入")
            return []
        
        query_roi_feature = query_roi_feature.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_roi_feature)
        
        similarities, indices = self.lesion_index.search(query_roi_feature, k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1: continue
            # 返回儲存在 map 中的完整元數據
            metadata = self.lesion_map[idx].copy() # 複製一份
            metadata['similarity'] = float(sim)
            results.append(metadata)
        return results

    def get_case_images_from_v2_features(self, case_id: str) -> Dict[str, Image.Image]:
        """
        輔助函式：從 V2 特徵庫中讀取原始影像 (NPY -> PIL)
        """
        case_data = self.features_dict_v2.get(case_id)
        if not case_data:
            print(f"❌ 錯誤: 案例 {case_id} 不在 V2 特徵庫中")
            return {}
            
        source_dir = case_data['source_dir']
        base_dir = 'preprocessed_images'
        
        image_views = {
            'RCC': 'I0000000.npy', 'LCC': 'I0000001.npy',
            'RMLO': 'I0000002.npy', 'LMLO': 'I0000003.npy'
        }
        
        pil_images = {}
        for view_name, file_name in image_views.items():
            file_path = os.path.join(base_dir, source_dir, case_id, file_name)
            if os.path.exists(file_path):
                try:
                    image_array = np.load(file_path)
                    # 轉換為 PIL (需要 RGB 以便 VLM 讀取)
                    if len(image_array.shape) == 3:
                        image_array = image_array.squeeze()
                    if image_array.dtype != np.uint8:
                        image_array = (image_array / image_array.max() * 255).astype(np.uint8)
                    pil_images[view_name] = Image.fromarray(image_array).convert('RGB')
                except Exception as e:
                    print(f"⚠️  讀取影像 {file_path} 失敗: {str(e)}")
        return pil_images


# ==============================================================================
# 階段 2.2: V2 RAG 系統 (多階段生成)
# ==============================================================================

class MammographyRAGSystemV2:
    """
    [V2 版]
    結合 V3 特徵提取器 和 V2 檢索系統，執行多階段 RAG
    """
    def __init__(self, reports_dir='Kang_Ning_General_Hospital/'):
        print("="*80)
        print("🚀 初始化 V2 RAG 系統 (多階段生成)...")
        
        # 1. 初始化 Anthropic API
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY 未設定")
        self.client = Anthropic(api_key=self.anthropic_api_key)
        
        # 2. 實例化 V3 特徵提取器 (API 版本)
        self.feature_extractor = ImageFeatureExtractor()
        
        # 3. 實例化 V2 檢索系統 (FAISS 客戶端)
        self.retrieval_system = MammographyRetrievalSystemV2()
        
        # 4. 載入報告資料庫 (用於 RAG 檢索)
        print("🔄 載入報告資料庫...")
        self.report_loader = MammographyDataLoader(reports_dir)
        self.reports = self.report_loader.load_all_reports()
        self.reports_dict = {report.case_id: report for report in self.reports}
        print(f"✅ 報告資料庫載入完成 (共 {len(self.reports_dict)} 份報告)")
        
        print("✅ V2 RAG 系統初始化完成！")

    def _image_to_base64(self, image_pil: Image.Image) -> str:
        """[V3 修正版] 將 PIL 影像轉換為 Base64 (使用 JPEG 壓縮)"""
        byte_arr = io.BytesIO()
        
        # 關鍵修改：
        # 1. 確保影像是 RGB (JPEG 必須)
        image_rgb = image_pil.convert('RGB')
        # 2. 使用 'JPEG' 格式
        # 3. 設定 quality=90，在品質和檔案大小間取得平衡
        image_rgb.save(byte_arr, format='JPEG', quality=90)
        
        encoded_string = base64.b64encode(byte_arr.getvalue()).decode('utf-8')
        
        # ！！！新增檢查！！！
        if len(encoded_string) * 0.75 > 5 * 1024 * 1024: # 估算 Base64 解碼後的大小
             # 如果還是太大，就用更低的品質重試
             byte_arr = io.BytesIO()
             image_rgb.save(byte_arr, format='JPEG', quality=75) # 75%
             encoded_string = base64.b64encode(byte_arr.getvalue()).decode('utf-8')
             
        return encoded_string

    def _generate_v2_prompt(self, 
                            query_case_id: str,
                            density_rag_reports: List[Dict[str, Any]],
                            lesion_rag_reports: List[Dict[str, Any]],
                            detected_lesions_count: int
                            ) -> str:
        """
        [V2 版]
        建構多階段 RAG 的 System Prompt
        """
        newline = '\n'
        
        # --- 密度 RAG 部分 ---
        density_prompt = "--- 密度/風格參考 (Density/Style References) ---\n"
        density_prompt += "指示: 僅使用此處的報告來決定「乳房密度」的措辭和整份報告的「格式」。\n"
        for i, rag_result in enumerate(density_rag_reports, 1):
            case_id = rag_result['case_id']
            report = self.reports_dict.get(case_id)
            if report:
                # 提取密度和 BI-RADS (如果有的話)
                density_match = re.search(r'1\..+\.', report.raw_text)
                birads_match = re.search(r'BI-RADS Category[^\.]+\.', report.raw_text)
                density = density_match.group(0) if density_match else "[密度描述]"
                birads = birads_match.group(0) if birads_match else "[BI-RADS 結論]"
                density_prompt += f"- 風格 {i} ({case_id}): {density.strip()} ... {birads.strip()}\n"
            
        # --- 病灶 RAG 部分 ---
        lesion_prompt = "--- 病灶分析參考 (Lesion Analysis References) ---\n"
        if detected_lesions_count > 0:
            lesion_prompt += f"指示: 我的檢測器在影像中找到了 {detected_lesions_count} 個可疑病灶 (顯示在下面)。\n"
            lesion_prompt += "請使用以下「相似病灶」的報告，來幫助你描述這些新發現的病灶 (例如大小、形狀、邊緣)。\n"
            
            for i, rag_result in enumerate(lesion_rag_reports, 1):
                case_id = rag_result['case_id']
                view = rag_result['view']
                report = self.reports_dict.get(case_id)
                if report:
                    # 我們只顯示報告的「Findings」部分
                    findings_match = re.search(r'(Bilateral screening mammograms.+?)(?=BI-RADS|$)', report.raw_text, re.DOTALL)
                    findings = findings_match.group(1).strip() if findings_match else report.raw_text
                    # 簡化，只取 150 字元
                    findings_snippet = findings.replace(newline, ' ').strip()[:150]
                    lesion_prompt += f"- 病灶 {i} (來自 {case_id}, {view}, 相似度 {rag_result['similarity']:.3f}): \"...{findings_snippet}...\"\n"
        else:
            lesion_prompt += "指示: 我的檢測器在影像中**沒有**找到明顯的病灶。\n"
            lesion_prompt += "請確認影像，如果確實沒有病灶，請參考「密度 RAG」報告來撰寫一份陰性 (BI-RADS 1) 報告。\n"
        

        # --- 最終組合 Prompt ---
        final_prompt = f"""You are an expert radiologist. Your task is to analyze the provided mammography images for patient {query_case_id} and generate a professional report.

**CRITICAL INSTRUCTIONS:**
1.  **IMAGE FIRST:** Your primary source of truth is the provided images. **Trust the images over the text references.**
2.  **DETECTED LESIONS:** I am providing you with **Full-View Images** (RCC, LCC, RMLO, LMLO) and **Cropped Lesion Images** (if any were detected). You MUST describe the findings in the Cropped Lesion Images.
3.  **USE RAG CONTEXT:**
    * Use the **Lesion Analysis References** to help *describe* the detected lesions.
    * Use the **Density/Style References** to help describe the *breast density* and *report format*.
4.  **TASK:** Synthesize ALL information into the mandatory format. If you see a lesion (even if RAG context is negative), report it (BI-RADS 0). If you see NO lesions (even if RAG context mentions one), report it as negative (BI-RADS 1).

{density_prompt}
{lesion_prompt}
---
**MANDATORY FORMAT (Fill this based on YOUR analysis of the images):**

<REPORT_TEXT>
**Bilateral screening mammograms**

1. [Your description of breast density based on Full-View Images]
2. [Your description of additional findings (if any)]
3. [Your description of key findings based on Cropped Lesion Images, e.g., location, size, margins]
</REPORT_TEXT>

<BI_RADS_CATEGORY>
[Your category (0, 1, 2...) based on ALL images. **MUST BE A SINGLE DIGIT**]
</BI_RADS_CATEGORY>

<COMPARISON>
[State 'Not specified/Unknown' unless comparison info is provided]
</COMPARISON>
"""
        return final_prompt

    def run_v2_evaluation(self, query_case_id: str, k_density: int = 3, k_lesion: int = 3):
        """
        [V2 版]
        執行完整的「檢測 -> 雙 RAG -> 多模態生成」流程
        """
        print(f"\n{'='*80}")
        print(f"🚀 開始 V2 評估: {query_case_id}")
        print("="*80)
        
        # 1. 取得真實報告 (用於最後比較)
        ground_truth_report = self.reports_dict.get(query_case_id)
        if not ground_truth_report:
            print(f"❌ 找不到 {query_case_id} 的真實報告，中止評估")
            return
            
        # 2. 提取查詢案例的特徵 (呼叫 V3 API 檢測器)
        print(f"🔄 (Step 1/5) 提取 {query_case_id} 的即時特徵 (呼叫 Roboflow API)...")
        # 我們需要原始案例的目錄
        case_data_v2 = self.retrieval_system.features_dict_v2.get(query_case_id)
        if not case_data_v2:
            print(f"❌ 找不到 {query_case_id} 的 V2 特徵 (無法定位影像)，中止評估")
            return
            
        source_dir = case_data_v2['source_dir']
        case_dir_path = os.path.join('preprocessed_images', source_dir, query_case_id)
        
        # 呼叫特徵提取器
        try:
            live_features = self.feature_extractor.extract_case_features(case_dir_path)
            total_lesions = sum(len(v.get('lesions', [])) for k, v in live_features.items() if k in ['RCC', 'LCC', 'RMLO', 'LMLO'])
            print(f"✅ (Step 1/5) 特徵提取完成。檢測到 {total_lesions} 個病灶。")
        except Exception as e:
            print(f"❌ (Step 1/5) 即時特徵提取失敗: {e}")
            return

        # 3. 執行「密度 RAG」
        print(f"🔄 (Step 2/5) 執行「密度 RAG」...")
        density_rag_reports = self.retrieval_system.search_global(
            query_feature=live_features['avg_all_global'],
            k=k_density,
            feature_type='avg_all_global'
        )
        print(f"✅ (Step 2/5) 找到 {len(density_rag_reports)} 個密度相似案例")

        # 4. 執行「病灶 RAG」
        print(f"🔄 (Step 3/5) 執行「病灶 RAG」...")
        all_lesion_rag_reports = []
        # 遍歷所有檢測到的病灶
        for view in ['RCC', 'LCC', 'RMLO', 'LMLO']:
            if view in live_features:
                for lesion in live_features[view]['lesions']:
                    print(f"  - 檢索 {view} 上的病灶 (Conf: {lesion['conf']:.2f})...")
                    lesion_rag_results = self.retrieval_system.search_lesion(
                        query_roi_feature=lesion['roi_feature'],
                        k=k_lesion
                    )
                    all_lesion_rag_reports.extend(lesion_rag_results)
        
        # 去除重複的 (可能多個病灶檢索到同一個案例)
        seen_case_ids = set()
        unique_lesion_rag_reports = []
        for report in all_lesion_rag_reports:
            if report['case_id'] not in seen_case_ids:
                unique_lesion_rag_reports.append(report)
                seen_case_ids.add(report['case_id'])
        print(f"✅ (Step 3/5) 找到 {len(unique_lesion_rag_reports)} 個獨特的相似病灶案例")
        
        # 5. 準備影像 (全幅 + 病灶裁剪)
        print(f"🔄 (Step 4/5) 準備多模態影像 (全幅 + 裁剪)...")
        content_list = []
        
        # 讀取 4 張全幅 PIL 影像
        full_pil_images = self.retrieval_system.get_case_images_from_v2_features(query_case_id)
        
        # a. 加入全幅影像
        for view, pil_img in full_pil_images.items():
            content_list.append({"type": "text", "text": f"--- Full-View Image: {view} ---"})
            content_list.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": self._image_to_base64(pil_img)} # <-- 修正 A
            })
            
        # b. 加入裁剪的病灶影像
        detected_lesions_count = 0
        for view in ['RCC', 'LCC', 'RMLO', 'LMLO']:
            if view in live_features and view in full_pil_images:
                pil_img = full_pil_images[view] # 取得對應的全幅影像
                for lesion in live_features[view]['lesions']:
                    detected_lesions_count += 1
                    bbox = lesion['bbox']
                    # 裁剪 (left, upper, right, lower)
                    cropped_pil = pil_img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                    loc_desc = ""
                    if 'rel_center' in lesion:
                        rx, ry = lesion['rel_center']
                        # 簡單的位置文字描述
                        h_pos = "Left" if rx < 0.33 else ("Right" if rx > 0.66 else "Center")
                        v_pos = "Top" if ry < 0.33 else ("Bottom" if ry > 0.66 else "Middle")
                        loc_desc = f"(Location in full image: {v_pos}-{h_pos}, x={rx:.2f}, y={ry:.2f})"

                    # 將位置描述加到 Prompt 中
                    content_list.append({
                        "type": "text",
                        "text": f"--- Cropped Lesion #{detected_lesions_count} (from {view} view, Conf: {lesion['conf']:.2f}) {loc_desc} ---"
                    })
                    content_list.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": self._image_to_base64(cropped_pil)} # <-- 修正 B
                    })
        print(f"✅ (Step 4/5) 影像準備完成 (4 張全幅, {detected_lesions_count} 張病灶裁剪)")

        # 6. 建構 Prompt
        system_prompt = self._generate_v2_prompt(
            query_case_id=query_case_id,
            density_rag_reports=density_rag_reports,
            lesion_rag_reports=unique_lesion_rag_reports,
            detected_lesions_count=detected_lesions_count
        )
        
        # 7. 呼叫 Claude API
        print(f"🔄 (Step 5/5) 呼叫 Claude API (模型: claude-sonnet-4-5-20250929)...")
        
        # 加入最後的觸發詞
        content_list.append({
            "type": "text",
            "text": "Please analyze all provided images and generate the mammography report based on my system instructions."
        })
        
        try:
            message = self.client.messages.create(
       
                model="claude-sonnet-4-5-20250929", 
                max_tokens=2048,
                temperature=0.1, # 醫療報告需要低溫、高確定性
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": content_list
                    }
                ]
            )
            generated_report = message.content[0].text
            print("✅ (Step 5/5) 報告生成完成！")
            
            # --- 最終結果顯示 ---
            print("\n" + "="*80)
            print(f"🔬 V2 RAG 評估結果: {query_case_id}")
            print("="*80)
            
            print(f"\n📋 真實報告 (Ground Truth):")
            print("-" * 80)
            print(ground_truth_report.raw_text)
            
            print(f"\n🤖 V2 生成報告 (AI Generated):")
            print("-" * 80)
            print(generated_report)
            
            print(f"\n📊 RAG 檢索資訊:")
            print("-" * 80)
            print(f"  檢測到的病灶數: {detected_lesions_count}")
            print("\n  --- 密度 RAG (Top 1) ---")
            if density_rag_reports:
                print(f"  - {density_rag_reports[0]['case_id']} (Sim: {density_rag_reports[0]['similarity']:.3f})")
            
            print("\n  --- 病灶 RAG (Top 1) ---")
            if unique_lesion_rag_reports:
                r = unique_lesion_rag_reports[0]
                print(f"  - {r['case_id']} / {r['view']} (Sim: {r['similarity']:.3f})")

            print(f"\n💰 API 使用:")
            print("-" * 80)
            print(f"  Input tokens: {message.usage.input_tokens}")
            print(f"  Output tokens: {message.usage.output_tokens}")
            print("="*80)
            
            return generated_report

        except Exception as e:
            print(f"❌ (Step 5/5) API 呼叫失敗: {e}")
            raise

# ==================== 執行 V2 評估 ====================
if __name__ == "__main__":
    
    # ！！！注意！！！
    # 執行前，請確保你已經設定了環境變數
    # export ANTHROPIC_API_KEY="..."
    # export ROBOFLOW_API_KEY="..."
    
    # 1. 初始化 V2 RAG 系統
    try:
        rag_system_v2 = MammographyRAGSystemV2(
            reports_dir='Kang_Ning_General_Hospital/'
        )
        
        # 2. 測試那個失敗的案例！
        test_case_id = "MAMO_DEID_20230721_-00009"
        
        # 3. 執行 V2 評估
        rag_system_v2.run_v2_evaluation(
            query_case_id=test_case_id,
            k_density=3,
            k_lesion=3
        )
        
        # 4. (可選) 測試另一個案例，例如你第一個成功的
        # test_case_id_2 = "MAMO_DEID_20230721_-00010"
        # rag_system_v2.run_v2_evaluation(
        #     query_case_id=test_case_id_2,
        #     k_density=3,
        #     k_lesion=3
        # )

    except ValueError as e:
        print(f"\n❌ 系統初始化失敗: {e}")
        print("請檢查你的環境變數設定。")
    except ImportError as e:
        print(f"\n❌ 匯入錯誤: {e}")
        print("請確保 'Get_Report.py' 和 'detection_and_feature_extractor.py' 檔案在同一個資料夾中")