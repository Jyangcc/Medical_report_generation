import json
import os
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm # 載入進度條
from collections import Counter

# ！！！我們現在依賴 V2 RAG 系統！！！
from rag_system_v2 import MammographyRAGSystemV2

TEST_CASES_FILE = 'test_cases.json'



def load_test_ids(test_file=TEST_CASES_FILE) -> List[str]:
    if not os.path.exists(test_file):
        print(f"❌ 錯誤: 找不到測試集檔案 {test_file}")
        print("請先執行 'prepare_split.py'")
        return None
    
    print(f"🔄 載入測試集 ID: {test_file}")
    with open(test_file, 'r') as f:
        test_ids = json.load(f)
    print(f"✅ 成功載入 {len(test_ids)} 個測試集 ID")
    return test_ids

def parse_birads_from_text(report_text: str) -> Optional[int]:
    """
    [V3 修正版]
    從 AI 報告文本中提取 <BI_RADS_CATEGORY> 標籤中的數字
    """
    if not isinstance(report_text, str):
        return None
        
    # 1. 優先嘗試抓取 XML 標籤
    # re.DOTALL 讓 . 可以匹配換行符
    match = re.search(r'<BI_RADS_CATEGORY>(.*?)</BI_RADS_CATEGORY>', report_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        text_inside = match.group(1).strip()
        # 從標籤內的文字中再抓數字
        digit_match = re.search(r'(\d)', text_inside)
        if digit_match:
            try:
                return int(digit_match.group(1))
            except:
                pass # 繼續往下

    # 2. 如果 XML 標籤失敗 (備用方案)，嘗試舊的 regex
    match_fallback = re.search(r'(?:BI-RADS|Category)[\s:]*(\d)', report_text, re.IGNORECASE)
    if match_fallback:
        try:
            return int(match_fallback.group(1))
        except:
            return None
            
    return None # 真的找不到了

def run_evaluation():
    print("="*80)
    print("🚀 開始在「測試集」上執行 V2 RAG 系統評估...")
    print("="*80)
    
    # 1. 載入測試案例 ID
    test_case_ids = load_test_ids()
    if not test_case_ids:
        return

    # 2. 初始化 V2 RAG 系統
    #    (它會自動載入 *只包含訓練資料* 的 FAISS 索引)
    try:
        print("🔄 初始化 RAG V2 系統 (將載入*訓練集*索引)...")
        rag_system = MammographyRAGSystemV2(
            reports_dir='Kang_Ning_General_Hospital/'
        )
        print("✅ RAG V2 系統初始化完成。")
    except Exception as e:
        print(f"❌ 系統初始化失敗: {e}")
        print("請確保所有 .index 和 .pkl 檔案都已由 'build_v2_indices_from_split.py' 生成")
        return

    # 3. 準備儲存結果
    results = [] # 儲存 (gt_birads, ai_birads)
    failed_cases = [] # 儲存執行失敗的案例

    # 4. 遍歷測試集並執行評估
    print(f"\n🔄 開始遍歷 {len(test_case_ids)} 個測試案例...")
    
    # 使用 TQDM 顯示進度條
    for case_id in tqdm(test_case_ids, desc="評估測試集"):
        try:
            # 4.1 取得真實報告 (Ground Truth)
            gt_report = rag_system.reports_dict.get(case_id)
            if not gt_report:
                print(f"⚠️ 警告: 找不到 {case_id} 的真實報告，跳過")
                failed_cases.append(case_id)
                continue
            
            gt_birads = parse_birads_from_text(gt_report.raw_text)

            # 4.2 執行 V2 RAG 系統 (這會呼叫 Roboflow 和 Anthropic API)
            # ！！！注意：這會花費金錢和時間 ！！！
            ai_report_text = rag_system.run_v2_evaluation(
                query_case_id=case_id,
                k_density=3,
                k_lesion=3
            )
            
            # 4.3 從 AI 生成的報告中解析 BI-RADS
            ai_birads = parse_birads_from_text(ai_report_text)
            
            # 4.4 儲存結果
            results.append({
                'case_id': case_id,
                'gt_birads': gt_birads,
                'ai_birads': ai_birads,
                'gt_text': gt_report.raw_text,
                'ai_text': ai_report_text
            })

        except Exception as e:
            print(f"❌ 案例 {case_id} 執行失敗: {e}")
            failed_cases.append(case_id)
    
    print("\n✅ 測試集評估完成！")
    print("="*80)
    print("📊 最終量化評估報告")
    print("="*80)

    # 5. 計算 BI-RADS 準確率
    correct_count = 0
    total_evaluated = len(results)
    
    if total_evaluated == 0:
        print("❌ 沒有任何案例成功執行，無法計算指標。")
        return

    # 建立混淆矩陣
    confusion_matrix = Counter() # (gt, ai) -> count
    
    for res in results:
        gt = res['gt_birads'] if res['gt_birads'] is not None else 'N/A'
        ai = res['ai_birads'] if res['ai_birads'] is not None else 'N/A'
        
        confusion_matrix[(gt, ai)] += 1
        
        if gt != 'N/A' and gt == ai:
            correct_count += 1

    accuracy = (correct_count / total_evaluated) * 100
    
    print(f"  - 總測試案例: {len(test_case_ids)}")
    print(f"  - 成功評估: {total_evaluated}")
    print(f"  - 執行失敗: {len(failed_cases)}")
    
    print("\n--- BI-RADS 類別準確率 ---")
    print(f"  - 準確率 (Accuracy): {correct_count} / {total_evaluated} = {accuracy:.2f}%")
    
    print("\n--- 混淆矩陣 (Confusion Matrix) ---")
    print("  (真實 BI-RADS, AI BI-RADS): 數量")
    for (gt, ai), count in confusion_matrix.items():
        print(f"  - ({gt}, {ai}): {count}")
        if gt == 0 and ai == 1:
            print(f"    🚨 嚴重錯誤 (偽陰性): {count} 次")
        if gt == 1 and ai == 0:
            print(f"    ⚠️ 安全錯誤 (偽陽性): {count} 次")

    # 6. 儲存詳細結果
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 詳細評估結果已儲存到: evaluation_results.json")

if __name__ == "__main__":
    # 確保你有設定環境變數
    if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("ROBOFLOW_API_KEY"):
        print("❌ 錯誤: 缺少 ANTHROPIC_API_KEY 或 ROBOFLOW_API_KEY 環境變數")
    else:
        run_evaluation()