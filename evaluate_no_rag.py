import json
import os
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from collections import Counter

# 導入 No-RAG 系統
from no_rag_system import MammographyNoRAGSystem
# 導入資料載入器以獲取真實報告 (用於比較)
from Get_Report import MammographyDataLoader

TEST_CASES_FILE = 'test_cases.json'

def load_test_ids():
    with open(TEST_CASES_FILE, 'r') as f:
        return json.load(f)

def parse_birads_from_text(report_text: str) -> Optional[int]:
    """[V3 修正版] 解析 BI-RADS (與 V2 評估腳本保持一致)"""
    if not isinstance(report_text, str): return None
    match = re.search(r'<BI_RADS_CATEGORY>(.*?)</BI_RADS_CATEGORY>', report_text, re.DOTALL | re.IGNORECASE)
    if match:
        digit_match = re.search(r'(\d)', match.group(1))
        if digit_match: return int(digit_match.group(1))
    match_fallback = re.search(r'(?:BI-RADS|Category)[\s:]*(\d)', report_text, re.IGNORECASE)
    if match_fallback: return int(match_fallback.group(1))
    return None

def run_evaluation():
    print("="*80)
    print("📉 開始在「測試集」上執行 No-RAG (對照組) 評估...")
    print("="*80)
    
    test_ids = load_test_ids()
    no_rag_system = MammographyNoRAGSystem()
    
    # 載入真實報告用於對照
    report_loader = MammographyDataLoader('Kang_Ning_General_Hospital/')
    all_reports = report_loader.load_all_reports()
    reports_dict = {r.case_id: r for r in all_reports}
    
    results = []
    failed_cases = []

    for case_id in tqdm(test_ids, desc="No-RAG 評估中"):
        try:
            gt_report = reports_dict.get(case_id)
            if not gt_report: continue
            
            # 執行 No-RAG 生成
            ai_text = no_rag_system.run_no_rag_evaluation(case_id)
            
            results.append({
                'case_id': case_id,
                'gt_birads': parse_birads_from_text(gt_report.raw_text),
                'ai_birads': parse_birads_from_text(ai_text),
                'gt_text': gt_report.raw_text,
                'ai_text': ai_text
            })
        except Exception as e:
            print(f"❌ {case_id} 失敗: {e}")
            failed_cases.append(case_id)

    # --- 計算統計數據 ---
    print("\n📊 No-RAG 最終評估報告")
    confusion_matrix = Counter()
    correct = 0
    for res in results:
        gt = res['gt_birads'] if res['gt_birads'] is not None else 'N/A'
        ai = res['ai_birads'] if res['ai_birads'] is not None else 'N/A'
        confusion_matrix[(gt, ai)] += 1
        if gt != 'N/A' and gt == ai: correct += 1

    print(f"  - 成功評估: {len(results)} / {len(test_ids)}")
    print(f"  - 準確率: {correct}/{len(results)} = {(correct/len(results))*100:.2f}%" if results else "N/A")
    print("\n--- 混淆矩陣 ---")
    for (gt, ai), count in confusion_matrix.items():
        print(f"  - (真:{gt}, AI:{ai}): {count}")

    with open('evaluation_results_NO_RAG.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n💾 結果已儲存: evaluation_results_NO_RAG.json")

if __name__ == "__main__":
    run_evaluation()