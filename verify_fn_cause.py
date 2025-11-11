import json
import re
from collections import defaultdict

def load_results(filename):
    """載入 JSON 評估結果"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_fn_cases(results):
    """找出所有偽陰性 (GT=0, AI=1) 的案例 ID"""
    fn_cases = []
    for r in results:
        gt = r['gt_birads']
        ai = r['ai_birads']
        
        if gt is None or ai is None: continue
        
        # 定義 異常 (Positive) 為 0 或 4, 5, 6
        # 定義 正常 (Negative) 為 1 或 2
        is_gt_positive = gt in [0, 4, 5, 6]
        is_ai_negative = ai in [1, 2]
        
        if is_gt_positive and is_ai_negative:
            fn_cases.append(r['case_id'])
            
    return fn_cases

def analyze_logs_for_cases(case_ids, log_filepath):
    """
    解析日誌檔案，找出指定案例的病灶檢測數量
    """
    print(f"\n🔄 正在讀取日誌檔案: {log_filepath} ...")
    try:
        with open(log_filepath, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"❌ 錯誤: 找不到日誌檔案 {log_filepath}")
        return None

    print(f"✅ 日誌讀取完畢。開始分析 {len(case_ids)} 個偽陰性案例...")
    
    analysis_results = {}
    
    for case_id in case_ids:
        # 使用正則表達式尋找該 case_id 的日誌區塊
        # 匹配 "🚀 開始 V2 評估: [CASE_ID]" ... 一直到 ... "檢測到 (\d+) 個病灶"
        # re.DOTALL 讓 '.' 可以匹配換行符號
        pattern = re.compile(
            r"🚀 開始 V2 評估: " + re.escape(case_id) + 
            r".*?檢測到 (\d+) 個病灶", 
            re.DOTALL
        )
        
        match = pattern.search(log_content)
        
        if match:
            detected_lesions_count = int(match.group(1))
            analysis_results[case_id] = detected_lesions_count
        else:
            analysis_results[case_id] = "日誌中未找到"
            
    return analysis_results

def main():
    results = load_results('evaluation_results.json')
    
    # 1. 找出所有偽陰性案例
    fn_cases = find_fn_cases(results)
    
    if not fn_cases:
        print("🎉 恭喜！在 evaluation_results.json 中沒有找到偽陰性案例！")
        return
        
    print(f"📊 找到了 {len(fn_cases)} 個偽陰性 (FN) 案例。正在驗證其根本原因...")
    
    # 2. 分析日誌
    log_results = analyze_logs_for_cases(fn_cases, 'log_rag_v2.txt')
    
    if log_results is None:
        return
        
    # 3. 打印報告
    print("\n" + "="*80)
    print("      偽陰性 (FN) 案例歸因分析報告")
    print("      (GT=0, AI=1)")
    print("="*80)
    
    detector_failures = 0
    vlm_failures = 0
    
    for case_id, lesion_count in log_results.items():
        if lesion_count == 0:
            print(f"  - 案例: {case_id}")
            print(f"    - YOLO 檢測到的病灶數: {lesion_count}")
            print(f"    - 歸因: 🚨 檢測器失敗 (Detector Failure)。VLM 沒看到病灶。")
            detector_failures += 1
        elif isinstance(lesion_count, int) and lesion_count > 0:
            print(f"  - 案例: {case_id}")
            print(f"    - YOLO 檢測到的病灶數: {lesion_count}")
            print(f"    - 歸因: ⚠️ VLM/RAG 失敗 (VLM/RAG Failure)。YOLO 找到了病灶，但 VLM 依然判斷為陰性。")
            vlm_failures += 1
        else:
            print(f"  - 案例: {case_id}")
            print(f"    - 結果: {lesion_count}")

    print("="*80)
    print("      總結")
    print("="*80)
    print(f"總偽陰性案例數: {len(fn_cases)}")
    print(f"歸因於「檢測器失敗」(YOLO 檢測數=0): {detector_failures} 例")
    print(f"歸因於「VLM/RAG 失敗」(YOLO 檢測數>0): {vlm_failures} 例")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()