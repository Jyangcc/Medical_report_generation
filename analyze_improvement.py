import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def load_results(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_metrics(results, system_name):
    y_true = []
    y_pred = []
    for r in results:
        if r['gt_birads'] is not None and r['ai_birads'] is not None:
            # 簡化為二元分類：0 (異常) vs 1 (正常)
            # 如果真實是 4, 5, 6 也算異常(0類)
            gt = 0 if r['gt_birads'] in [0, 4, 5, 6] else 1
            pred = 0 if r['ai_birads'] in [0, 4, 5, 6] else 1
            y_true.append(gt)
            y_pred.append(pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1, 0]).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 # 召回率 (抓出病人的能力)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 # 特異度 (確認健康人的能力)
    
    print(f"\n--- {system_name} 醫療指標 ---")
    print(f"敏感度 (Sensitivity/Recall): {sensitivity:.2%} (越高越好，減少漏診)")
    print(f"特異度 (Specificity):       {specificity:.2%} (越高越好，減少誤判)")
    return y_true, y_pred

def main():
    rag_results = {r['case_id']: r for r in load_results('evaluation_results.json')}
    no_rag_results = {r['case_id']: r for r in load_results('evaluation_results_NO_RAG.json')}
    
    common_ids = set(rag_results.keys()) & set(no_rag_results.keys())
    print(f"共同評估案例數: {len(common_ids)}")

    rag_list = [rag_results[cid] for cid in common_ids]
    no_rag_list = [no_rag_results[cid] for cid in common_ids]

    calculate_metrics(no_rag_list, "No-RAG (Baseline)")
    calculate_metrics(rag_list, "V2 RAG (Ours)")

    print("\n=== 關鍵案例分析 ===")
    improved_cases = []
    for cid in common_ids:
        r_rag = rag_results[cid]
        r_no = no_rag_results[cid]
        
        # 尋找：真實是 0，No-RAG 猜 1 (錯)，但 RAG 猜 0 (對) 的案例
        if r_rag['gt_birads'] == 0 and r_no['ai_birads'] == 1 and r_rag['ai_birads'] == 0:
            improved_cases.append(cid)
            
    print(f"RAG 成功救援 (RAG對, No-RAG錯) 的 BI-RADS 0 案例數: {len(improved_cases)}")
    if improved_cases:
        print(f"範例救援案例 ID: {improved_cases[:5]}")
        # 你可以手動去查看這些案例的報告，看看 RAG 多寫了什麼

if __name__ == "__main__":
    main()