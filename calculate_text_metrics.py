import json
import numpy as np
from tqdm import tqdm
import evaluate # Hugging Face 的評估庫

def load_results(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_nlg_metrics(results, system_name):
    print(f"\n正在計算 {system_name} 的 NLG 指標...")
    
    # 準備資料
    predictions = [r['ai_text'] for r in results if r['ai_text'] and r['gt_text']]
    references = [r['gt_text'] for r in results if r['ai_text'] and r['gt_text']]
    
    if not predictions:
        print("沒有有效的預測結果可供計算。")
        return

    # 1. 計算 ROUGE
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=predictions, references=references)
    print(f"--- {system_name} ROUGE Scores ---")
    print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")

    # 2. 計算 BERTScore (使用 distilbert-base-uncased 模型，速度較快)
    # 注意：第一次執行會下載模型，可能需要一點時間
    bertscore = evaluate.load("bertscore")
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")
    
    print(f"--- {system_name} BERTScore ---")
    print(f"Precision: {np.mean(bert_results['precision']):.4f}")
    print(f"Recall:    {np.mean(bert_results['recall']):.4f}")
    print(f"F1:        {np.mean(bert_results['f1']):.4f}") # 這通常是最重要的單一指標

def main():
    # 嘗試載入兩個結果檔
    try:
        rag_results = load_results('evaluation_results.json')
        print(f"載入 RAG 結果: {len(rag_results)} 筆")
        calculate_nlg_metrics(rag_results, "V2 RAG")
    except FileNotFoundError:
        print("找不到 evaluation_results.json")

    try:
        no_rag_results = load_results('evaluation_results_NO_RAG.json')
        print(f"載入 No-RAG 結果: {len(no_rag_results)} 筆")
        calculate_nlg_metrics(no_rag_results, "No-RAG")
    except FileNotFoundError:
        print("找不到 evaluation_results_NO_RAG.json")

if __name__ == "__main__":
    main()
