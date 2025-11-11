#!/bin/bash

# 設定遇到錯誤時不要立刻停止，因為我們希望兩個都盡量跑完
set +e

echo "========================================================"
echo "🛌 醫療報告生成系統 - 夜間自動評估程序"
echo "開始時間: $(date)"
echo "========================================================"

# 確保環境變數存在 (請在執行此腳本前先 export 好，或者在這裡取消註解並填入)
# export ANTHROPIC_API_KEY="sk-ant-..."
# export ROBOFLOW_API_KEY="..."
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

if [ -z "$ANTHROPIC_API_KEY" ] || [ -z "$ROBOFLOW_API_KEY" ]; then
    echo "❌ 錯誤: 請先設定 ANTHROPIC_API_KEY 和 ROBOFLOW_API_KEY 環境變數！"
    exit 1
fi

# echo -e "\n---------- 階段 1/2: 執行 V2 RAG 系統評估 ----------"
# echo "正在執行 python evaluate_on_test_set.py ..."
# echo "日誌將輸出到: log_rag_v2.txt (請稍候，這需要一段時間...)"
# # 使用 unbuffer (如果有的話) 或直接執行，將 stdout 和 stderr 都導向檔案
# python evaluate_on_test_set.py > log_rag_v2.txt 2>&1
# if [ $? -eq 0 ]; then
#     echo "✅ V2 RAG 評估完成！"
# else
#     echo "⚠️ V2 RAG 評估似乎發生了錯誤，請檢查 log_rag_v2.txt"
# fi

echo -e "\n---------- 階段 2/2: 執行 No-RAG 對照組評估 ----------"
echo "正在執行 python evaluate_no_rag.py ..."
echo "日誌將輸出到: log_no_rag.txt"
python evaluate_no_rag.py > log_no_rag.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✅ No-RAG 評估完成！"
else
    echo "⚠️ No-RAG 評估似乎發生了錯誤，請檢查 log_no_rag.txt"
fi

echo -e "\n========================================================"
echo "🎉 所有評估工作已結束！"
echo "結束時間: $(date)"
echo "請檢查生成的 .json 結果檔案和 .txt 日誌檔案。"
echo "晚安！"
echo "========================================================"