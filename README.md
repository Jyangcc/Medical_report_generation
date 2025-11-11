 Medical Report Generation for Mammography via Dual-Stream RAG

This is the official repository for the NTHU CS Undergraduate Project (2025), "SafeMammo." This project introduces a "Safety-First" AI reporting system to reduce high false-negative rates in mammography interpretation by augmenting Vision-Language Models (VLMs) with a novel Dual-Stream, Lesion-Aware Retrieval-Augmented Generation (RAG) architecture.

摘要 (Abstract)

Standard Vision-Language Models (VLMs) often fail to detect subtle, small lesions in high-resolution mammograms, leading to clinically dangerous false-negative rates. This project integrates explicit object detection (YOLOv8) with a Dual-Stream RAG mechanism. The system first localizes potential lesions and then uses BiomedCLIP to extract decoupled global (density) and local (lesion) features. These features query two separate clinical databases to retrieve visually similar historical cases.

A multimodal VLM (Claude) synthesizes these inputs (full images, lesion crops, and retrieved reports) to generate a professional report. Our experiments show this RAG-enhanced system improves clinical sensitivity for suspicious findings (BI-RADS 0) from 35.42% (baseline) to 70.83%, effectively doubling the detection rate of potential cancers.

系統架構 (System Architecture)

Our core innovation is the Dual-Stream RAG mechanism, which decouples the task of "density assessment" (a global task) from "lesion characterization" (a local task).

Figure 1: The full Dual-Stream RAG pipeline, from detection to generation.

This architecture feeds a complex, multimodal prompt to the VLM, ensuring it has all the necessary visual evidence and clinical context to make an informed decision.

Figure 2: The structure of the multimodal prompt fed to the VLM.

主要成果 (Key Results)

Our system demonstrates a clear and significant improvement in clinical safety by drastically reducing false negatives.

1. 臨床效能 (Clinical Performance)

The RAG system doubles the sensitivity (ability to find cancer) compared to the baseline VLM, proving the value of our retrieval architecture.

Figure 3: V2 RAG (Our System) vs. No-RAG (Baseline) on clinical sensitivity and specificity.

2. 錯誤分析 (Error Analysis)

The confusion matrices show that our V2 RAG system successfully converted a large number of False Negatives (bottom-left) into True Positives (top-left).

Figure 4: Normalized confusion matrices showing the shift from FN to TP.

🚀 如何使用 (How to Use This Repository)

This guide outlines the complete pipeline, from data setup to final evaluation, allowing you to reproduce our experimental results.

1. 專案設定 (Setup)

a. 複製儲存庫:

git clone [Your-Repo-URL]
cd Medical_report_generation


b. 安裝依賴 (Dependencies):
We recommend using a Python virtual environment.

pip install -r requirements.txt
# Key packages include:
# pip install torch faiss-cpu anthropic inference-sdk
# pip install evaluate bert_score rouge_score scikit-learn
# pip install matplotlib seaborn pandas open-clip-torch


c. 設定環境變數 (CRITICAL):
This system relies on two external APIs. You must set these environment variables in your terminal.

# Your API key from Anthropic (for Claude VLM)
export ANTHROPIC_API_KEY="sk-ant-..."

# Your private API key from Roboflow (for YOLOv8 Detector)
export ROBOFLOW_API_KEY="..."

# (Optional) For resolving library conflicts on macOS/Linux
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE


2. 資料準備 (Data Placement)

This project does not include the raw data due to privacy. You must provide your own dataset following this structure:

Medical_report_generation/
├── preprocessed_images/       <-- (YOUR NPY IMAGES)
│   ├── 20230721_1st/
│   │   ├── MAMO_DEID_...-00001/
│   │   │   ├── I0000000.npy  (RCC)
│   │   │   ├── I0000001.npy  (LCC)
│   │   │   ├── I0000002.npy  (RMLO)
│   │   │   └── I0000003.npy  (LMLO)
│   │   └── ...
│   ├── 20230728_2nd/
│   └── 20230804_3rd/
│
└── Kang_Ning_General_Hospital/  <-- (YOUR XLSX REPORTS)
    ├── MAMO_DEID_20230721_NOPID.xlsx
    ├── MAMO_DEID_20230728_NOPID.xlsx
    └── MAMO_DEID_20230804_NOPID.xlsx


3. 實驗流程 (Execution Pipeline)

Follow these steps sequentially to build the databases and run the evaluations.

Step 1: Create Train/Test Split
This script randomly splits your 394 case IDs into train_cases.json and test_cases.json.

python prepare_split.py


Step 2: Build Feature Database (V2)
This is the most time-consuming step. It will iterate through all 394 cases, call the Roboflow API to detect lesions, and call BiomedCLIP to extract local/global features.

# This will take a long time (e.g., 50+ minutes)
python run_extraction.py
# Output: mammography_features_v2.pkl


Step 3: Build FAISS Indices
This script reads mammography_features_v2.pkl and uses the training set (train_cases.json) to build the two FAISS retrieval indices.

python build_v2_indices_from_split.py
# Output: faiss_global_...index, faiss_lesion_...index, etc.


Step 4: Run All Evaluations (The "Go-to-Sleep" Script)
You are now ready to run the final experiment. This script will execute the full evaluation on the 79 test cases for both our V2 RAG system and the No-RAG baseline.

This will take several hours and make many API calls.

# Make the script executable
chmod +x run_all_evaluations.sh

# Run it (and go to sleep)
./run_all_evaluations.sh

# Outputs:
# - evaluation_results.json (V2 RAG results)
# - evaluation_results_NO_RAG.json (Baseline results)
# - log_rag_v2.txt (Detailed log for V2)
# - log_no_rag.txt (Detailed log for baseline)


Step 5: Analyze Results
After the evaluation is complete, use these scripts to generate the final metrics and graphs for your report.

# Get Sensitivity, Specificity, and key "rescued" cases
python analyze_improvement.py

# Get ROUGE, BERTScore F1
python calculate_text_metrics.py

# Generate the result charts (PNG files)
python plot_clinical_metrics.py
python plot_confusion_matrices.py


專案檔案結構 (Project File Structure)

/ (Root)

rag_system_v2.py: (Core) Main logic for the V2 RAG system.

no_rag_system.py: (Core) Logic for the No-RAG baseline system.

detection_and_feature_extractor.py: (Core) Handles Roboflow API (YOLO) and local BiomedCLIP feature extraction.

Get_Report.py: Utility for loading and parsing .xlsx report files.

/Data Pipeline

prepare_split.py: (Step 1) Creates train_cases.json and test_cases.json.

run_extraction.py: (Step 2) Runs the feature extractor on all data to create mammography_features_v2.pkl.

build_v2_indices_from_split.py: (Step 3) Builds FAISS indices from the training set.

/Evaluation

run_all_evaluations.sh: (Step 4) Main executable script to run both evaluations.

evaluate_on_test_set.py: Runs the V2 RAG system on the test set.

evaluate_no_rag.py: Runs the No-RAG baseline on the test set.

/Analysis

analyze_improvement.py: Calculates clinical metrics (Sensitivity, etc.).

calculate_text_metrics.py: Calculates NLG metrics (ROUGE, BERTScore).

plot_clinical_metrics.py: Generates the clinical results bar chart.

plot_confusion_matrices.py: Generates the confusion matrix heatmaps.

verify_fn_cause.py: Script to analyze why false negatives occurred.

/Data (Git Ignored)

preprocessed_images/: (Ignored) Your .npy images.

Kang_Ning_General_Hospital/: (Ignored) Your .xlsx reports.

*.pkl, *.index, *.json, *.log: (Ignored) All generated data files.