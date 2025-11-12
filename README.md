# Medical Report Generation via Dual-Stream RAG

This repository contains the complete codebase for the NTHU CS Undergraduate Project, a system designed to improve the clinical safety of AI-generated mammography reports by reducing false negatives.

## Key Results

This system demonstrates a significant improvement over a baseline VLM-only approach:

Clinical Sensitivity (Recall): Doubled from 35.42% to 70.83%, successfully "rescuing" over half of the cases missed by the baseline.

Report Quality (ROUGE-2): Increased by +46.2%, indicating a higher accuracy in using professional medical terminology.

## System Architecture

Our core innovation is the Dual-Stream RAG mechanism, which decouples the global task of density assessment from the local task of lesion characterization.

### How to Reproduce the Experiment

This guide outlines the complete step-by-step pipeline to reproduce the results from our paper.

Step 0: Setup

1. Install Dependencies

This project requires Python 3.10+ and the following packages:

#### Install all required packages
```bash
pip install torch faiss-cpu anthropic inference-sdk
pip install evaluate bert_score rouge_score scikit-learn
pip install matplotlib seaborn pandas open-clip-torch tqdm
```

2. Set Environment Variables (CRITICAL)

This system relies on two external APIs. You must set these environment variables in your terminal.

- Your API key from Anthropic (for Claude VLM)
export ANTHROPIC_API_KEY="sk-ant-..."

- Your private API key from Roboflow (for YOLOv8 Detector)
export ROBOFLOW_API_KEY="..."

- (Recommended) For resolving library conflicts on macOS/Linux
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE


#### Step 1: Data Placement (Requires Private Data)

This repository does not include the private hospital data. Before running, you must manually place your data according to the following structure (these paths are ignored by .gitignore):

/

├── preprocessed_images/       <-- (YOUR .NPY IMAGE FOLDERS)

│   ├── 20230721_1st/

│   │   ├── MAMO_DEID_...-00001/

│   │   │   └── I0000000.npy ...

│   │   └── ...

│   ├── 20230728_2nd/

│   └── 20230804_3rd/

│

└── Hospital/  <-- (YOUR .XLSX REPORT FILES)

    ├── MAMO_DEID_20230721_NOPID.xlsx
    
    └── ...


#### Step 2: Run the Full Evaluation Pipeline

This is a 4-stage automated process. Run these scripts in order.

- Stage 1: Create Train/Test Split

This randomly splits your 394 case IDs into train_cases.json and test_cases.json.

python prepare_split.py


- Stage 2: Build Feature Database

This script iterates through all 394 cases, calls the Roboflow API to detect lesions, and uses BiomedCLIP to extract local/global features.

This is the most time-consuming step (approx. 50-60 minutes).

```bash
python run_extraction.py
```
Output: mammography_features_v2.pkl


- Stage 3: Build FAISS Indices (from Training Set)

This script builds the Density and Lesion retrieval indices using only the 315 training cases, preventing data leakage.

```bash
python build_v2_indices_from_split.py
```
Outputs: faiss_global_...index, faiss_lesion_...index, etc.


- Stage 4: Run Final Evaluation (RAG vs. No-RAG)

This master script executes the full evaluation on the 79 "unseen" test cases for both our V2 RAG system and the No-RAG baseline.

This will take several hours and make many API calls.

- Make the script executable
chmod +x run_all_evaluations.sh

- Run it (e.g., overnight)
./run_all_evaluations.sh

##### Main Outputs:
- evaluation_results.json (V2 RAG results)
- evaluation_results_NO_RAG.json (Baseline results)


#### Step 3: Analyze Results & Generate Plots

After the evaluation is complete, use these scripts to generate the final metrics and graphs for your report.

- 1. Get Clinical Metrics (Sensitivity, Specificity)
```bash
python analyze_improvement.py
```
- 2. Get NLG Metrics (ROUGE, BERTScore)
```bash
python calculate_text_metrics.py
```
- 3. (Optional) Verify False Negative Causes
```bash
python verify_fn_cause.py
```

- 4. Generate PNG plots
```bash
python plot_clinical_metrics.py
python plot_confusion_matrices.py
```

#### Core Scripts

``` bash
rag_system_v2.py: (Core) Main logic for the V2 RAG system (Detection + Dual-Stream Retrieval + VLM).

no_rag_system.py: (Core) Logic for the No-RAG baseline system (Detection + VLM only).

detection_and_feature_extractor.py: (Core) Handles Roboflow API (YOLO) and local BiomedCLIP feature extraction.

run_all_evaluations.sh: (Main) Master script to reproduce the experiment.

analyze_improvement.py: Calculates and prints final clinical metrics.
```