import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional, List
import re

@dataclass
class MammographyReport:
    """乳房X光報告的資料結構"""
    
    # 基本資訊
    case_id: str  # 例如: MAMO_DEID_20230721_-00001
    
    # 報告內容
    findings: str  # 檢查所見 (Findings)
    impression: str  # 診斷結論 (BI-RADS Category)
    
    # 額外資訊
    breast_density: str  # 乳房密度描述
    old_comparison: bool
    old_comparison_date: Optional[str]
    
    # 原始文本
    raw_text: str
    
    @classmethod
    def parse_from_text(cls, case_id: str, raw_text: str):
        """
        從原始報告文本解析出結構化資料
        
        Args:
            case_id: 案例編號 (例如 MAMO_DEID_20230721_-00001)
            raw_text: 原始報告文本
        """
        if pd.isna(raw_text) or not isinstance(raw_text, str):
            raw_text = ""
        
        # 提取 Findings (檢查所見) - 從 "Bilateral" 到 "BI-RADS" 之間的內容
        findings_match = re.search(r'(Bilateral screening mammograms.+?)(?=BI-RADS|$)', raw_text, re.DOTALL)
        findings = findings_match.group(1).strip() if findings_match else raw_text.strip()
        
        # 提取 Impression (診斷結論) - BI-RADS 分類
        impression_match = re.search(r'(BI-RADS Category[^\.]+\.)', raw_text)
        impression = impression_match.group(1).strip() if impression_match else ""
        
        # 提取乳房密度描述
        density_patterns = [
            r'breasts are (heterogeneously dense)',
            r'breasts are (almost entirely fatty)',
            r'breasts are (scattered fibroglandular)',
            r'breasts are (extremely dense)',
            r'(scattered areas of fibroglandular density)',
            r'(heterogeneously dense)',
        ]
        breast_density = ""
        for pattern in density_patterns:
            density_match = re.search(pattern, raw_text, re.IGNORECASE)
            if density_match:
                breast_density = density_match.group(1)
                break
        
        # 提取舊影像比對資訊
        old_comparison = "Old images comparison: Yes" in raw_text
        old_date_match = re.search(r'Date:\s*(\d{4}/\d{2}/\d{2})', raw_text)
        old_comparison_date = old_date_match.group(1) if old_date_match else None
        
        return cls(
            case_id=case_id,
            findings=findings,
            impression=impression,
            breast_density=breast_density,
            old_comparison=old_comparison,
            old_comparison_date=old_comparison_date,
            raw_text=raw_text
        )
    
    def to_dict(self):
        """轉換成字典格式"""
        return {
            'case_id': self.case_id,
            'findings': self.findings,
            'impression': self.impression,
            'breast_density': self.breast_density,
            'old_comparison': self.old_comparison,
            'old_comparison_date': self.old_comparison_date,
            'raw_text': self.raw_text
        }
    
    def get_formatted_report(self):
        """回傳格式化的報告文本"""
        report = f"""
            {'='*60}
            Case ID: {self.case_id}
            {'='*60}

            FINDINGS:
            {self.findings}

            IMPRESSION:
            {self.impression}

            BREAST DENSITY:
            {self.breast_density if self.breast_density else 'Not specified'}

            COMPARISON:
            Old images comparison: {'Yes' if self.old_comparison else 'No'}
            {f'Comparison date: {self.old_comparison_date}' if self.old_comparison_date else 'No prior images available'}
            {'='*60}
        """
        return report.strip()
    
    def __repr__(self):
        return f"MammographyReport(case_id={self.case_id}, BI-RADS={self.get_birads_category()})"
    
    def get_birads_category(self) -> Optional[int]:
        """提取 BI-RADS 分類數字"""
        match = re.search(r'Category (\d+)', self.impression)
        return int(match.group(1)) if match else None


class MammographyDataLoader:
    """載入並管理所有乳房X光報告"""

    def __init__(self, full_files: List[str], dir_path: str = 'Kang_Ning_General_Hospital/'):
        self.dir_path = dir_path
        self.reports: List[MammographyReport] = []
        self.names = ['20230721', '20230728', '20230804']
        self.full_files = full_files

    def load_all_reports(self):
        """載入所有報告檔案"""
        for name in self.names:
            report_file = f"MAMO_DEID_{name}_NOPID.xlsx"
            full_path = os.path.join(self.dir_path, report_file)
            
            try:
                # 使用 pandas 讀取 Excel
                # 注意：根據截圖，第一列是 "Status"，第二列開始才是資料
                df = pd.read_excel(full_path, header=0)  # header=0 表示第一行是欄位名
                
                print(f"✅ Loaded {report_file}: {len(df)} records")
                print(f"   Columns: {df.columns.tolist()}")
                
                # 解析每一筆報告
                for idx, row in df.iterrows():
                    # A 欄是 case_id (第一欄，通常是 'Status' 或沒有名稱)
                    # B 欄是報告內容 (第二欄)
                    case_id = str(row.iloc[0]).strip()
                    if case_id not in self.full_files:
                        continue
                    
                    raw_text = str(row.iloc[1]) if len(row) > 1 else ""
                    
                    # 跳過空白行或無效資料
                    if pd.isna(case_id) or case_id == 'nan' or case_id == '':
                        continue
                    
                    # 解析報告
                    report = MammographyReport.parse_from_text(case_id, raw_text)
                    self.reports.append(report)
                    
            except Exception as e:
                print(f"❌ Error reading {report_file}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n📊 Total reports loaded: {len(self.reports)}")
        return self.reports
    
    def get_report_by_id(self, case_id: str) -> Optional[MammographyReport]:
        """根據 case_id 取得報告"""
        for report in self.reports:
            if report.case_id == case_id:
                return report
        return None
    
    def get_reports_by_birads(self, category: int) -> List[MammographyReport]:
        """根據 BI-RADS 分類篩選報告"""
        return [r for r in self.reports if r.get_birads_category() == category]
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """匯出成 pandas DataFrame"""
        return pd.DataFrame([r.to_dict() for r in self.reports])
    
    def get_statistics(self):
        """取得統計資訊"""
        df = self.export_to_dataframe()
        stats = {
            'total_cases': len(self.reports),
            'with_old_comparison': df['old_comparison'].sum(),
            'birads_distribution': {}
        }
        
        # 統計 BI-RADS 分布
        for report in self.reports:
            category = report.get_birads_category()
            if category:
                stats['birads_distribution'][f'Category {category}'] = \
                    stats['birads_distribution'].get(f'Category {category}', 0) + 1
        
        return stats


# ==================== 使用範例 ====================

if __name__ == "__main__":
    
    
    # Read full files
    full_files = []
    with open('docs/full_files.txt', 'r') as f:
        for line in f:
            full_files.append(line.strip())

    # 初始化載入器
    loader = MammographyDataLoader(full_files, dir_path='preprocessed_images/')
    
    # 載入所有報告
    print("🔄 Loading all reports...\n")
    reports = loader.load_all_reports()
    
    # 顯示統計資訊
    print("\n📊 Statistics:")
    stats = loader.get_statistics()
    print(f"Total cases: {stats['total_cases']}")
    print(f"Cases with old comparison: {stats['with_old_comparison']}")
    print("\nBI-RADS Distribution:")
    for category, count in stats['birads_distribution'].items():
        print(f"  {category}: {count}")
    
    # 測試：顯示第一筆報告
    if reports:
        print("\n" + "="*60)
        print("📄 Sample Report (First Case):")
        print("="*60)
        print(reports[0].get_formatted_report())
    
    # 測試：根據 case_id 查詢
    print("\n" + "="*60)
    test_case = "MAMO_DEID_20230721_-00004"
    report = loader.get_report_by_id(test_case)
    if report:
        print(f"✅ Found report for {test_case}")
        print(report.get_formatted_report())
    else:
        print(f"❌ Report not found for {test_case}")
    
    # 測試：篩選 BI-RADS Category 0 的案例
    print("\n" + "="*60)
    birads_0_cases = loader.get_reports_by_birads(0)
    print(f"📊 BI-RADS Category 0 cases: {len(birads_0_cases)}")
    if birads_0_cases:
        print("\nFirst Category 0 case:")
        print(birads_0_cases[0].get_formatted_report())
    
    # 匯出成 DataFrame 並儲存
    # print("\n" + "="*60)
    # df = loader.export_to_dataframe()
    # print(f"✅ DataFrame shape: {df.shape}")
    # print("\nFirst 3 rows:")
    # print(df[['case_id', 'impression', 'breast_density']].head(3))
    
    # # 選擇性：存成 CSV
    # output_path = 'processed_reports.csv'
    # df.to_csv(output_path, index=False)
    # print(f"\n💾 Saved to {output_path}")