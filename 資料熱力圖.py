# 匯入所需的函式庫
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 設定中文字型
plt.rcParams['font.family'] = 'Microsoft YaHei'

# 設置隨機種子
np.random.seed(42)

def 載入與預處理數據(文件路徑):
    # 讀取資料
    數據框 = pd.read_csv(文件路徑)
    
    # 定義特徵預測名稱
    特徵列 = [
        'age',        # 年齡
        'sex',        # 性別 (0: 女性, 1: 男性)
        'cp',         # 胸痛類型 (0: 無胸痛, 1: 頸部, 2: 心絞痛, 3: 非心絞痛)
        'trestbps',   # 靜態血壓 (mm Hg)
        'chol',       # 胆固醇 (mg/dl)
        'fbs',        # 空腹血糖 (1: 大於 120 mg/dl, 0: 小於等於 120 mg/dl)
        'restecg',    # 靜態心電圖結果 (0: 正常, 1: ST-T 波異常, 2: 左心室肥大)
        'thalach',    # 最大心跳速率 (bpm)
        'exang',      # 運動引起的心絞痛 (1: 有, 0: 無)
        'oldpeak',    # 由於運動引起的 ST 降低 (以毫米為單位)
        'slope',      # ST 段的坡度 (0: 上升, 1: 平坦, 2: 下降)
        'ca',         # 螢光透視結果
        'thal'        # 地中海貧血 (3: 正常, 6: 固定缺陷, 7: 可逆缺陷)
    ]

    return 數據框, 特徵列

def 繪製完整相關性熱力圖(數據框, 特徵列):
    """繪製資料集特徵之間的完整相關性熱力圖"""
    
    # 計算相關矩陣 (包含目標變數)
    相關矩陣 = 數據框[特徵列 + ['target']].corr()
    
    # 設定圖的大小
    plt.figure(figsize=(12, 10))
    
    # 使用 seaborn 繪製完整熱力圖 (不使用遮罩)
    sns.heatmap(
        相關矩陣, 
        annot=True,                     # 顯示數值
        fmt='.2f',                      # 數值格式為小數點後兩位
        cmap='coolwarm',                # 使用紅藍對比色彩
        linewidths=0.5,                 # 設置單元格之間的線寬
        vmin=-1, vmax=1,                # 設定色彩映射的範圍
        cbar_kws={'label': '相關係數'}   # 色彩條標籤
    )
    
    # 設定標題和標籤
    plt.title('心臟病資料集特徵完整相關性熱力圖', fontsize=16, pad=20)
    
    # 調整軸標籤
    plt.xticks(rotation=45, ha='right', fontsize=10)  # 旋轉 x 軸標籤以避免重疊
    plt.yticks(fontsize=10)
    
    # 調整版面，確保熱力圖完整顯示
    plt.tight_layout()
    
    # 顯示圖表
    plt.show()
    
    # 除了熱力圖，同時返回相關矩陣，以便查看數值
    return 相關矩陣

def 分析目標相關性(相關矩陣):
    """分析並印出與目標變數的相關性排序"""
    
    # 提取與目標變數的相關性
    與目標相關 = 相關矩陣['target'].drop('target')  # 移除目標自身的相關性
    
    # 按照絕對值大小排序
    絕對值排序 = 與目標相關.abs().sort_values(ascending=False)
    
    print("特徵與心臟病診斷目標的相關性 (按重要性排序):")
    print("====================================")
    
    # 列印排序後的相關性
    for 特徵 in 絕對值排序.index:
        相關值 = 與目標相關[特徵]
        相關方向 = "正相關" if 相關值 > 0 else "負相關"
        print(f"{特徵}: {相關值:.3f} ({相關方向})")
    
    print("\n正相關表示該特徵值越高，患心臟病可能性越高")
    print("負相關表示該特徵值越高，患心臟病可能性越低")

# 主程序
if __name__ == "__main__":
    文件路徑 = r"H:\心跳正常\heart.csv"  # 替換為你的資料檔案路徑
    數據框, 特徵列 = 載入與預處理數據(文件路徑)
    
    # 繪製熱力圖並獲取相關矩陣
    相關矩陣 = 繪製完整相關性熱力圖(數據框, 特徵列)
    
    # 分析目標相關性
    分析目標相關性(相關矩陣)