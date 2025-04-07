import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 設定matplotlib使用中文字型
plt.rcParams['font.family'] = 'Microsoft YaHei'

# 設定隨機種子以確保結果可重現
np.random.seed(42)
tf.random.set_seed(42)

def plot_training_history(history, fold_num=None):
    """
    繪製模型訓練過程中的各項指標變化圖表
    
    參數:
    history: 模型訓練歷史記錄
    fold_num: 當前交叉驗證的折數
    """
    
    # 設定圖表版面配置：2x2的子圖
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 繪製損失函數圖表
    axes[0, 0].plot(history.history['loss'], label='訓練損失', linewidth=2)
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='驗證損失', linewidth=2)
    
    # 左側主標題
    axes[0, 0].set_title('模型損失', fontsize=12, pad=20, loc='left')
    # 右側折數標題
    if fold_num is not None:
        axes[0, 0].set_title(f'第 {fold_num} 折', fontsize=12, pad=20, loc='right')
    
    axes[0, 0].set_xlabel('epochs', fontsize=10)
    axes[0, 0].set_ylabel('損失值', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)

    # 繪製準確率圖表
    axes[0, 1].plot(history.history['accuracy'], label='訓練準確率', linewidth=2)
    if 'val_accuracy' in history.history:
        axes[0, 1].plot(history.history['val_accuracy'], label='驗證準確率', linewidth=2)
    
    # 左側主標題
    axes[0, 1].set_title('模型準確率', fontsize=12, pad=20, loc='left')
    # 右側折數標題
    if fold_num is not None:
        axes[0, 1].set_title(f'第 {fold_num} 折', fontsize=12, pad=20, loc='right')
    
    axes[0, 1].set_xlabel('epochs', fontsize=10)
    axes[0, 1].set_ylabel('準確率', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)

    # 繪製AUC圖表（如果有記錄）
    if 'auc' in history.history and 'val_auc' in history.history:
        axes[1, 0].plot(history.history['auc'], label='訓練 AUC', linewidth=2)
        axes[1, 0].plot(history.history['val_auc'], label='驗證 AUC', linewidth=2)
        
        # 左側主標題
        axes[1, 0].set_title('模型 AUC', fontsize=12, pad=20, loc='left')
        # 右側折數標題
        if fold_num is not None:
            axes[1, 0].set_title(f'第 {fold_num} 折', fontsize=12, pad=20, loc='right')
        
        axes[1, 0].set_xlabel('epochs', fontsize=10)
        axes[1, 0].set_ylabel('AUC 值', fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)

    # 繪製學習率變化圖表（如果有記錄）
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='學習率', linewidth=2, color='green')
        
        # 左側主標題
        axes[1, 1].set_title('學習率變化', fontsize=12, pad=20, loc='left')
        # 右側折數標題
        if fold_num is not None:
            axes[1, 1].set_title(f'第 {fold_num} 折', fontsize=12, pad=20, loc='right')
        
        axes[1, 1].set_xlabel('epochs', fontsize=10)
        axes[1, 1].set_ylabel('學習率值', fontsize=12)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        axes[1, 1].set_yscale('log')  # 使用對數尺度顯示學習率

    plt.tight_layout()
    plt.show()

def 載入與預處理數據(文件路徑):
    """
    載入資料並進行初步的預處理
    
    參數:
    文件路徑: CSV檔案的路徑
    
    返回:
    X: 特徵矩陣
    y: 目標變數
    特徵列: 特徵名稱列表
    """
    # 從CSV檔案讀取資料
    數據框 = pd.read_csv(文件路徑)
    
    # 定義用於預測的特徵欄位
    特徵列 = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
              'thalach', 'exang', 'oldpeak', 'slope', 'thal']
    
    # 分割特徵(X)和目標變數(y)
    X = 數據框[特徵列].values
    y = 數據框['target'].values
    
    return X, y, 特徵列

def 建立_LSTM模型(輸入維度):
    """
    建立LSTM神經網路模型
    
    參數:
    輸入維度: 輸入特徵的維度
    
    返回:
    編譯後的Keras模型
    """
    模型 = Sequential([
        # LSTM層，用於處理序列數據
        LSTM(128, input_shape=(輸入維度, 1), activation='tanh', return_sequences=False),
        Dropout(0.3),  # 防止過擬合
        Dense(64, activation='relu'),  # 全連接層
        Dropout(0.2),  # 防止過擬合
        Dense(2, activation='softmax')  # 輸出層，用於二分類
    ])
    
    # 編譯模型
    模型.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',  # 分類問題的損失函數
        metrics=['accuracy', tf.keras.metrics.AUC()]  # 評估指標
    )
    return 模型

def 繪製混淆矩陣(真實標籤, 預測標籤, 折數=None):
    """
    繪製混淆矩陣視覺化圖表
    
    參數:
    真實標籤: 實際的標籤值
    預測標籤: 模型預測的標籤值
    折數: 當前交叉驗證的折數
    """
    類別名稱 = ['無心臟病', '有心臟病']
    混淆矩陣 = confusion_matrix(真實標籤, 預測標籤)
    sns.heatmap(混淆矩陣, annot=True, fmt='d', cmap='Blues', 
                xticklabels=類別名稱, yticklabels=類別名稱)
    
    if 折數:
        plt.title('混淆矩陣', fontsize=14, pad=20, loc='left')
        plt.title(f'第 {折數} 折', fontsize=14, pad=20, loc='right')
    else:
        plt.title('混淆矩陣', fontsize=14, pad=20)
    
    plt.xlabel('預測標籤', fontsize=12)
    plt.ylabel('真實標籤', fontsize=12)
    plt.show()

def 訓練_LSTM模型(X, y):
    """
    使用K折交叉驗證訓練LSTM模型
    
    參數:
    X: 特徵矩陣
    y: 目標變數
    """
    # 初始化5折交叉驗證
    k折 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    平均準確率, 平均AUC = [], []
    
    # 進行K折交叉驗證
    for 折數, (訓練索引, 驗證索引) in enumerate(k折.split(X, y), 1):
        # 分割訓練集和驗證集
        X_訓練, X_驗證 = X[訓練索引], X[驗證索引]
        y_訓練, y_驗證 = y[訓練索引], y[驗證索引]
        
        # 特徵標準化
        標準化器 = StandardScaler()
        X_訓練 = 標準化器.fit_transform(X_訓練).reshape(-1, X.shape[1], 1)
        X_驗證 = 標準化器.transform(X_驗證).reshape(-1, X.shape[1], 1)
        
        # 將標籤轉換為one-hot編碼
        y_訓練 = to_categorical(y_訓練, num_classes=2)
        y_驗證 = to_categorical(y_驗證, num_classes=2)
        
        # 建立並訓練模型
        模型 = 建立_LSTM模型(X_訓練.shape[1])
        history = 模型.fit(
            X_訓練, y_訓練,
            validation_data=(X_驗證, y_驗證),
            epochs=3000,  # 訓練週期
            batch_size=32,  # 批次大小
            verbose=1  # 顯示訓練進度
        )
        
        # 繪製訓練過程圖表
        plot_training_history(history, 折數)
        
        # 進行預測
        y_預測 = np.argmax(模型.predict(X_驗證), axis=1)
        y_驗證真值 = np.argmax(y_驗證, axis=1)
        
        # 計算評估指標
        準確率 = (y_預測 == y_驗證真值).mean()
        平均準確率.append(準確率)
        
        # 計算AUC
        fpr, tpr, _ = roc_curve(y_驗證真值, y_預測)
        auc值 = auc(fpr, tpr)
        平均AUC.append(auc值)
        
        print(f"第 {折數} 折 - 準確率: {準確率:.4f}, AUC: {auc值:.4f}")
        
        # 繪製混淆矩陣
        繪製混淆矩陣(y_驗證真值, y_預測, 折數)
    
    # 輸出平均性能指標
    print(f"\n平均準確率: {np.mean(平均準確率):.4f}, 平均AUC: {np.mean(平均AUC):.4f}")

# 主程序入口
if __name__ == "__main__":
    文件路徑 = r"E:\心臟病\heart2.csv"  # 資料檔案路徑
    X, y, 特徵列 = 載入與預處理數據(文件路徑)
    訓練_LSTM模型(X, y)







# 提高模型的準確性可以從以下幾個方面進行調整：

# 數據預處理：

# 特徵標準化：神經網絡模型通常對特徵的範圍和分佈較為敏感，因此對特徵進行標準化或正規化可以提高模型的效果。你可以使用 StandardScaler 或 MinMaxScaler 來標準化特徵數據。
# 處理缺失值：如果數據中有缺失值，應該進行處理。可以選擇丟棄缺失值行，或用均值/中位數等進行填補。
# 模型架構調整：

# 增加層數或神經元：如果模型容量過小，可能無法學到足夠的特徵。你可以嘗試增加神經網絡的層數或每層的神經元數量。
# 改變激活函數：ReLU 是一個很常見的選擇，但有時候可以考慮其他激活函數，如 Leaky ReLU 或 ELU，這些可以避免 ReLU 的「死神經元」問題。
# 使用 Dropout：Dropout 是一種正則化技術，能幫助防止過擬合。你可以在某些層之間加入 Dropout 層。
# 訓練過程調整：

# 增加訓練輪次：2000 輪的訓練可能過多或過少，應該根據損失和準確率的趨勢進行調整。你可以觀察訓練過程中的損失和準確率變化，並進行早停（Early Stopping）來避免過擬合。
# 調整學習率：Adam 優化器的學習率可能需要調整。你可以嘗試不同的學習率，看看哪一個效果最好。
# 批次大小調整：批次大小對訓練速度和結果有影響。你可以試著調整批次大小（例如，32、128）來獲得更好的結果。
# 使用更多的訓練數據：

# 增強數據集：如果數據量不夠多，可以嘗試進行數據增強，例如使用交叉驗證來減少模型的偏差，或者收集更多樣本。
# 合成數據：可以考慮生成合成數據（例如 SMOTE）來平衡樣本數量，尤其是當某些類別的樣本過少時。















# 提高模型的準確性可以從以下幾個方面進行調整：

# 數據預處理：

# 特徵標準化：神經網絡模型通常對特徵的範圍和分佈較為敏感，因此對特徵進行標準化或正規化可以提高模型的效果。你可以使用 StandardScaler 或 MinMaxScaler 來標準化特徵數據。
# 處理缺失值：如果數據中有缺失值，應該進行處理。可以選擇丟棄缺失值行，或用均值/中位數等進行填補。
# 模型架構調整：

# 增加層數或神經元：如果模型容量過小，可能無法學到足夠的特徵。你可以嘗試增加神經網絡的層數或每層的神經元數量。
# 改變激活函數：ReLU 是一個很常見的選擇，但有時候可以考慮其他激活函數，如 Leaky ReLU 或 ELU，這些可以避免 ReLU 的「死神經元」問題。
# 使用 Dropout：Dropout 是一種正則化技術，能幫助防止過擬合。你可以在某些層之間加入 Dropout 層。
# 訓練過程調整：

# 增加訓練輪次：2000 輪的訓練可能過多或過少，應該根據損失和準確率的趨勢進行調整。你可以觀察訓練過程中的損失和準確率變化，並進行早停（Early Stopping）來避免過擬合。
# 調整學習率：Adam 優化器的學習率可能需要調整。你可以嘗試不同的學習率，看看哪一個效果最好。
# 批次大小調整：批次大小對訓練速度和結果有影響。你可以試著調整批次大小（例如，32、128）來獲得更好的結果。
# 使用更多的訓練數據：

# 增強數據集：如果數據量不夠多，可以嘗試進行數據增強，例如使用交叉驗證來減少模型的偏差，或者收集更多樣本。
# 合成數據：可以考慮生成合成數據（例如 SMOTE）來平衡樣本數量，尤其是當某些類別的樣本過少時。








# 提高模型的準確性可以從以下幾個方面進行調整：

# 數據預處理：

# 特徵標準化：神經網絡模型通常對特徵的範圍和分佈較為敏感，因此對特徵進行標準化或正規化可以提高模型的效果。你可以使用 StandardScaler 或 MinMaxScaler 來標準化特徵數據。
# 處理缺失值：如果數據中有缺失值，應該進行處理。可以選擇丟棄缺失值行，或用均值/中位數等進行填補。
# 模型架構調整：

# 增加層數或神經元：如果模型容量過小，可能無法學到足夠的特徵。你可以嘗試增加神經網絡的層數或每層的神經元數量。
# 改變激活函數：ReLU 是一個很常見的選擇，但有時候可以考慮其他激活函數，如 Leaky ReLU 或 ELU，這些可以避免 ReLU 的「死神經元」問題。
# 使用 Dropout：Dropout 是一種正則化技術，能幫助防止過擬合。你可以在某些層之間加入 Dropout 層。
# 訓練過程調整：

# 增加訓練輪次：2000 輪的訓練可能過多或過少，應該根據損失和準確率的趨勢進行調整。你可以觀察訓練過程中的損失和準確率變化，並進行早停（Early Stopping）來避免過擬合。
# 調整學習率：Adam 優化器的學習率可能需要調整。你可以嘗試不同的學習率，看看哪一個效果最好。
# 批次大小調整：批次大小對訓練速度和結果有影響。你可以試著調整批次大小（例如，32、128）來獲得更好的結果。
# 使用更多的訓練數據：

# 增強數據集：如果數據量不夠多，可以嘗試進行數據增強，例如使用交叉驗證來減少模型的偏差，或者收集更多樣本。
# 合成數據：可以考慮生成合成數據（例如 SMOTE）來平衡樣本數量，尤其是當某些類別的樣本過少時。