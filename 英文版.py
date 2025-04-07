# 引入所需的庫
import pandas as pd  # 用於數據處理
import numpy as np  # 用於數據操作
import tensorflow as tf  # 引入 TensorFlow 用於深度學習
from sklearn.preprocessing import StandardScaler  # 引入標準化工具
from sklearn.model_selection import StratifiedKFold  # 引入分層 K 折交叉驗證
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix  # 引入評估指標
from tensorflow.keras.models import Sequential, load_model  # 引入 Sequential 模型和保存/加載模型功能
from tensorflow.keras.layers import Dense, LSTM, Dropout  # 引入神經網絡層：全連接層、LSTM 層、Dropout 層
from tensorflow.keras.utils import to_categorical  # 用於將目標標籤轉換為分類格式
from tensorflow.keras.callbacks import ModelCheckpoint  # 引入模型檢查點回調
import matplotlib.pyplot as plt  # 用於畫圖
import seaborn as sns  # 用於畫圖，特別是熱圖
import os  # 用於文件和目錄操作

# 設定隨機種子以確保結果可重現
np.random.seed(42)
tf.random.set_seed(42)

def plot_training_history(history, fold_num=None):
    """
    Plot model training metrics
    
    Parameters:
    history: Model training history
    fold_num: Current cross-validation fold number (optional, shown in title if provided)
    """
    
    # Set up 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot loss curve
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    
    # Set title and labels for loss plot
    axes[0, 0].set_title('Model Loss', fontsize=12, pad=20, loc='left')
    if fold_num is not None:
        axes[0, 0].set_title(f'Fold {fold_num}', fontsize=12, pad=20, loc='right')
    
    axes[0, 0].set_xlabel('Epochs', fontsize=10)
    axes[0, 0].set_ylabel('Loss Value', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)

    # Plot accuracy curve
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    if 'val_accuracy' in history.history:
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    
    # Set title and labels for accuracy plot
    axes[0, 1].set_title('Model Accuracy', fontsize=12, pad=20, loc='left')
    if fold_num is not None:
        axes[0, 1].set_title(f'Fold {fold_num}', fontsize=12, pad=20, loc='right')
    
    axes[0, 1].set_xlabel('Epochs', fontsize=10)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)

    # Plot AUC curve if available
    if 'auc' in history.history and 'val_auc' in history.history:
        axes[1, 0].plot(history.history['auc'], label='Training AUC', linewidth=2)
        axes[1, 0].plot(history.history['val_auc'], label='Validation AUC', linewidth=2)
        
        # Set title and labels for AUC plot
        axes[1, 0].set_title('Model AUC', fontsize=12, pad=20, loc='left')
        if fold_num is not None:
            axes[1, 0].set_title(f'Fold {fold_num}', fontsize=12, pad=20, loc='right')
        
        axes[1, 0].set_xlabel('Epochs', fontsize=10)
        axes[1, 0].set_ylabel('AUC Value', fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, linestyle='--', alpha=0.7)

    # Plot learning rate curve if available
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate', linewidth=2, color='green')
        
        # Set title and labels for learning rate plot
        axes[1, 1].set_title('Learning Rate Changes', fontsize=12, pad=20, loc='left')
        if fold_num is not None:
            axes[1, 1].set_title(f'Fold {fold_num}', fontsize=12, pad=20, loc='right')
        
        axes[1, 1].set_xlabel('Epochs', fontsize=10)
        axes[1, 1].set_ylabel('Learning Rate Value', fontsize=12)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        axes[1, 1].set_yscale('log')  # Use log scale for learning rate

    # Adjust layout to ensure plots do not overlap
    plt.tight_layout()
    plt.show()


def load_and_preprocess_data(file_path):
    """
    加載並進行初步數據預處理
    
    參數：
    file_path: CSV 檔案的路徑
    
    返回：
    X: 特徵矩陣
    y: 目標變量
    feature_columns: 特徵欄位名稱列表
    """
    # 從 CSV 檔案讀取數據
    dataframe = pd.read_csv(file_path)
    
    # 定義用於預測的特徵列
    feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                       'thalach', 'exang', 'oldpeak', 'slope', 'thal']
    
    # 切分特徵 (X) 和目標變量 (y)
    X = dataframe[feature_columns].values
    y = dataframe['target'].values
    
    return X, y, feature_columns

def create_LSTM_model(input_dimension):
    """
    創建 LSTM 神經網絡模型
    
    參數：
    input_dimension: 輸入特徵的維度
    
    返回：
    編譯過的 Keras 模型
    """
    model = Sequential([
        # LSTM 層，用於處理序列數據
        LSTM(128, input_shape=(input_dimension, 1), activation='tanh', return_sequences=False),
        Dropout(0.5),  # 防止過擬合
        Dense(64, activation='relu'),  # 全連接層
        Dropout(0.3),  # 防止過擬合
        Dense(2, activation='softmax')  # 輸出層，二分類問題
    ])
    
    # 編譯模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Adam 優化器，學習率設為 0.0005
        loss='categorical_crossentropy',  # 類別交叉熵損失函數
        metrics=['accuracy', tf.keras.metrics.AUC()]  # 評估指標：準確度和 AUC
    )
    
    return model


def plot_confusion_matrix(true_labels, predicted_labels, fold_num=None):
    """
    繪製混淆矩陣的可視化圖
    
    參數：
    true_labels: 真實標籤（實際的類別標籤）
    predicted_labels: 模型預測的標籤
    fold_num: 當前交叉驗證的折數（可選，若提供則顯示在標題中）
    """
    # 定義類別名稱，這裡假設有兩個類別：沒有心臟病 (No Heart Disease) 和有心臟病 (Heart Disease)
    class_names = ['No Heart Disease', 'Heart Disease']
    
    # 計算混淆矩陣，這是根據真實標籤與預測標籤計算的
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # 使用 seaborn 畫出混淆矩陣的熱圖
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',  # annot=True 顯示數字，fmt='d' 代表顯示整數
                xticklabels=class_names, yticklabels=class_names)  # 顯示類別名稱
    
    # 如果提供了 fold_num，則顯示該折數
    if fold_num:
        plt.title('Confusion Matrix', fontsize=14, pad=20, loc='left')  # 設定標題的位置
        plt.title(f'Fold {fold_num}', fontsize=14, pad=20, loc='right')  # 顯示折數
    else:
        # 如果沒有提供 fold_num，只顯示標題
        plt.title('Confusion Matrix', fontsize=14, pad=20)
    
    # 設置 x 軸和 y 軸的標籤
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    
    # 顯示圖形
    plt.show()


def plot_roc_curve(true_labels, pred_probs, fold_num=None):
    """
    繪製 ROC 曲線
    
    參數：
    true_labels: 真實標籤
    pred_probs: 模型預測的標籤為正類的概率
    fold_num: 當前交叉驗證的折數（可選）
    
    返回：
    auc_value: 計算得到的 AUC 值
    """
    # 計算 ROC 曲線的點
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    # 計算 AUC 值
    auc_value = auc(fpr, tpr)
    
    # 繪製 ROC 曲線
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_value:.4f})')
    
    # 繪製隨機猜測的基準線
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
    # 設置圖形屬性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    
    # 設置標題
    if fold_num:
        plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, pad=20, loc='left')
        plt.title(f'Fold {fold_num}', fontsize=14, pad=20, loc='right')
    else:
        plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, pad=20)
    
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    return auc_value


def plot_accuracy_comparison(accuracy_values):
    """
    繪製所有折數的準確度比較圖
    
    參數：
    accuracy_values: 所有折數的準確度列表
    """
    # 準備數據
    folds = [f'Fold {i+1}' for i in range(len(accuracy_values))]
    mean_acc = np.mean(accuracy_values)
    
    # 創建條形圖
    plt.figure(figsize=(12, 8))
    bars = plt.bar(folds, accuracy_values, color='steelblue', alpha=0.7)
    
    # 添加平均線
    plt.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean Accuracy: {mean_acc:.4f}')
    
    # 在每個條上添加數值標籤
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 設置圖形屬性
    plt.title('Accuracy Comparison Across 5-Fold Cross Validation', fontsize=16, pad=20)
    plt.xlabel('Cross Validation Fold', fontsize=14)
    plt.ylabel('Accuracy Score', fontsize=14)
    plt.ylim([min(accuracy_values) - 0.05, max(accuracy_values) + 0.05])
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_LSTM_model(X, y, save_dir="E:\\心臟病"):
    """
    使用 K 折交叉驗證訓練 LSTM 模型，並保存最佳模型
    
    參數：
    X: 特徵矩陣
    y: 目標變量（標籤）
    save_dir: 保存模型的目錄路徑
    """
    # 確保保存目錄存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化 5 折交叉驗證
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []  # 用來儲存每折的準確度
    auc_scores = []       # 用來儲存每折的 AUC 值
    
    # 用於追蹤最佳模型
    best_fold = 0
    best_accuracy = 0
    best_auc = 0
    best_scaler = None
    
    # 進行 K 折交叉驗證
    for fold_num, (train_index, val_index) in enumerate(k_fold.split(X, y), 1):
        print(f"\n{'='*50}")
        print(f"Starting Fold {fold_num} of 5")
        print(f"{'='*50}")
        
        # 切分訓練集和驗證集
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # 特徵標準化處理
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).reshape(-1, X.shape[1], 1)  # 對訓練數據進行標準化
        X_val = scaler.transform(X_val).reshape(-1, X.shape[1], 1)  # 只對驗證數據進行轉換
        
        # 將標籤轉換為 one-hot 編碼格式
        y_train = to_categorical(y_train, num_classes=2)  # 假設有兩個類別
        y_val = to_categorical(y_val, num_classes=2)  # 同樣對驗證集的標籤進行 one-hot 編碼
        
        # 設置模型檢查點回調，保存每一折的最佳模型
        fold_model_path = os.path.join(save_dir, f'lstm_model_fold_{fold_num}.keras')
        checkpoint = ModelCheckpoint(
            fold_model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # 創建 LSTM 模型並訓練
        model = create_LSTM_model(X_train.shape[1])
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=500,  # 訓練輪數
            batch_size=128,  # 批量大小
            verbose=1,  # 顯示訓練過程
            callbacks=[checkpoint]  # 加入檢查點回調
        )
        
        # 加載該折的最佳模型權重
        model = load_model(fold_model_path)
        
        # 繪製訓練歷史圖
        plot_training_history(history, fold_num)
        
        # 預測驗證集結果
        y_pred_probs = model.predict(X_val)  # 獲取預測概率
        y_pred = np.argmax(y_pred_probs, axis=1)  # 取最大概率的類別作為預測結果
        y_val_true = np.argmax(y_val, axis=1)  # 真實標籤
        
        # 計算準確度
        accuracy = (y_pred == y_val_true).mean()  # 預測結果與真實標籤的相等比例
        accuracy_scores.append(accuracy)  # 儲存每一折的準確度
        
        # 繪製 ROC 曲線並計算 AUC
        auc_value = plot_roc_curve(y_val_true, y_pred_probs[:, 1], fold_num)  # 傳入正類的概率
        auc_scores.append(auc_value)  # 儲存每一折的 AUC 值
        
        # 顯示每一折的準確度和 AUC 值
        print(f"Fold {fold_num} - Accuracy: {accuracy:.4f}, AUC: {auc_value:.4f}")
        
        # 繪製混淆矩陣
        plot_confusion_matrix(y_val_true, y_pred, fold_num)
        
        # 更新最佳模型（基於準確度和 AUC 的綜合評估）
        combined_score = (accuracy + auc_value) / 2  # 簡單的組合指標
        best_combined = (best_accuracy + best_auc) / 2
        
        if combined_score > best_combined:
            best_fold = fold_num
            best_accuracy = accuracy
            best_auc = auc_value
            best_scaler = scaler
            
            # 保存最佳模型及其 Scaler
            best_model_path = os.path.join(save_dir, 'best_lstm_model.h5')
            model.save(best_model_path)
            
            # 保存 Scaler
            import joblib
            scaler_path = os.path.join(save_dir, 'best_model_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            
            # 保存模型性能資訊
            with open(os.path.join(save_dir, 'best_model_info.txt'), 'w') as f:
                f.write(f"Best Model: Fold {best_fold}\n")
                f.write(f"Accuracy: {best_accuracy:.6f}\n")
                f.write(f"AUC: {best_auc:.6f}\n")
                f.write(f"Combined Score: {combined_score:.6f}\n")
    
    # 繪製所有折數的準確度比較圖
    plot_accuracy_comparison(accuracy_scores)
    
    # 輸出所有折數的平均性能指標
    print(f"\n{'='*50}")
    print(f"Cross Validation Results Summary")
    print(f"{'='*50}")
    print(f"Accuracy scores for all folds: {[f'{acc:.4f}' for acc in accuracy_scores]}")
    print(f"AUC scores for all folds: {[f'{auc_val:.4f}' for auc_val in auc_scores]}")
    print(f"Average Accuracy: {np.mean(accuracy_scores):.4f} (±{np.std(accuracy_scores):.4f})")
    print(f"Average AUC: {np.mean(auc_scores):.4f} (±{np.std(auc_scores):.4f})")
    print(f"{'='*50}")
    
    # 輸出最佳模型資訊
    print(f"\n{'='*50}")
    print(f"Best Model Information")
    print(f"{'='*50}")
    print(f"Best Model: Fold {best_fold}")
    print(f"Accuracy: {best_accuracy:.6f}")
    print(f"AUC: {best_auc:.6f}")
    print(f"Model and Scaler saved to: {save_dir}")
    print(f"{'='*50}")
    
    # 返回最佳模型的相關資訊
    return {
        'best_fold': best_fold,
        'best_accuracy': best_accuracy,
        'best_auc': best_auc,
        'model_path': os.path.join(save_dir, 'best_lstm_model.h5'),
        'scaler_path': os.path.join(save_dir, 'best_model_scaler.pkl')
    }

# 添加預測新數據的函數
def predict_with_best_model(model_path, scaler_path, new_data):
    """
    使用訓練好的最佳模型進行預測
    
    參數：
    model_path: 模型文件的路徑
    scaler_path: 標準化縮放器的路徑
    new_data: 要預測的新數據 (未縮放的原始數據)
    
    返回：
    predictions: 預測的類別
    probabilities: 預測的概率
    """
    import joblib
    
    # 加載模型和縮放器
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # 縮放新數據
    scaled_data = scaler.transform(new_data).reshape(-1, new_data.shape[1], 1)
    
    # 進行預測
    pred_probs = model.predict(scaled_data)
    predictions = np.argmax(pred_probs, axis=1)
    
    return predictions, pred_probs

# Main program entry point
if __name__ == "__main__":
    
    file_path = r"E:\心臟病\heart2.csv"  # Data file path
    X, y, feature_columns = load_and_preprocess_data(file_path)
    best_model_info = train_LSTM_model(X, y)
    
    print(f"\nTraining complete! Best model saved at: {best_model_info['model_path']}")
    print(f"To use this model for prediction, use the function: predict_with_best_model()")



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