# 引入所需的庫
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

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

# MLP 模型結構
def create_MLP_model(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.001)),  
        BatchNormalization(),  # 增加 BN 提高穩定性
        Dropout(0.4),  

        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),  
        BatchNormalization(),
        Dropout(0.3),  

        Dense(2, activation='softmax')  # 二分類輸出
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC()])
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


# 訓練 MLP 模型（加入 Learning Rate 調整，移除 EarlyStopping）
def train_MLP_model(X, y):
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    avg_accuracy, avg_auc = [], []
    
    for fold_num, (train_index, val_index) in enumerate(k_fold.split(X, y), 1):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # 標準化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # One-hot 編碼
        y_train = to_categorical(y_train, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)

        # 創建 MLP 模型
        model = create_MLP_model(X_train.shape[1])

        # 設置 ReduceLROnPlateau
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

        # 訓練模型
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=200,  # 降低 epoch，並用 ReduceLROnPlateau 控制學習率
                            batch_size=128,
                            verbose=1,
                            callbacks=[reduce_lr])

        # 繪製訓練歷史圖
        plot_training_history(history, fold_num)
        
        # 預測驗證集結果
        y_pred = np.argmax(model.predict(X_val), axis=1)  # 取最大概率的類別作為預測結果
        y_val_true = np.argmax(y_val, axis=1)  # 真實標籤
        
        # 計算準確度
        accuracy = (y_pred == y_val_true).mean()  # 預測結果與真實標籤的相等比例
        avg_accuracy.append(accuracy)  # 儲存每一折的準確度
        
        # 計算 AUC（曲線下面積）
        fpr, tpr, _ = roc_curve(y_val_true, y_pred)  # 計算偽陽性率 (fpr) 和真正率 (tpr)
        auc_value = auc(fpr, tpr)  # 計算 AUC
        avg_auc.append(auc_value)  # 儲存每一折的 AUC 值
        
        # 顯示每一折的準確度和 AUC 值
        print(f"Fold {fold_num} - Accuracy: {accuracy:.4f}, AUC: {auc_value:.4f}")
        
        # 繪製混淆矩陣
        plot_confusion_matrix(y_val_true, y_pred, fold_num)
    
    # 輸出所有折數的平均性能指標
    print(f"\nAverage Accuracy: {np.mean(avg_accuracy):.4f}, Average AUC: {np.mean(avg_auc):.4f}")

# Main program entry point
if __name__ == "__main__":
    file_path = r"E:\心臟病\heart2.csv"  # Data file path
    X, y, feature_columns = load_and_preprocess_data(file_path)
    train_MLP_model(X, y)
