# 引入所需的庫
import pandas as pd  # 用於數據處理
import numpy as np  # 用於數據操作
import tensorflow as tf  # 引入 TensorFlow 用於深度學習
from sklearn.preprocessing import StandardScaler  # 引入標準化工具
from sklearn.model_selection import StratifiedKFold  # 引入分層 K 折交叉驗證
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix  # 引入評估指標
from tensorflow.keras.models import Sequential  # 引入 Sequential 模型
from tensorflow.keras.layers import Dense, LSTM, Dropout  # 引入神經網絡層：全連接層、LSTM 層、Dropout 層
from tensorflow.keras.utils import to_categorical  # 用於將目標標籤轉換為分類格式
import matplotlib.pyplot as plt  # 用於畫圖
import seaborn as sns  # 用於畫圖，特別是熱圖

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

def create_improved_LSTM_model(input_dimension):
    """
    創建一個改進的LSTM模型
    
    參數:
    input_dimension: 輸入特徵的維度
    
    返回:
    編譯好的Keras模型
    """
    model = Sequential([
        # 雙向LSTM可以從兩個方向學習序列特徵
        tf.keras.layers.Bidirectional(
            LSTM(256, activation='tanh', return_sequences=True,
                recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
            input_shape=(input_dimension, 1)
        ),
        Dropout(0.5),
        
        tf.keras.layers.Bidirectional(
            LSTM(128, activation='tanh', return_sequences=False)
        ),
        Dropout(0.5),
        
        tf.keras.layers.BatchNormalization(),
        
        Dense(128, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.01),
              kernel_initializer='he_normal'),
        Dropout(0.4),
        
        tf.keras.layers.BatchNormalization(),
        
        Dense(64, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01),
              kernel_initializer='he_normal'),
        Dropout(0.3),
        
        Dense(2, activation='softmax')
    ])
    
    # Use a constant learning rate instead of a scheduler
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Constant learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
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


def train_improved_LSTM_model(X, y, feature_columns):
    """
    使用改進的方法訓練LSTM模型
    
    參數:
    X: 特徵矩陣
    y: 目標變量
    feature_columns: 特徵列名
    """
    # 進行特徵工程
    X_enhanced = enhanced_feature_engineering(X, feature_columns)
    
    # 初始化5折交叉驗證
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_models = []  # 存儲所有訓練好的模型
    all_scalers = []  # 存儲所有的數據標準化器
    fold_results = {'accuracy': [], 'auc': [], 'precision': [], 'recall': [], 'f1': []}
    
    # 進行K折交叉驗證
    for fold_num, (train_index, val_index) in enumerate(k_fold.split(X_enhanced, y), 1):
        print(f"\n{'='*50}\nTraining Fold {fold_num}\n{'='*50}")
        
        # 分割訓練集和驗證集
        X_train, X_val = X_enhanced[train_index], X_enhanced[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # 特徵標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        all_scalers.append(scaler)
        
        # 數據增強
        X_train_scaled, y_train = augment_data(X_train_scaled.reshape(-1, X_train.shape[1], 1), y_train)
        X_val_scaled = X_val_scaled.reshape(-1, X_val.shape[1], 1)
        
        # 轉換為分類格式
        y_train_cat = to_categorical(y_train, num_classes=2)
        y_val_cat = to_categorical(y_val, num_classes=2)
        
        # 創建模型
        model = create_improved_LSTM_model(X_train_scaled.shape[1])
        
        # 設置回調函數
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        # 訓練模型
# Training model without ReduceLROnPlateau
        history = model.fit(
            X_train_scaled, y_train_cat,
            validation_data=(X_val_scaled, y_val_cat),
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping],  # Remove reduce_lr from here
            verbose=1
        )
        
        # 保存模型
        all_models.append(model)
        
        # 繪製訓練歷史
        plot_training_history(history, fold_num)
        
        # 評估模型
        y_pred_prob = model.predict(X_val_scaled)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_val_true = np.argmax(y_val_cat, axis=1)
        
        # 計算評估指標
        accuracy = (y_pred == y_val_true).mean()
        fpr, tpr, _ = roc_curve(y_val_true, y_pred_prob[:, 1])
        auc_value = auc(fpr, tpr)
        
        # 記錄評估結果
        fold_results['accuracy'].append(accuracy)
        fold_results['auc'].append(auc_value)
        
        # 計算並記錄精確度、召回率和F1分數
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_val_true, y_pred)
        recall = recall_score(y_val_true, y_pred)
        f1 = f1_score(y_val_true, y_pred)
        
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['f1'].append(f1)
        
        print(f"Fold {fold_num} Results:")
        print(f"Accuracy: {accuracy:.4f}, AUC: {auc_value:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # 繪製混淆矩陣
        plot_confusion_matrix(y_val_true, y_pred, fold_num)
        
        # 繪製ROC曲線
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_value:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Fold {fold_num}')
        plt.legend(loc="lower right")
        plt.show()
    
    # 輸出所有折數的平均性能指標
    print("\n" + "="*50)
    print("Average Results Across All Folds:")
    for metric, values in fold_results.items():
        print(f"Average {metric.capitalize()}: {np.mean(values):.4f}")
    
    return all_models, all_scalers, fold_results

# 特徵工程函數
def enhanced_feature_engineering(X, feature_columns):
    """
    進行增強的特徵工程
    
    參數:
    X: 特徵矩陣
    feature_columns: 特徵列名
    
    返回:
    增強後的特徵矩陣
    """
    # 創建DataFrame以便進行特徵工程
    X_df = pd.DataFrame(X, columns=feature_columns)
    
    # 創建新特徵 - 血壓與心率的比例
    X_df['bp_hr_ratio'] = X_df['trestbps'] / X_df['thalach']
    
    # 創建年齡分組
    X_df['age_group'] = pd.cut(X_df['age'], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3])
    
    # 創建膽固醇分組
    X_df['chol_level'] = pd.cut(X_df['chol'], bins=[0, 200, 240, 600], labels=[0, 1, 2])
    
    # 特徵間的交互作用
    X_df['age_chol'] = X_df['age'] * X_df['chol'] / 1000  # 縮放以防數值過大
    X_df['age_trestbps'] = X_df['age'] * X_df['trestbps'] / 1000
    X_df['chol_trestbps'] = X_df['chol'] * X_df['trestbps'] / 10000
    
    # 非線性變換
    X_df['oldpeak_squared'] = X_df['oldpeak'] ** 2
    X_df['trestbps_log'] = np.log1p(X_df['trestbps'])
    
    # 將分類特徵進行獨熱編碼
    categorical_features = ['cp', 'restecg', 'slope', 'thal', 'age_group', 'chol_level']
    X_encoded = pd.get_dummies(X_df, columns=categorical_features, drop_first=True)
    
    return X_encoded.values

# 數據增強函數
def augment_data(X_train, y_train):
    """
    針對不平衡類別進行數據增強
    
    參數:
    X_train: 訓練特徵
    y_train: 訓練標籤
    
    返回:
    增強後的訓練特徵和標籤
    """
    try:
        from imblearn.over_sampling import SMOTE
        
        # 檢查類別分布
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"Original class distribution: {class_dist}")
        
        # 如果類別不平衡，則應用SMOTE
        min_count = min(counts)
        if min_count / max(counts) < 0.8:  # 如果少數類別比例小於80%
            print("Applying SMOTE for class balancing")
            smote = SMOTE(random_state=42)
            X_flat = X_train.reshape(X_train.shape[0], -1)
            X_resampled, y_resampled = smote.fit_resample(X_flat, y_train)
            
            # 檢查重采樣後的類別分布
            unique, counts = np.unique(y_resampled, return_counts=True)
            print(f"After SMOTE class distribution: {dict(zip(unique, counts))}")
            
            # 將數據調整回原來的形狀
            X_resampled = X_resampled.reshape(-1, X_train.shape[1], X_train.shape[2])
            return X_resampled, y_resampled
        else:
            print("Class distribution is already balanced, skipping SMOTE")
            return X_train, y_train
    except ImportError:
        print("Warning: imblearn not installed. Skipping SMOTE.")
        return X_train, y_train

# 改進的LSTM模型創建函數
def create_improved_LSTM_model(input_dimension):
    """
    創建一個改進的LSTM模型
    
    參數:
    input_dimension: 輸入特徵的維度
    
    返回:
    編譯好的Keras模型
    """
    model = Sequential([
        # 雙向LSTM可以從兩個方向學習序列特徵
        tf.keras.layers.Bidirectional(
            LSTM(256, activation='tanh', return_sequences=True,
                recurrent_regularizer=tf.keras.regularizers.l2(0.01)),
            input_shape=(input_dimension, 1)
        ),
        Dropout(0.5),
        
        tf.keras.layers.Bidirectional(
            LSTM(128, activation='tanh', return_sequences=False)
        ),
        Dropout(0.5),
        
        tf.keras.layers.BatchNormalization(),
        
        Dense(128, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.01),
              kernel_initializer='he_normal'),
        Dropout(0.4),
        
        tf.keras.layers.BatchNormalization(),
        
        Dense(64, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01),
              kernel_initializer='he_normal'),
        Dropout(0.3),
        
        Dense(2, activation='softmax')
    ])
    
    # 使用學習率調度器
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=100,
        decay_rate=0.9,
        staircase=True)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

# 集成預測函數  
def ensemble_prediction(X_test, models, scalers, X_original_shape):
    """
    使用模型集成進行預測
    
    參數:
    X_test: 測試特徵
    models: 訓練好的模型列表
    scalers: 標準化器列表
    X_original_shape: 原始特徵的形狀，用於特徵工程

    返回:
    集成預測結果
    """
    predictions = []
    
    # 對每個模型進行預測
    for i, (model, scaler) in enumerate(zip(models, scalers)):
        # 標準化數據
        X_test_scaled = scaler.transform(X_test).reshape(-1, X_test.shape[1], 1)
        
        # 獲取預測概率
        pred_prob = model.predict(X_test_scaled)
        predictions.append(pred_prob)
    
    # 平均所有模型的預測結果
    ensemble_pred_prob = np.mean(predictions, axis=0)
    ensemble_pred = np.argmax(ensemble_pred_prob, axis=1)
    
    return ensemble_pred, ensemble_pred_prob

def plot_confusion_matrix(true_labels, predicted_labels, fold_num=None):
    """
    繪製混淆矩陣的可視化圖
    
    參數：
    true_labels: 真實標籤（實際的類別標籤）
    predicted_labels: 模型預測的標籤
    fold_num: 當前交叉驗證的折數（可選，若提供則顯示在標題中）
    """
    # 計算混淆矩陣
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(8, 6))
    
    # 定義類別名稱，這裡假設有兩個類別：沒有心臟病 (No Heart Disease) 和有心臟病 (Heart Disease)
    class_names = ['No Heart Disease', 'Heart Disease']
    
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

# 主函數調用
if __name__ == "__main__":
    file_path = r"E:\心臟病\heart2.csv"  # 數據文件路徑
    X, y, feature_columns = load_and_preprocess_data(file_path)
    
    # 訓練改進的模型
    models, scalers, results = train_improved_LSTM_model(X, y, feature_columns)
    
    # 保存最佳模型
    best_model_index = np.argmax(results['accuracy'])
    best_model = models[best_model_index]
    best_model.save('best_heart_disease_model.h5')
    print(f"Best model (fold {best_model_index+1}) saved with accuracy: {results['accuracy'][best_model_index]:.4f}")





##要提高 F1 分數和準確率，您可以調整以下幾個關鍵方面：

# 特徵工程優化：

# 添加更多臨床相關特徵，如 BMI 指數計算（如果有身高體重數據）
# 尋找更有意義的特徵組合，例如年齡與血壓的交互作用
# 嘗試不同的特徵選擇方法，如 PCA 或 RFE (Recursive Feature Elimination)


# 模型架構調整：

# 減少模型復雜度，如果出現過擬合（例如減少 LSTM 的層數或單元數）
# 增加模型復雜度，如果出現欠擬合（例如增加更多層或更多單元）
# 嘗試不同的 LSTM 變體，如 GRU 或融合 CNN-LSTM 混合模型


# 超參數調整：

# 學習率：嘗試更小的起始學習率（如 0.0005 或 0.0001）
# Batch size：嘗試不同的批次大小（16, 32, 64）
# Dropout 率：調整 dropout 比例（0.2-0.5 之間）
# 正則化：添加 L1 或 L2 正則化以減少過擬合


# 類別不平衡處理：

# 使用類別權重（class_weight 參數）
# 使用重採樣技術，如 SMOTE 或 RandomOverSampler
# 調整閾值（不一定使用 0.5 作為預測閾值）


# 損失函數與評估指標：

# 嘗試 focal loss 來處理類別不平衡
# 使用 F1 分數作為監控指標而不是準確率


# 數據增強與預處理：

# 強化數據標準化處理
# 移除離群值或進行非線性轉換
# 使用更穩健的數據缺失處理策略


# 集成方法：

# 結合多個不同的模型預測（如 LSTM、XGBoost、Random Forest）
# 實施堆疊（stacking）或投票（voting）策略


# 交叉驗證策略：

# 使用分層交叉驗證確保各折中類別分布一致
# 增加交叉驗證的折數（例如從 5 折增加到 10 折）



# 您可以根據模型當前的表現有針對性地進行調整，例如：

# 如果模型過擬合（訓練準確率高但驗證準確率低），增加正則化強度
# 如果模型欠擬合（訓練和驗證準確率都低），增加模型複雜度或特徵
# 如果特定類別的預測效果較差，使用類別權重或調整閾值

# 每次調整後，記得追蹤 F1 分數和精確率/召回率的變化，以確保改進確實發生在您關注的指標上。