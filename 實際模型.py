import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# 設定隨機種子以確保結果可重現
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(file_path, test_size=0.2):
    """
    加載數據、進行特徵工程並分割為訓練集和測試集
    
    參數：
    file_path: CSV 檔案的路徑
    test_size: 測試集比例，默認為 0.2
    
    返回：
    X_train, X_test: 訓練和測試特徵
    y_train, y_test: 訓練和測試標籤
    feature_columns: 特徵欄位名稱
    scaler: 已擬合的標準化器，用於將來的數據轉換
    """
    # 從 CSV 檔案讀取數據
    dataframe = pd.read_csv(file_path)
    
    # 查看數據中是否有缺失值，並處理
    print("檢查缺失值：")
    print(dataframe.isnull().sum())
    
    # 處理可能的缺失值
    dataframe = dataframe.fillna(dataframe.median())
    
    # 定義用於預測的特徵列
    feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'thal']
    
    # 特徵工程：添加新特徵
    # 年齡與心率的比率（生理意義：年齡越大，最大心率通常越低）
    dataframe['age_thalach_ratio'] = dataframe['age'] / (dataframe['thalach'] + 1)
    feature_columns.append('age_thalach_ratio')
    
    # 血壓與心率的比率（可能與心臟負荷有關）
    dataframe['bp_thalach_ratio'] = dataframe['trestbps'] / (dataframe['thalach'] + 1)
    feature_columns.append('bp_thalach_ratio')
    
    # 膽固醇與年齡的比率（年齡標準化的膽固醇水平）
    dataframe['chol_age_ratio'] = dataframe['chol'] / (dataframe['age'] + 1)
    feature_columns.append('chol_age_ratio')
    
    # 切分特徵 (X) 和目標變量 (y)
    X = dataframe[feature_columns].values
    y = dataframe['target'].values
    
    # 特徵選擇：使用隨機森林找出最重要的特徵
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 顯示特徵重要性
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n特徵重要性：")
    print(importance_df)
    
    # 選擇前 90% 重要的特徵
    selector = SelectFromModel(rf, threshold="mean", prefit=True)
    X_selected = selector.transform(X)
    selected_indices = selector.get_support()
    selected_features = [feature_columns[i] for i in range(len(feature_columns)) if selected_indices[i]]
    
    print(f"\n選擇的特徵 ({len(selected_features)}/{len(feature_columns)}):")
    print(selected_features)
    
    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # 特徵標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, selected_features, scaler, selector

def create_LSTM_model(input_dimension, dropout_rate=0.5):
    """
    創建改進的 LSTM 神經網絡模型
    
    參數：
    input_dimension: 輸入特徵的維度
    dropout_rate: Dropout 層的比率，用於防止過擬合
    
    返回：
    編譯過的 Keras 模型
    """
    model = Sequential([
        # 雙向 LSTM，可以從兩個方向學習序列模式
        Bidirectional(LSTM(128, activation='tanh', return_sequences=True, 
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)), 
                      input_shape=(input_dimension, 1)),
        BatchNormalization(),  # 加速訓練並提供輕微的正則化效果
        Dropout(dropout_rate),  # 防止過擬合
        
        # 第二層 LSTM
        Bidirectional(LSTM(64, activation='tanh', return_sequences=False)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # 全連接層
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # 輸出層，二分類問題
        Dense(2, activation='softmax')
    ])
    
    # 編譯模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), 
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall()]
    )
    
    return model

def train_with_cross_validation(X, y, input_dimension, n_splits=5):
    """
    使用交叉驗證訓練模型
    
    參數：
    X: 特徵數據
    y: 標籤
    input_dimension: 輸入特徵的維度
    n_splits: 交叉驗證拆分數
    
    返回：
    最佳模型
    訓練歷史
    """
    # 準備交叉驗證
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_aucs = []
    best_accuracy = 0
    best_model = None
    best_history = None
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n訓練折疊 {fold+1}/{n_splits}")
        
        # 分割數據
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # 重塑數據以適應 LSTM 的輸入格式
        X_train_reshaped = X_train_fold.reshape(-1, input_dimension, 1)
        X_val_reshaped = X_val_fold.reshape(-1, input_dimension, 1)
        
        # 將標籤轉換為 one-hot 編碼
        y_train_cat = to_categorical(y_train_fold, num_classes=2)
        y_val_cat = to_categorical(y_val_fold, num_classes=2)
        
        # 創建模型
        model = create_LSTM_model(input_dimension)
        
        # 回調函數 - 修復 .h5 問題
        callbacks = [
            # 提前停止，避免過擬合
            EarlyStopping(
                monitor='val_accuracy',
                patience=30,
                restore_best_weights=True
            ),
            # 動態調整學習率
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.00001
            )
            # 移除 ModelCheckpoint，以避免檔案格式錯誤
        ]
        
        # 訓練模型
        history = model.fit(
            X_train_reshaped, y_train_cat,
            validation_data=(X_val_reshaped, y_val_cat),
            epochs=300,  # 足夠的訓練輪次，配合 EarlyStopping
            batch_size=4,
            callbacks=callbacks,
            verbose=1
        )
        
        # 評估模型
        y_val_pred = model.predict(X_val_reshaped)
        val_accuracy = accuracy_score(y_val_fold, np.argmax(y_val_pred, axis=1))
        
        # 計算 AUC
        y_val_true_one_hot = to_categorical(y_val_fold, num_classes=2)
        val_auc = roc_auc_score(y_val_true_one_hot, y_val_pred)
        
        print(f"折疊 {fold+1} 驗證準確率: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
        fold_accuracies.append(val_accuracy)
        fold_aucs.append(val_auc)
        
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
            best_history = history
    
    print(f"\n交叉驗證平均準確率: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"交叉驗證平均 AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    
    return best_model, best_history

def train_final_model(X_train, y_train, X_test, y_test, input_dimension):
    """
    訓練最終模型
    
    參數：
    X_train, y_train: 訓練數據和標籤
    X_test, y_test: 測試數據和標籤
    input_dimension: 輸入特徵的維度
    
    返回：
    訓練好的模型
    模型訓練歷史
    """
    # 重塑數據以適應 LSTM 的輸入格式
    X_train_reshaped = X_train.reshape(-1, input_dimension, 1)
    X_test_reshaped = X_test.reshape(-1, input_dimension, 1)
    
    # 將標籤轉換為 one-hot 編碼
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)
    
    # 創建模型
    model = create_LSTM_model(input_dimension)
    
    # 回調函數 - 修復 .h5 問題
    callbacks = [
        # 提前停止，避免過擬合
        EarlyStopping(
            monitor='val_accuracy',
            patience=30,
            restore_best_weights=True
        ),
        # 動態調整學習率
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001
        )
        # 移除 ModelCheckpoint，以避免檔案格式錯誤
    ]
    
    # 訓練模型
    history = model.fit(
        X_train_reshaped, y_train_cat,
        validation_data=(X_test_reshaped, y_test_cat),
        epochs=300,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def visualize_training_history(history):
    """
    視覺化訓練歷史
    
    參數：
    history: 模型訓練歷史
    """
    # 繪製準確率歷史
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 繪製損失歷史
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test):
    """
    評估模型並顯示結果
    
    參數：
    model: 訓練好的模型
    X_test: 測試數據
    y_test: 測試標籤
    
    返回：
    準確率（float）
    """
    # 重塑測試數據
    X_test_reshaped = X_test.reshape(-1, X_test.shape[1], 1)
    
    # 獲取預測結果
    y_pred_probs = model.predict(X_test_reshaped)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # One-hot 編碼的真實標籤，用於 ROC 曲線
    y_test_cat = to_categorical(y_test, num_classes=2)
    
    # 計算準確率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ 最終模型測試集準確率: {accuracy:.4f}\n")
    
    # 混淆矩陣
    class_names = ['無心臟病', '有心臟病']
    cm = tf.math.confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('最終模型混淆矩陣')
    plt.xlabel('預測標籤')
    plt.ylabel('真實標籤')
    plt.tight_layout()
    plt.show()
    
    # ROC 曲線
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC 曲線 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('偽陽性率 (False Positive Rate)')
    plt.ylabel('真陽性率 (True Positive Rate)')
    plt.title('ROC 曲線')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    
    # PR 曲線
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_probs[:, 1])
    plt.plot(recall, precision, label='Precision-Recall 曲線')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall 曲線')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()
    
    # 顯示分類報告
    print("\n📊 分類報告:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return accuracy, roc_auc

def save_deployment_model(model, scaler, selector, feature_names, model_dir='./model'):
    """
    保存模型和相關轉換器用於部署
    
    參數：
    model: 訓練好的模型
    scaler: 標準化器
    selector: 特徵選擇器
    feature_names: 原始特徵名稱列表
    model_dir: 保存模型的目錄
    """
    # 創建目錄（如果不存在）
    os.makedirs(model_dir, exist_ok=True)
    
    # 檢查您的 TensorFlow 版本，決定模型保存方式
    if tf.__version__.startswith('2.'):
        # TF 2.x 版本
        try:
            # 首先嘗試 .h5 格式
            save_model(model, os.path.join(model_dir, 'heart_disease_model.h5'))
        except:
            # 如果失敗，嘗試舊版的 SavedModel 格式
            tf.saved_model.save(model, os.path.join(model_dir, 'heart_disease_model'))
            print("以 SavedModel 格式保存模型")
    else:
        # 舊版本的 TF
        save_model(model, os.path.join(model_dir, 'heart_disease_model.h5'))
    
    # 保存標準化器
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    # 保存特徵選擇器
    joblib.dump(selector, os.path.join(model_dir, 'feature_selector.pkl'))
    
    # 保存特徵名稱列表
    with open(os.path.join(model_dir, 'feature_names.txt'), 'w') as f:
        f.write('\n'.join(feature_names))
    
    print(f"模型和轉換器已保存至 {model_dir}")

def predict_heart_disease(patient_data, model_dir='./model'):
    """
    使用已訓練的模型進行心臟病預測
    
    參數：
    patient_data: 患者數據，字典格式的特徵
    model_dir: 模型目錄
    
    返回：
    預測結果和概率
    """
    # 根據模型保存格式決定載入方式
    model_path = os.path.join(model_dir, 'heart_disease_model.h5')
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model_path = os.path.join(model_dir, 'heart_disease_model')
        if os.path.exists(model_path):
            model = tf.saved_model.load(model_path)
        else:
            raise FileNotFoundError("找不到已保存的模型")
    
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    selector = joblib.load(os.path.join(model_dir, 'feature_selector.pkl'))
    
    # 讀取特徵名稱
    with open(os.path.join(model_dir, 'feature_names.txt'), 'r') as f:
        feature_names = f.read().splitlines()
    
    # 添加派生特徵
    patient_data['age_thalach_ratio'] = patient_data['age'] / (patient_data['thalach'] + 1)
    patient_data['bp_thalach_ratio'] = patient_data['trestbps'] / (patient_data['thalach'] + 1)
    patient_data['chol_age_ratio'] = patient_data['chol'] / (patient_data['age'] + 1)
    
    # 準備特徵，確保順序正確
    input_features = []
    for feature in feature_names:
        if feature in patient_data:
            input_features.append(patient_data[feature])
    
    # 轉換為 numpy 數組
    patient_features = np.array([input_features])
    
    # 特徵選擇
    patient_features_selected = selector.transform(patient_features)
    
    # 標準化特徵
    patient_features_scaled = scaler.transform(patient_features_selected)
    
    # 重塑數據以適應 LSTM 輸入格式
    patient_features_reshaped = patient_features_scaled.reshape(-1, patient_features_scaled.shape[1], 1)
    
    # 預測
    prediction_probs = model.predict(patient_features_reshaped)[0]
    prediction_class = np.argmax(prediction_probs)
    
    # 結果
    result = {
        'prediction': '有心臟病' if prediction_class == 1 else '無心臟病',
        'probability': float(prediction_probs[prediction_class]),
        'heart_disease_prob': float(prediction_probs[1])
    }
    
    return result

def log_roc_score(model, X_test, y_test, log_file='model_performance_log.txt'):
    """
    記錄模型的 ROC 分數到日誌文件，用於追蹤性能改進
    
    參數：
    model: 訓練好的模型
    X_test: 測試數據
    y_test: 測試標籤
    log_file: 日誌文件名
    """
    # 重塑測試數據
    X_test_reshaped = X_test.reshape(-1, X_test.shape[1], 1)
    
    # 獲取預測結果
    y_pred_probs = model.predict(X_test_reshaped)
    
    # One-hot 編碼的真實標籤
    y_test_cat = to_categorical(y_test, num_classes=2)
    
    # 計算 ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # 獲取當前時間
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 記錄到文件
    with open(log_file, 'a') as f:
        f.write(f"{current_time}, AUC: {roc_auc:.4f}\n")
    
    return roc_auc

# 主程序
def main():
    # 資料文件路徑
    file_path = r"E:\心臟病\heart2.csv"  # 請更改為您的實際路徑
    
    print("======= 心臟病預測模型訓練 (改進版) =======")
    
    # 加載和預處理數據
    X_train, X_test, y_train, y_test, selected_features, scaler, selector = load_and_preprocess_data(file_path)
    
    print(f"\n最終選擇的特徵數量: {X_train.shape[1]}")
    
    # 使用交叉驗證訓練模型
    print("\n使用 5 折交叉驗證訓練模型...")
    cv_model, cv_history = train_with_cross_validation(
        np.concatenate((X_train, X_test)), 
        np.concatenate((y_train, y_test)), 
        X_train.shape[1], 
        n_splits=5
    )
    
    # 在最終數據上訓練模型
    print("\n在完整訓練集上訓練最終模型...")
    final_model, history = train_final_model(X_train, y_train, X_test, y_test, X_train.shape[1])
    
    # 視覺化訓練歷史
    visualize_training_history(history)
    
    # 評估模型
    accuracy, roc_auc = evaluate_model(final_model, X_test, y_test)
    
    # 記錄模型性能
    log_roc_score(final_model, X_test, y_test)
    
    # 保存模型和轉換器用於部署
    # 獲取所有特徵名稱（包括工程特徵）
    all_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'thal', 
                   'age_thalach_ratio', 'bp_thalach_ratio', 'chol_age_ratio']
    
    save_deployment_model(final_model, scaler, selector, all_features)
    
    # 示範如何使用模型進行預測
    sample_patient = {
        'age': 63,
        'sex': 1,  # 男性
        'cp': 3,   # 典型心絞痛
        'trestbps': 150,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 180,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'thal': 2
    }
    
    result = predict_heart_disease(sample_patient)
    print("\n預測結果示例:")
    print(f"預測: {result['prediction']}")
    print(f"心臟病可能性: {result['heart_disease_prob']:.2%}")
    
    print("\n======= 模型訓練與評估完成 =======")

if __name__ == "__main__":
    # 導入額外需要的庫
    from sklearn.metrics import roc_auc_score
    main()