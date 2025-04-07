import os  # 導入操作系統相關功能的模組
import numpy as np  # 導入 NumPy，用於數值計算
import joblib  # 導入 joblib，用於模型和數據的序列化
from flask import Flask, render_template, request  # 導入 Flask 相關模組
from tensorflow import keras  # 導入 Keras，用於加載深度學習模型
import logging  # 導入 logging，用於記錄日誌


# 設置日誌級別為 DEBUG
logging.basicConfig(level=logging.DEBUG)

# 初始化 Flask 應用
app = Flask(__name__, template_folder='templates')
basedir = os.path.abspath(os.path.dirname(__file__))  # 獲取當前文件夾的絕對路徑

# 加載模型和標準化器的路徑
model_path = os.path.join(basedir, 'heart_disease_model.keras')
scaler_path = os.path.join(basedir, 'scaler.pkl')

try:
    # 嘗試加載深度學習模型和標準化器
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    logging.info("模型和標準化器成功加載")
except Exception as e:
    # 如果加載失敗，記錄錯誤信息
    logging.error(f"加載模型或標準化器時發生錯誤: {str(e)}")
    model = None
    scaler = None

# 定義模型需要的特徵，去除膽固醇、收縮壓、舒張壓和血糖
features = [
    'age', 'male', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
    'prevalentStroke', 'prevalentHyp', 'diabetes', 'BMI', 'heartRate'
]


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None  # 初始化預測結果
    risk_level = None  # 初始化風險等級
    
    if request.method == 'POST':
        logging.info("接收到 POST 請求")
        try:
            # 收集從表單獲取的輸入數據
            input_data = [float(request.form.get(feature, 0)) for feature in features]
            logging.info(f"輸入數據：{input_data}")
            
            # 檢查模型和標準化器是否加載成功
            if model is not None and scaler is not None:
                # 將輸入數據轉換為 NumPy 陣列並進行標準化
                input_array = np.array(input_data).reshape(1, -1)  # 重塑為二維陣列
                input_scaled = scaler.transform(input_array)  # 使用標準化器進行標準化
                
                # 使用模型進行預測
                prediction = model.predict(input_scaled)[0]  # 取得預測結果
                prediction_probability = prediction[1]  # 取得心臟病風險的機率
                prediction_result = round(prediction_probability * 100, 2)  # 轉換為百分比
                
                # 根據預測結果判斷風險等級
                if prediction_result < 10:
                    risk_level = "低風險"
                elif prediction_result < 20:
                    risk_level = "中等風險"
                else:
                    risk_level = "高風險"
                
                logging.info(f"預測結果：{prediction_result}%, 風險等級：{risk_level}")
            else:
                logging.error("模型或標準化器未加載，無法進行預測")
        except Exception as e:
            logging.error(f"預測過程中發生錯誤：{str(e)}")
            prediction_result = None
            risk_level = None
    
    # 渲染 HTML 模板並傳遞數據
    return render_template(
        'index.html',
        features=features,
        prediction=prediction_result,
        risk_level=risk_level
    )


if __name__ == '__main__':
    logging.info("啟動 Flask 應用程序")
    app.run(debug=True, host='0.0.0.0', port=5000)  # 啟動應用，設置為調試模式
