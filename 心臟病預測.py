import tkinter as tk  # 導入 tkinter，建立 GUI 應用程式
from tkinter import messagebox, ttk  # 從 tkinter 導入消息框和主題控件
import pymysql  # 導入 pymysql，用於 MySQL 數據庫操作
import numpy as np  # 導入 NumPy，用於數值計算
import tensorflow as tf  # 導入 TensorFlow，用於機器學習模型
import joblib  # 導入 joblib，用於模型和數據的序列化
from datetime import datetime  # 導入日期時間模組
from typing import Dict, List, Tuple, Optional  # 導入類型提示
import logging  # 導入日誌模組

# 配置日誌，將日誌寫入 MySQL 數據庫
class MySQLHandler(logging.Handler):
    def __init__(self, db_config):
        super().__init__()  # 初始化基類
        self.db_config = db_config  # 儲存數據庫配置

    def emit(self, record):
        log_entry = self.format(record)  # 格式化日誌記錄
        try:
            with pymysql.connect(**self.db_config) as conn:  # 連接數據庫
                with conn.cursor() as cursor:  # 創建游標
                    # 將日誌消息插入到數據庫
                    cursor.execute("INSERT INTO heart_disease_predictions (log_message) VALUES (%s)", (log_entry,))
                conn.commit()  # 提交事務
        except Exception as e:
            print(f"Failed to log to database: {str(e)}")  # 日誌寫入失敗

class InputValidator:
    """驗證用戶輸入的類別"""
    RANGES = {
        'male': (0, 1, True),  # 性別，二元值
        'age': (20, 100, False),  # 年齡範圍
        'education': (0, 2, True),  # 教育程度，二元值
        'currentSmoker': (0, 1, True),  # 是否吸煙，二元值
        'cigsPerDay': (0, 100, False),  # 每日吸煙量範圍
        'BPMeds': (0, 1, True),  # 是否服用降壓藥，二元值
        'prevalentStroke': (0, 1, True),  # 是否有中風病史，二元值
        'prevalentHyp': (0, 1, True),  # 是否有高血壓，二元值
        'diabetes': (0, 1, True),  # 是否有糖尿病，二元值
        'totChol': (100, 500, False),  # 總膽固醇範圍
        'sysBP': (90, 200, False),  # 收縮壓範圍
        'diaBP': (60, 130, False),  # 舒張壓範圍
        'BMI': (15, 50, False),  # 體質指數範圍
        'heartRate': (40, 120, False),  # 心率範圍
        'glucose': (40, 400, False)  # 血糖範圍
    }

    @staticmethod
    def validate(value: str, field: str) -> Tuple[bool, Optional[str]]:
        """驗證輸入值是否在範圍內"""
        try:
            value = float(value)  # 將值轉換為浮點數
            min_val, max_val, is_binary = InputValidator.RANGES[field]  # 根據欄位獲取範圍
            
            if is_binary and value not in [0, 1]:  # 檢查二元值
                return False, f"{field} 必須為 0 或 1"
            
            if not (min_val <= value <= max_val):  # 檢查範圍
                return False, f"{field} 必須在 {min_val} 到 {max_val} 之間"
                
            return True, None  # 驗證通過
        except ValueError:
            return False, f"{field} 必須為數字"  # 轉換失敗
        except KeyError:
            return False, f"未知欄位: {field}"  # 欄位不存在

class RiskAnalyzer:
    """分析風險因素的類別"""
    RISK_THRESHOLDS = {
        'age': (45, "年齡風險較高"),
        'sysBP': (130, "收縮壓偏高"),
        'diaBP': (80, "舒張壓偏高"),
        'glucose': (100, "血糖偏高"),
        'BMI': (25, "體重過重"),
        'totChol': (200, "總膽固醇偏高"),
        'heartRate': (80, "心率偏快"),
        'cigsPerDay': (0, "有吸菸習慣")
    }

    @staticmethod
    def analyze(input_data: Dict[str, float]) -> List[str]:
        """分析輸入數據的風險因素"""
        risk_factors = []
        
        # 檢查每個風險因子的閾值
        for field, (threshold, message) in RiskAnalyzer.RISK_THRESHOLDS.items():
            if field in input_data:
                if field == 'cigsPerDay' and input_data[field] > threshold:
                    risk_factors.append(message)
                elif field != 'cigsPerDay' and input_data[field] >= threshold:
                    risk_factors.append(message)
        
        # 特殊風險因素檢查
        if input_data.get('currentSmoker') == 1:
            risk_factors.append("目前有吸菸")
        if input_data.get('diabetes') == 1:
            risk_factors.append("有糖尿病病史")
        if input_data.get('prevalentStroke') == 1:
            risk_factors.append("有中風病史")
        if input_data.get('prevalentHyp') == 1:
            risk_factors.append("有高血壓病史")
        if input_data.get('BPMeds') == 1:
            risk_factors.append("正在服用降血壓藥物")
            
        return risk_factors if risk_factors else ["目前無明顯風險因素"]
class HeartDiseasePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("進階心臟病風險預測系統")
        
        # 修改資料庫名稱格式
        self.db_config = {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': 'oneokrock12345',
            'database': 'heart rate monitor',  # 修改為底線格式
            'charset': 'utf8mb4'
        }
        self.db_manager = DatabaseManager()
        
        try:
            # 確保模型正確載入
            self.model = tf.keras.models.load_model('heart_disease_model.keras')
            self.scaler = joblib.load('scaler.pkl')
            logging.info("Model and scaler loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            messagebox.showerror("錯誤", f"模型載入失敗: {str(e)}")
            return

        self.setup_ui()
        self.setup_styles()


class DatabaseManager:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': 'oneokrock12345',
            'database': 'heart rate monitor',
            'charset': 'utf8mb4'
        }
        self.setup_database()

    def setup_database(self):
        """設置資料庫和表格"""
        try:
            # 首先創建資料庫連接（不指定資料庫）
            conn = pymysql.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                charset=self.db_config['charset']
            )
            
            with conn.cursor() as cursor:
                # 創建資料庫
                cursor.execute("CREATE DATABASE IF NOT EXISTS `heart_rate_monitor`")
                cursor.execute("USE `heart_rate_monitor`")
                
                # 刪除現有的表格（如果存在）
                cursor.execute("DROP TABLE IF EXISTS heart_disease_predictions")
                
                # 創建新的表格，確保包含所有必要的欄位
                cursor.execute("""
                CREATE TABLE heart_disease_predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    male FLOAT,
                    age FLOAT,
                    education FLOAT,
                    currentSmoker FLOAT,
                    cigsPerDay FLOAT,
                    BPMeds FLOAT,
                    prevalentStroke FLOAT,
                    prevalentHyp FLOAT,
                    diabetes FLOAT,
                    totChol FLOAT,
                    sysBP FLOAT,
                    diaBP FLOAT,
                    BMI FLOAT,
                    heartRate FLOAT,
                    glucose FLOAT,
                    prediction FLOAT,
                    risk_level TINYINT,
                    risk_factors TEXT,
                    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                conn.commit()
            logging.info("Database and tables setup completed successfully")
        except Exception as e:
            logging.error(f"Database setup failed: {str(e)}")
            raise

    def save_prediction(self, input_data, prediction, risk_factors):
        """保存預測結果和輸入數據到資料庫"""
        try:
            risk_level = 1 if prediction >= 0.5 else 0
            
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                sql = """
                INSERT INTO heart_disease_predictions (
                    male, age, education, currentSmoker, cigsPerDay,
                    BPMeds, prevalentStroke, prevalentHyp, diabetes,
                    totChol, sysBP, diaBP, BMI, heartRate, glucose,
                    prediction, risk_level, risk_factors
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
                """
                cursor.execute(sql, (
                    float(input_data['male']), 
                    float(input_data['age']), 
                    float(input_data['education']),
                    float(input_data['currentSmoker']), 
                    float(input_data['cigsPerDay']),
                    float(input_data['BPMeds']), 
                    float(input_data['prevalentStroke']),
                    float(input_data['prevalentHyp']), 
                    float(input_data['diabetes']),
                    float(input_data['totChol']), 
                    float(input_data['sysBP']),
                    float(input_data['diaBP']), 
                    float(input_data['BMI']),
                    float(input_data['heartRate']), 
                    float(input_data['glucose']),
                    float(prediction), 
                    risk_level, 
                    ','.join(risk_factors)
                ))
                conn.commit()
            logging.info(f"Successfully saved prediction with risk level {risk_level}")
            return True
        except Exception as e:
            logging.error(f"Failed to save prediction: {str(e)}")
            return False

    def get_prediction_history(self, limit: int = 10):
        """獲取預測歷史紀錄"""
        try:
            conn = pymysql.connect(**self.db_config)
            with conn.cursor() as cursor:
                sql = """
                SELECT 
                    id, prediction_time, prediction, risk_factors,
                    male, age, education, currentSmoker, cigsPerDay,
                    BPMeds, prevalentStroke, prevalentHyp, diabetes,
                    totChol, sysBP, diaBP, BMI, heartRate, glucose,
                    risk_level
                FROM heart_disease_predictions
                ORDER BY prediction_time DESC
                LIMIT %s
                """
                cursor.execute(sql, (limit,))
                return cursor.fetchall()
        except Exception as e:
            logging.error(f"Failed to retrieve prediction history: {str(e)}")
            return []



class HeartDiseasePredictor:
    """心臟病預測主應用程式類別"""
    def __init__(self, root):
        self.root = root
        self.root.title("進階心臟病風險預測系統")  # 設定窗口標題
        self.db_manager = DatabaseManager()  # 初始化資料庫管理器
        
        try:
            # 載入機器學習模型和標準化器
            self.model = tf.keras.models.load_model('heart_disease_model.keras')
            self.scaler = joblib.load('scaler.pkl')
            logging.info("Model and scaler loaded successfully")  # 載入成功日誌
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")  # 載入失敗日誌
            messagebox.showerror("錯誤", f"模型載入失敗: {str(e)}")  # 顯示錯誤消息
            return

        self.setup_ui()  # 設置用戶界面
        self.setup_styles()  # 設置樣式

    def setup_styles(self):
        """配置自定義樣式"""
        style = ttk.Style()
        style.configure('Risk.TLabel', font=('Arial', 12, 'bold'))  # 風險標籤樣式
        style.configure('Header.TLabel', font=('Arial', 11))  # 標題樣式
        style.configure('Value.TLabel', font=('Arial', 10))  # 值樣式

    def setup_ui(self):
        """設置主要用戶界面"""
        self.notebook = ttk.Notebook(self.root)  # 創建選項卡控件
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)

        self.input_frame = ttk.Frame(self.notebook)  # 數據輸入框架
        self.history_frame = ttk.Frame(self.notebook)  # 預測歷史框架
        
        self.notebook.add(self.input_frame, text='數據輸入')  # 添加數據輸入頁面
        self.notebook.add(self.history_frame, text='預測歷史')  # 添加預測歷史頁面

        self.create_input_page()  # 創建數據輸入頁面
        self.create_history_page()  # 創建預測歷史頁面

    def create_input_page(self):
        """創建輸入表單頁面"""
        canvas = tk.Canvas(self.input_frame)  # 創建畫布
        scrollbar = ttk.Scrollbar(self.input_frame, orient="vertical", command=canvas.yview)  # 創建垂直滾動條
        scrollable_frame = ttk.Frame(canvas)  # 可滾動框架

        # 當框架尺寸改變時，更新畫布的可滾動範圍
        scrollable_frame.bind("<Configure>", 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")  # 將可滾動框架放入畫布中
        canvas.configure(yscrollcommand=scrollbar.set)  # 設置滾動條的命令

        # 設置網格配置
        self.input_frame.grid_rowconfigure(0, weight=1)
        self.input_frame.grid_columnconfigure(0, weight=1)
        canvas.grid(row=0, column=0, sticky="nsew")  # 放置畫布
        scrollbar.grid(row=0, column=1, sticky="ns")  # 放置滾動條

        self.input_fields = {}  # 儲存輸入欄位的字典
        self.create_input_fields(scrollable_frame)  # 創建輸入欄位
        self.create_results_area(scrollable_frame)  # 創建結果顯示區域

    def create_input_fields(self, parent):
        """創建所有輸入欄位及其標籤和驗證"""
        fields = [
            ("性別", "male", "1 表示男性, 0 表示女性", True),  # 性別欄位
            ("年齡", "age", "年齡", False),  # 年齡欄位
            ("教育水平", "education", "教育程度", True, True),  # 教育程度欄位
            ("是否吸煙", "currentSmoker", "1 表示吸煙, 0 表示不吸煙", True),  # 吸煙欄位
            ("每日吸煙量", "cigsPerDay", "每日吸煙數量", False),  # 每日吸煙量欄位
            ("是否服用降壓藥", "BPMeds", "1 表示服用, 0 表示未服用", True),  # 降壓藥欄位
            ("是否有中風病史", "prevalentStroke", "1 表示有, 0 表示沒有", True),  # 中風病史欄位
            ("是否有高血壓", "prevalentHyp", "1 表示有, 0 表示沒有", True),  # 高血壓欄位
            ("是否有糖尿病", "diabetes", "1 表示有, 0 表示沒有", True),  # 糖尿病欄位
            ("總膽固醇", "totChol", "總膽固醇 (mg/dL)", False),  # 總膽固醇欄位
            ("收縮壓", "sysBP", "收縮壓 (mmHg)", False),  # 收縮壓欄位
            ("舒張壓", "diaBP", "舒張壓 (mmHg)", False),  # 舒張壓欄位
            ("BMI", "BMI", "體質指數", False),  # BMI欄位
            ("心率", "heartRate", "靜息心率 (次/分鐘)", False),  # 心率欄位
            ("血糖", "glucose", "空腹血糖 (mg/dL)", False)  # 血糖欄位
        ]

        # 創建每個欄位的標籤和輸入控件
        for i, field_info in enumerate(fields):
            frame = ttk.Frame(parent)  # 每個欄位的框架
            frame.pack(fill="x", padx=10, pady=5)  # 填滿整個行並設置邊距

            label, key, range_text = field_info[:3]  # 獲取欄位標籤、鍵和範圍文本
            is_binary = field_info[3] if len(field_info) > 3 else False  # 是否為二元值
            is_education = len(field_info) > 4 and field_info[4]  # 是否為教育程度欄位

            # 創建標籤
            ttk.Label(frame, text=f"{label} ({range_text})", style='Header.TLabel').pack(side="left", padx=5)
            
            # 根據欄位類型創建不同的輸入控件
            if is_education:
                var = tk.StringVar(value="0")
                widget = ttk.Combobox(frame, textvariable=var, values=['0', '1', '2'], width=2)  # 教育程度下拉框
                widget.set("0")
                ttk.Label(frame, text="(0=未受教育, 1=部分受教育, 2=大學及以上)", style='Value.TLabel').pack(side="right", padx=5)
            elif is_binary:
                var = tk.StringVar(value="0")
                widget = ttk.Combobox(frame, textvariable=var, values=['0', '1'], width=5)  # 二元值下拉框
                widget.set("0")
            else:
                var = tk.StringVar()
                widget = ttk.Entry(frame, textvariable=var, width=10)  # 普通文本框
            
            widget.pack(side="left")  # 放置控件
            self.input_fields[key] = var  # 儲存該欄位的變量

    def create_results_area(self, parent):
        """創建結果顯示區域"""
        result_frame = ttk.Frame(parent)  # 結果框架
        result_frame.pack(fill="x", padx=10, pady=20)

        self.result_label = ttk.Label(result_frame, text="", style='Risk.TLabel', wraplength=400)  # 結果標籤
        self.result_label.pack(pady=10)

        ttk.Button(result_frame, text="預測風險", command=self.predict).pack(pady=10)  # 預測按鈕

    def create_history_page(self):
        """創建預測歷史頁面"""
        columns = ('編號', '日期時間', '風險程度', '主要風險因素')  # 設置列名
        self.tree = ttk.Treeview(self.history_frame, columns=columns, show='headings')  # 創建樹狀視圖

        # 設置每一列的寬度
        column_widths = {'編號': 50, '日期時間': 150, '風險程度': 100, '主要風險因素': 300}
        for col in columns:
            self.tree.heading(col, text=col)  # 設置列標題
            self.tree.column(col, width=column_widths.get(col, 150))  # 設置列寬

        scrollbar = ttk.Scrollbar(self.history_frame, orient="vertical", command=self.tree.yview)  # 創建垂直滾動條
        self.tree.configure(yscrollcommand=scrollbar.set)  # 設置滾動條的命令

        self.tree.grid(row=0, column=0, sticky="nsew")  # 放置樹狀視圖
        scrollbar.grid(row=0, column=1, sticky="ns")  # 放置滾動條

        self.history_frame.grid_rowconfigure(0, weight=1)  # 設置行的權重
        self.history_frame.grid_columnconfigure(0, weight=1)  # 設置列的權重

    def predict(self):
        """執行預測並更新用戶界面"""
        try:
            # 收集輸入數據
            input_data = {}
            for key, var in self.input_fields.items():
                value = var.get()
                is_valid, message = InputValidator.validate(value, key)
                if not is_valid:
                    messagebox.showerror("輸入錯誤", message)
                    return
                input_data[key] = float(value)

            # 確保特徵順序正確
            features = [
                'male', 'age', 'education', 'currentSmoker', 'cigsPerDay',
                'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',
                'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
            ]
            
            # 創建輸入數組
            input_values = [input_data[feature] for feature in features]
            input_array = np.array(input_values).reshape(1, -1)
            
            # 數據預處理
            input_scaled = self.scaler.transform(input_array)
            
            # 進行預測
            prediction = float(self.model.predict(input_scaled)[0][0])

            # 確保預測值在 0-1 之間
            prediction = max(0.0, min(1.0, prediction))

            # 計算風險百分比
            risk_percentage = prediction * 100  # 此處直接轉換為百分比

            # 判斷風險等級
            risk_level, color = self.determine_risk_level(risk_percentage)

            # 分析風險因素
            risk_factors = RiskAnalyzer.analyze(input_data)
            
            # 更新界面
            self.update_results(risk_percentage, risk_factors)

            # 保存到資料庫
            if self.db_manager.save_prediction(input_data, prediction, risk_factors):
                self.update_history()
                logging.info(f"Successfully saved prediction. Risk: {risk_percentage}%")
            else:
                messagebox.showwarning("警告", "預測結果已顯示但無法保存到數據庫")

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            messagebox.showerror("錯誤", f"預測過程出錯: {str(e)}")




    def update_results(self, risk_percentage: float, risk_factors: List[str]):
        """更新結果顯示區域"""
        
        # Determine risk level and color
        risk_level, color = self.determine_risk_level(risk_percentage)

        # Format risk factors for display
        risk_factors_text = self.format_risk_factors(risk_factors)
        
        # Create result text
        result_text = self.create_result_text(risk_percentage, risk_level, risk_factors_text)
        
        # Update the label
        self.result_label.config(text=result_text)

    def determine_risk_level(self, risk_percentage: float):
        """判斷風險等級"""
        if risk_percentage < 10:
            return "低度風險", "green"
        elif risk_percentage < 20:
            return "中度風險", "orange"
        else:
            return "高度風險", "red"

    def format_risk_factors(self, risk_factors: List[str]):
        """格式化風險因素列表"""
        return "\n".join([f"• {factor}" for factor in risk_factors])

    def create_result_text(self, risk_percentage: float, risk_level: str, risk_factors_text: str):
        """創建結果文本"""
        return f"""十年內心臟病風險預測結果：

    風險機率：{risk_percentage:.1f}%
    風險等級：{risk_level}

    主要風險因素：
    {risk_factors_text}

    建議：
    • 定期進行健康檢查
    • 保持良好的生活習慣
    • 若風險較高，建議諮詢專業醫師意見"""


    def update_history(self):
        """更新預測歷史記錄"""
        # 清空現有記錄
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 獲取歷史記錄
        history = self.db_manager.get_prediction_history()
        
        # 更新顯示
        for record in history:
            id, timestamp, prediction, risk_factors, *other_fields = record
            
            # 計算風險百分比和等級
            risk_percentage = prediction * 100
            if risk_percentage < 10:
                risk_level = "低度風險"
            elif risk_percentage < 20:
                risk_level = "中度風險"
            else:
                risk_level = "高度風險"
                
            # 格式化顯示數據
            formatted_risk = f"{risk_percentage:.1f}% ({risk_level})"
            
            self.tree.insert('', 'end', values=(
                id,
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                formatted_risk,
                risk_factors
            ))
            
if __name__ == "__main__":
    root = tk.Tk()  # 創建主窗口
    root.geometry("800x600")  # 設定窗口大小
    app = HeartDiseasePredictor(root)  # 初始化應用程序
    root.mainloop()  # 開始主循環

