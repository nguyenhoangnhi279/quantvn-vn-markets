import os
from dotenv import load_dotenv
from quantvn.vn.data.utils import client
import pandas as pd
import numpy as np
from quantvn.vn.data import get_stock_hist
from quantvn.vn.metrics import Backtest_Derivates

# Thiết lập môi trường
load_dotenv()
api_key = os.getenv("QUANTVN_API_KEY")

# Khởi tạo kết nối với hệ thống QuantVN
client(apikey=api_key)

# Hàm chiến lược giao dịch
def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chiến lược giao dịch cụ thể:
    - Dùng MA để bám theo xu hướng chính.
    - Dùng mốc RSI = 50 để xác nhận sức mạnh dòng tiền tại thời điểm.
    - Bỏ qua giới hạn quá mua và quá bán để tối đa hóa lợi nhuận.
    """
    df = df.copy()
    
    # Đảm bảo cột ở định dạng số học
    df["Close"] = pd.to_numeric(df["Close"])

    # Xác định chu kỳ cho MA
    # Chu kỳ ngắn : Giá gần nhất với giá hiện tại để phản ứng nhanh với dòng tiền.
    fast_window = 10 
    # Chu kỳ dài: Giá trung bình dài hạn để xác định xu hướng chính của thị trường dòng tiền.
    slow_window = 50  
    
    df["MA_Fast"] = df["Close"].rolling(fast_window).mean()
    df["MA_Slow"] = df["Close"].rolling(slow_window).mean()

    # Đo lường sức mạnh dòng tiền với RSI
    # Tính mức thay đổi giá giữa các phiên
    delta = df["Close"].diff()
    
    # Tách riêng các phiên tăng và phiên giảm
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Tính trung bình trượt lũy thừa (EMA) của các phiên tăng và giảm.    
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    
    # Tính RSI
    rs = ema_up / ema_down
    df["RSI"] = 100 - (100 / (1 + rs))

    # Khởi tạo trạng thái ban đầu là 0 
    df["position"] = 0
    
    # Điều kiện mua là khi MA ngắn cắt lên MA dài (uptrend) và RSI > 50 (xác nhận dòng tiền đang vào mạnh)
    buy_condition = (df["MA_Fast"] > df["MA_Slow"]) & (df["RSI"] > 50)
    df.loc[buy_condition, "position"] = 1  
    
    # Điều kiện bán là khi MA ngắn cắt xuống MA dài (downtrend) và RSI < 50 (xác nhận dòng tiền đang ra mạnh)    # Khi MA ngắn cắt xuống MA dài (Xác nhận Downtrend) 
    sell_condition = (df["MA_Fast"] < df["MA_Slow"]) & (df["RSI"] < 50) 
    df.loc[sell_condition, "position"] = -1 

    # Xử lý các giá trị NaN ở đầu
    df["position"] = df["position"].fillna(0)

    return df

# Chạy chiến lược 
if __name__ == "__main__":
    # Lấy dữ liệu mã FPT trên khung thời gian 1 Giờ 
    df = get_stock_hist("FPT", resolution="1H")
    
    if df.empty:
        print("Không lấy được dữ liệu")
    else:        
        # Chạy dữ liệu qua hàm chiến lược
        df_result = gen_position(df)
        # In ra kết quả để kiểm tra
        print(df_result[['time', 'Close', 'MA_Fast', 'MA_Slow', 'RSI', 'position']])