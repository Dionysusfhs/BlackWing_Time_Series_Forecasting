import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# 计算RSI（相对强弱指数）
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 计算动量指标
def calculate_momentum(data, period=10):
    momentum = data['feature'] - data['feature'].shift(period)
    return momentum

# 计算布林带
def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data['feature'].rolling(window=window).mean()
    rolling_std = data['feature'].rolling(window=window).std()
    data['bollinger_middle'] = rolling_mean
    data['bollinger_upper'] = rolling_mean + (rolling_std * num_std)
    data['bollinger_lower'] = rolling_mean - (rolling_std * num_std)

# 简单的移动平均线计算
def calculate_moving_average(data, window=20):
    return data['feature'].rolling(window=window).mean()

def model(input_data: pd.DataFrame, *args) -> pd.Series:
    # 确保输入数据包含 'date' 和 'feature' 列
    if 'date' not in input_data.columns or 'feature' not in input_data.columns or 'target' not in input_data.columns:
        raise ValueError("Input data must contain 'date', 'feature', and 'target' columns")

    # 数据准备
    sample_data = input_data.copy()
    sample_data['date'] = pd.to_datetime(sample_data['date'])
    # sample_data.rename(columns={'index': 'feature'}, inplace=True)

    # 计算并返回20日均线
    #feature = sample_data["feature"]
    #date = sample_data["date"]
    #predict = calculate_moving_average(sample_data, 20)
    #result = pd.Series(index=date, data=predict, name='20_Day_MA')
    
    # 绘制feature列
    # plt.figure(figsize=(14, 6))
    # plt.plot(sample_data['date'], sample_data['feature'], label='Feature', color='blue')
    # plt.title('Feature Over Time')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.show()

    # 计算均线数据作为特征
    sample_data['feature_ma5'] = sample_data['feature'].rolling(window=5).mean()
    sample_data['feature_ma10'] = sample_data['feature'].rolling(window=10).mean()
    sample_data['feature_ma30'] = sample_data['feature'].rolling(window=30).mean()
    sample_data['feature_ma60'] = sample_data['feature'].rolling(window=60).mean()

    # 绘制均线
    # plt.figure(figsize=(14, 8))
    # plt.plot(sample_data['date'], sample_data['feature'], label='Feature', color='blue')
    # plt.plot(sample_data['date'], sample_data['feature_ma5'], label='5-Day MA', color='orange', linestyle='--')
    # plt.plot(sample_data['date'], sample_data['feature_ma10'], label='10-Day MA', color='green', linestyle='--')
    # plt.plot(sample_data['date'], sample_data['feature_ma30'], label='30-Day MA', color='red', linestyle='--')
    # plt.plot(sample_data['date'], sample_data['feature_ma60'], label='60-Day MA', color='purple', linestyle='--')
    # plt.title('Index with Moving Averages')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.show()

    # 计算并绘制RSI
    sample_data['rsi_14'] = calculate_rsi(sample_data['feature'])
    # plt.figure(figsize=(14, 6))
    # plt.plot(sample_data['date'], sample_data['rsi_14'], label='RSI 14', color='green')
    # plt.axhline(70, color='red', linestyle='--')
    # plt.axhline(30, color='blue', linestyle='--')
    # plt.title('Relative Strength Index (RSI)')
    # plt.xlabel('Date')
    # plt.ylabel('RSI')
    # plt.legend()
    # plt.show()

    # 计算并绘制动量指标
    sample_data['momentum_10'] = calculate_momentum(sample_data, period=10)
    # plt.figure(figsize=(14, 8))
    # plt.plot(sample_data['date'], sample_data['feature'], label='Feature', color='blue')
    # plt.plot(sample_data['date'], sample_data['momentum_10'], label='Momentum 10', color='purple')
    # plt.title('Index and Momentum Indicator')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.show()

    # 计算并绘制布林带
    calculate_bollinger_bands(sample_data, window=20, num_std=2)
    # plt.figure(figsize=(14, 8))
    # plt.plot(sample_data['date'], sample_data['feature'], label='Feature', color='blue')
    # plt.plot(sample_data['date'], sample_data['bollinger_middle'], label='Middle Band (SMA 20)', color='orange', linestyle='--')
    # plt.plot(sample_data['date'], sample_data['bollinger_upper'], label='Upper Band (SMA 20 + 2 Std Dev)', color='green', linestyle='--')
    # plt.plot(sample_data['date'], sample_data['bollinger_lower'], label='Lower Band (SMA 20 - 2 Std Dev)', color='red', linestyle='--')
    # plt.fill_between(sample_data['date'], sample_data['bollinger_upper'], sample_data['bollinger_lower'], color='grey', alpha=0.1)
    # plt.title('Bollinger Bands')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.show()

    # 获取并合并指数数据
    index_codes = {
        'ss': '000001.SS',
        'sz': '399001.SZ',
        'ndsk': '^IXIC'
    }

    index_data = {}
    for name, code in index_codes.items():
        index_data[name] = yf.download(code, start="2019-01-02", end="2022-01-01")['Close']
    
    index_df = pd.DataFrame(index_data)
    index_df.reset_index(inplace=True)
    index_df.rename(columns={'Date': 'date'}, inplace=True)
    sample_data = pd.merge(sample_data, index_df, on='date', how='left')

    # 绘制上证指数
    # plt.figure(figsize=(10, 5))
    # plt.plot(index_df['date'], index_df['ss'], label='000001.SS', color='blue')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.title('Feature of 000001.SS')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # 绘制深证成指
    # plt.figure(figsize=(10, 5))
    # plt.plot(index_df['date'], index_df['sz'], label='399001.SZ', color='red')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.title('Feature of 399001.SZ')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # 绘制纳达斯科
    # plt.figure(figsize=(10, 5))
    # plt.plot(index_df['date'], index_df['ndsk'], label='ndsk', color='purple')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.title('Feature of ndsk')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # 滞后特征
    for lag in range(1, 8):  # 过去一周的滞后特征
        sample_data[f'index_lag_{lag}'] = sample_data['feature'].shift(lag)

    # 21日滚动波动率
    sample_data['volatility'] = sample_data['feature'].pct_change().rolling(window=21).std()

    # 日收益率
    sample_data['daily_return'] = sample_data['feature'].pct_change()
    sample_data['weekly_return'] = sample_data['feature'].pct_change(5)
    sample_data['monthly_return'] = sample_data['feature'].pct_change(21)

    # Feature离散化
    sample_data['feature_bin'] = pd.qcut(sample_data['feature'], 10, labels=False)

    # 确保没有缺失值
    sample_data.dropna(inplace=True)

    # 分割特征和目标变量
    X = sample_data.drop(columns=['date', 'feature'])
    y = sample_data['feature']

    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 定义随机森林回归模型
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)

    # 定义 XGBoost 回归模型
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', 
                                 colsample_bytree=0.3, 
                                 learning_rate=0.1,
                                 max_depth=5, 
                                 alpha=10, 
                                 n_estimators=100)

    # 定义投票回归器
    voting_model = VotingRegressor(estimators=[
        ('random_forest', rf_model),
        ('xgboost', xgb_model)
]    )

    # 训练投票回归器
    voting_model.fit(X_train, y_train)

    # 进行预测
    y_pred_train = voting_model.predict(X_train)
    y_pred_test = voting_model.predict(X_test)

    # 评估模型
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"Training RMSE: {train_rmse}")
    print(f"Testing RMSE: {test_rmse}")

    # 绘制预测值与实际值的对比图
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred_test, alpha=0.3)
    plt.xlabel('Actual Status')
    plt.ylabel('Predicted Status')
    plt.title('Testing Set: Actual vs Predicted Status')
    plt.show() 

    # 返回每个日期对应的预测值
    results_df = pd.DataFrame({
        'Date': X_test.index, 
        'True_Values': y_test.values,
        'Predictions': y_pred_test
})
    return results_df

## 以下代码作为各位同学测试使用，实际提交代码时可以不用提交下面代码
if __name__=="__main__":
    sample_data = pd.read_csv("sample_data.csv")
    # other_data = pd.read_csv("这里是其他外部数据")
    other_data = None
    # predict = model(sample_data,other_data)
    predict = model(sample_data)
    print(predict)

