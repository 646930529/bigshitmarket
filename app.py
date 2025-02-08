from flask import Flask, render_template
import akshare as ak
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
from contextlib import contextmanager
from typing import List, Tuple, Optional



DATABASE = 'stock_cache.db'
conn = sqlite3.connect(DATABASE)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS index_data (
    symbol TEXT,
    today DATE,
    date DATE,
    close REAL
)
''')
conn.commit()
dataCount = 1000
isminmax = False
x_min = None
x_max = None



app = Flask(__name__)

def get_cached_data(symbol: str, today: str) -> pd.DataFrame:
    cursor.execute("SELECT * FROM index_data WHERE symbol = ? AND today = ?", (symbol, today))
    if cursor.fetchone():
        print("读取：", symbol, today)
        return pd.read_sql(
            "SELECT * FROM index_data WHERE symbol = ? AND today = ? ORDER BY date ASC",
            conn, params=(symbol, today)
        )
    return None

def save_to_db(df: pd.DataFrame, symbol: str, today: str):
    df.to_sql('index_data', conn, if_exists='append', index=False)

def get_index(symbol: str) -> pd.DataFrame:
    today = datetime.now().strftime('%Y-%m-%d')
    cached_data = get_cached_data(symbol, today)
    if cached_data is not None:
        return cached_data.tail(dataCount)
    

    print("写入：", symbol, today)
    df = ak.stock_zh_index_daily(symbol)
    df['date'] = pd.to_datetime(df['date'])
    df['symbol'] = symbol
    df['today'] = today
    df = df[['symbol', 'today', 'date', 'close']]
    save_to_db(df, symbol, today)
    return df.tail(dataCount)


def get_index_hk(symbol: str, start_date: str = "20200101") -> pd.DataFrame:
    today = datetime.now().strftime('%Y-%m-%d')
    cached_data = get_cached_data(symbol, today)
    if cached_data is not None:
        return cached_data.tail(dataCount)
    

    print("写入：", symbol, today)
    df = ak.stock_hk_hist(symbol=symbol, period="daily", start_date=start_date, adjust="qfq")
    df = df[['日期', '收盘']].rename(columns={'日期': 'date', '收盘': 'close'})
    df['date'] = pd.to_datetime(df['date'])
    df['symbol'] = symbol
    df['today'] = today
    df = df[['symbol', 'today', 'date', 'close']]
    save_to_db(df, symbol, today)
    return df.tail(dataCount)


def get_index_fund(symbol: str) -> pd.DataFrame:
    today = datetime.now().strftime('%Y-%m-%d')
    cached_data = get_cached_data(symbol, today)
    if cached_data is not None:
        return cached_data.tail(dataCount)
    

    print("写入：", symbol, today)
    df = ak.fund_etf_hist_sina(symbol)
    df['date'] = pd.to_datetime(df['date'])
    df['symbol'] = symbol
    df['today'] = today
    df = df[['symbol', 'today', 'date', 'close']]
    save_to_db(df, symbol, today)
    return df.tail(dataCount)


def create_figure(symbol: str, name: str, calltype: int) -> go.Figure:
    function_mapping = {
        1: get_index,
        2: get_index_hk,
        3: get_index_fund
    }
    data = function_mapping[calltype](symbol)
    fig = go.Figure()
    add_basic_plot(fig, data, name)
    add_rsi_signals(fig, data)
    add_ma_plot(fig, data, 250)
    return fig

def create_plots() -> List[str]:
    indices = [
        ('sh000001', 'Shanghai', 1),
        ('sz399001', 'Shenzhen', 1),
        ('sh000688', 'Ke50', 1),
        ('sz399006', 'Chuangye', 1),
        ('bj899050', 'Beijing', 1),
        ('sh000300', 'zhongzheng300', 1),
        ('sh000905', 'zhongzheng500', 1),
        ('sh000852', 'zhongzheng1000', 1),
        ('sh000015', 'Hongli', 1),

        ('sz159691', 'hkhongli', 3),
        
        ('02800', 'hsi', 2),
        ('03033', 'hktech', 2),

    ]
    return [create_figure(symbol, name, is_hk).to_html(full_html=False) 
            for symbol, name, is_hk in indices]

def add_basic_plot(fig, data, name):
    global isminmax
    global x_min
    global x_max


    # 确保date列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(data['date']):
        data['date'] = pd.to_datetime(data['date'])
    
    # 计算x轴范围
    if not isminmax:
        x_min = data['date'].min() + pd.Timedelta(days=300)  # 左移30天
        x_max = data['date'].max() + pd.Timedelta(days=300) 
        isminmax = True


    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['close'],
        mode='lines',
        name=name,
        line=dict(color='#1f77b4')
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=50, r=20, t=40, b=50),
        plot_bgcolor='#2d2d2d',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        # dragmode='pan',  # 设置为可拖动模式
        xaxis=dict(
            gridcolor='#444',
            linecolor='#666',
            zerolinecolor='#444',
            automargin=True,
            showline=True,
            mirror=True,
            range=[x_min, x_max]
        ),
        yaxis=dict(
            gridcolor='#444',
            linecolor='#666',
            zerolinecolor='#444',
            automargin=True,
            showline=True,
            mirror=True
        )
    )

def add_rsi_signals(fig, data, period=14, overbought=70, oversold=30):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    buy_signals = data[data['rsi'] < oversold]
    sell_signals = data[data['rsi'] > overbought]
    fig.add_trace(go.Scatter(
        x=buy_signals['date'],
        y=buy_signals['close'],
        mode='markers',
        marker=dict(symbol='triangle-up', color='#00ff00', size=10),
        name='RSI Buy'
    ))
    fig.add_trace(go.Scatter(
        x=sell_signals['date'],
        y=sell_signals['close'],
        mode='markers',
        marker=dict(symbol='triangle-down', color='#ff0000', size=10),
        name='RSI Sell'
    ))

def add_ma_plot(fig, data, period=250):
    data[f'ma_{period}'] = data['close'].rolling(window=period).mean()
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data[f'ma_{period}'],
        mode='lines',
        line=dict(color='#ff7f0e', width=2),
        name=f'ma_{period}'
    ))

@app.route('/')
def index():
    return render_template('index.html', plot_htmls=plot_htmls)

if __name__ == '__main__':
    plot_htmls = create_plots()
    app.run(debug=True) 