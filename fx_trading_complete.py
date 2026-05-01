"""
USD/JPY FX Trading Signal Generator - 完全統合版
ターゲット価格計算 + グラフ視覚化 + メール通知
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import sys
import smtplib

import warnings
warnings.filterwarnings('ignore')

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os



# グラフ用
try:
    import matplotlib
    matplotlib.use('Agg')  # GUI不要モード
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.dates import DateFormatter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] matplotlib がインストールされていません")
    print("  グラフ機能は使用できません")
    print("  インストール: pip install matplotlib")

# ========== Windows cmd 対応ロギング設定 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fx_trading_complete.log', encoding='utf-8'),
        logging.StreamHandler(stream=sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """取引シグナル（ボラティリティ情報付き）"""
    timestamp: datetime
    pair: str
    signal_type: str
    strength: float
    price: float
    timeframe: str
    indicators: Dict
    reasoning: List[str]
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None


class RealDataFetcher:
    """実際のレートをYahoo Financeから取得"""
    
    @staticmethod
    def get_real_data(symbol: str = "USDJPY=X", period: str = "100d", 
                      interval: str = "1h") -> Optional[pd.DataFrame]:
        """Yahoo Financeからデータ取得"""
        try:
            import yfinance as yf
            
            logger.info(f"Yahoo Finance から {symbol} のデータを取得中...")
            
            data = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                timeout=30
            )
            
            if data is None or len(data) == 0:
                logger.warning(f"{symbol} のデータ取得に失敗しました")
                return None
            
            # MultiIndex カラムの処理
            df = data.copy()
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            df.columns = [str(col).lower() for col in df.columns]
            
            # datetime カラムを確認
            datetime_col = None
            for col in df.columns:
                if 'date' in col or 'time' in col:
                    datetime_col = col
                    break
            
            if datetime_col is None:
                logger.error("日時カラムが見つかりません")
                return None
            
            if datetime_col != 'datetime':
                df.rename(columns={datetime_col: 'datetime'}, inplace=True)
            
            # 必要なカラムを確認
            required_cols = ['datetime', 'open', 'high', 'low', 'close']
            available_cols = [col for col in df.columns if col in required_cols]
            
            if len(available_cols) < 4:
                logger.error("必要なカラムが不足しています")
                return None
            
            df = df[required_cols].copy()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            if len(df) < 20:
                logger.warning(f"データが少なすぎます: {len(df)}本")
                return None
            
            latest_price = df['close'].iloc[-1]
            
            logger.info(f"[OK] 取得成功: {len(df)}本のキャンドル")
            logger.info(f"  最新価格: {latest_price:.4f}")
            logger.info(f"  価格範囲: {df['close'].min():.4f} - {df['close'].max():.4f}")
            
            return df
        
        except ImportError:
            logger.error("yfinance がインストールされていません")
            return None
        except Exception as e:
            logger.error(f"データ取得エラー: {e}")
            return None


class VolatilityAnalysis:
    """ボラティリティ分析"""
    
    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR計算"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_historical_volatility(data: pd.Series, period: int = 20) -> pd.Series:
        """Historical Volatility計算"""
        log_returns = np.log(data / data.shift(1))
        hv = log_returns.rolling(window=period).std() * np.sqrt(252)
        return hv
    
    @staticmethod
    def calculate_volatility_ratio(data: pd.Series, short_period: int = 10, 
                                  long_period: int = 20) -> pd.Series:
        """Volatility Ratio計算"""
        short_vol = data.rolling(window=short_period).std()
        long_vol = data.rolling(window=long_period).std()
        volatility_ratio = short_vol / long_vol
        return volatility_ratio


class EnhancedTechnicalAnalysis:
    """拡張テクニカル分析"""
    
    def __init__(self):
        self.va = VolatilityAnalysis()
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_stochastic(data: pd.DataFrame, period: int = 14, 
                            k_period: int = 3, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        low = data['low'].rolling(window=period).min()
        high = data['high'].rolling(window=period).max()
        
        k_percent = 100 * (data['close'] - low) / (high - low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent


class TargetPriceCalculator:
    """ターゲット価格計算"""
    
    @staticmethod
    def calculate_targets(current_price: float, atr: float, 
                         signal_type: str) -> Dict[str, float]:
        """
        ターゲット価格を計算
        
        Args:
            current_price: 現在価格
            atr: ATR値
            signal_type: BUY or SELL
        
        Returns:
            計算結果の辞書
        """
        
        if signal_type == "BUY":
            stop_loss = current_price - (atr * 2)
            take_profit_1 = current_price + (atr * 1.5)
            take_profit_2 = current_price + (atr * 3.0)
            entry_price_low = current_price - 0.05
            entry_price_high = current_price
            
        else:  # SELL
            stop_loss = current_price + (atr * 2)
            take_profit_1 = current_price - (atr * 1.5)
            take_profit_2 = current_price - (atr * 3.0)
            entry_price_low = current_price
            entry_price_high = current_price + 0.05
        
        return {
            'entry_price_low': entry_price_low,
            'entry_price_high': entry_price_high,
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'risk': abs(stop_loss - current_price),
            'reward_1': abs(take_profit_1 - current_price),
            'reward_2': abs(take_profit_2 - current_price),
        }


class GraphVisualizer:
    """グラフ視覚化（改善版）"""
    
    @staticmethod
    def create_chart(df: pd.DataFrame, signal: Signal, symbol: str = "USDJPY") -> Optional[str]:
        """
        チャートを作成して画像ファイルに保存（見やすい改善版）
        
        Returns:
            保存されたファイルパス
        """
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("[WARNING] matplotlib が利用できません。グラフは作成されません")
            return None
        
        try:
            # 最後の50本のキャンドルだけを使用
            df_plot = df.tail(50).copy()
            
            # 指標を計算
            ta = EnhancedTechnicalAnalysis()
            close = df['close']
            
            ma20 = ta.calculate_sma(close, 20)
            ma50 = ta.calculate_sma(close, 50)
            rsi = ta.calculate_rsi(close, 14)
            
            # フィギュアを作成（大きめに）
            fig = plt.figure(figsize=(16, 10))
            fig.patch.set_facecolor('white')
            
            # ========== サブプロット1: ローソク足と移動平均線 ==========
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.set_facecolor('#f8f9fa')
            
            # ローソク足をプロット
            x_indices = range(len(df_plot))
            for idx, (i, row) in enumerate(df_plot.iterrows()):
                if row['close'] >= row['open']:
                    color = 'green'
                    body_height = row['close'] - row['open']
                else:
                    color = 'red'
                    body_height = row['open'] - row['close']
                
                # ヒゲ（高値-安値）
                ax1.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=1.5)
                # 実体（始値-終値）
                rect_y = min(row['open'], row['close'])
                rect = plt.Rectangle((idx-0.3, rect_y), 0.6, abs(body_height), 
                                     facecolor=color, edgecolor=color, linewidth=1)
                ax1.add_patch(rect)
            
            # 移動平均線
            ma20_plot = ma20.tail(50).values
            ma50_plot = ma50.tail(50).values
            
            ax1.plot(x_indices, ma20_plot, label='MA20(青)', color='blue', linewidth=2, alpha=0.8)
            ax1.plot(x_indices, ma50_plot, label='MA50(オレンジ)', color='orange', linewidth=2, alpha=0.8)
            
            # ========== ターゲット価格を表示 ==========
            current_price = df['close'].iloc[-1]
            
            # 現在価格（黒の太い点線）
            ax1.axhline(y=current_price, color='black', linestyle='--', linewidth=2.5, 
                       label=f'現在価格: {current_price:.2f}円', zorder=5)
            
            # ストップロス（黄色の太い線）
            if signal.stop_loss:
                ax1.axhline(y=signal.stop_loss, color='gold', linestyle='-', linewidth=3.5, 
                           label=f'ストップロス: {signal.stop_loss:.2f}円', zorder=4)
                # ラベルを追加
                ax1.text(len(df_plot)-1, signal.stop_loss, f' SL:{signal.stop_loss:.2f}', 
                        va='center', fontsize=10, color='darkgoldenrod', weight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # 利確目標1（緑の太い線）
            if signal.take_profit_1:
                ax1.axhline(y=signal.take_profit_1, color='lime', linestyle='-', linewidth=3.5, 
                           label=f'利確1: {signal.take_profit_1:.2f}円', zorder=4)
                ax1.text(len(df_plot)-1, signal.take_profit_1, f' TP1:{signal.take_profit_1:.2f}', 
                        va='center', fontsize=10, color='darkgreen', weight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            
            # 利確目標2（緑の点線）
            if signal.take_profit_2:
                ax1.axhline(y=signal.take_profit_2, color='green', linestyle='--', linewidth=2.5, 
                           label=f'利確2: {signal.take_profit_2:.2f}円', zorder=4)
                ax1.text(len(df_plot)-1, signal.take_profit_2, f' TP2:{signal.take_profit_2:.2f}', 
                        va='center', fontsize=9, color='darkgreen',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
            # ========== 軸とラベルを設定 ==========
            ax1.set_ylabel('価格 (円)', fontsize=12, weight='bold')
            ax1.set_title(f'{symbol} 1時間足チャート - {datetime.now().strftime("%Y年%m月%d日 %H:%M")}', 
                         fontsize=14, weight='bold', pad=20)
            
            # 時間軸を日本語で表示
            time_labels = []
            for i, (idx, row) in enumerate(df_plot.iterrows()):
                if i % 5 == 0:  # 5本ごとに表示
                    time_str = pd.to_datetime(row['datetime']).strftime('%H:%M')
                    time_labels.append(time_str)
                else:
                    time_labels.append('')
            
            ax1.set_xticks(x_indices)
            ax1.set_xticklabels(time_labels, rotation=45, ha='right')
            
            # グリッドと凡例
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
            ax1.set_ylim([df_plot['low'].min() - 0.5, df_plot['high'].max() + 0.5])
            
            # ========== サブプロット2: RSI ==========
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.set_facecolor('#f8f9fa')
            
            rsi_plot = rsi.tail(50).values
            
            # RSIラインを太くプロット
            ax2.plot(x_indices, rsi_plot, label='RSI(14)', color='purple', linewidth=2.5)
            
            # 過売・過買ラインを太く表示
            ax2.axhline(y=30, color='green', linestyle='-', linewidth=2.5, alpha=0.7, 
                       label='過売(30)')
            ax2.axhline(y=70, color='red', linestyle='-', linewidth=2.5, alpha=0.7, 
                       label='過買(70)')
            
            # 背景を塗る
            ax2.fill_between(x_indices, 30, 70, alpha=0.15, color='gray')
            
            # 軸とラベルを設定
            ax2.set_ylabel('RSI', fontsize=12, weight='bold')
            ax2.set_xlabel('時刻（左から古い順）', fontsize=12, weight='bold')
            ax2.set_ylim([0, 100])
            ax2.set_xticks(x_indices)
            ax2.set_xticklabels(time_labels, rotation=45, ha='right')
            
            # グリッドと凡例
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='upper left', fontsize=11, framealpha=0.95)
            
            # 全体を調整
            plt.tight_layout()
            
            # ファイルに保存
            filename = f'chart_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"[OK] グラフを作成しました: {filename}")
            return filename
        
        except Exception as e:
            logger.error(f"グラフ作成エラー: {e}")
            return None


class EmailNotifier:
    """メール通知"""
    
    @staticmethod
    def send_signal_email(signal: Signal, gmail_address: str, 
                         gmail_password: str, chart_file: Optional[str] = None) -> bool:
        """
        シグナルをメールで通知
        
        Args:
            signal: シグナルオブジェクト
            gmail_address: Gmail アドレス
            gmail_password: Gmail アプリケーションパスワード
            chart_file: グラフファイルパス（オプション）
        
        Returns:
            成功したかどうか
        """
        
        try:
            # メール本文を作成
            signal_emoji = "[BUY]" if signal.signal_type == "BUY" else "[SELL]"
            
            subject = f"USD/JPY: {signal_emoji} {signal.signal_type} シグナル (強度{signal.strength:.0f}%)"
            
            body = f"""
FX取引シグナルが発生しました

【シグナル情報】
シグナルタイプ: {signal.signal_type}
時刻: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
現在価格: {signal.price:.4f} 円
強度: {signal.strength:.1f}%
時間足: {signal.timeframe}

【推奨売買額】
推奨エントリー: {signal.stop_loss:.4f} - {signal.take_profit_1:.4f} 円
ストップロス: {signal.stop_loss:.4f} 円
利確目標1: {signal.take_profit_1:.4f} 円
利確目標2: {signal.take_profit_2:.4f} 円

【指標情報】
"""
            
            for key, value in signal.indicators.items():
                if value != '':
                    body += f"{key}: {value}\n"
            
            body += f"\n【判定理由】\n"
            for reason in signal.reasoning:
                body += f"- {reason}\n"
            
            # メッセージを作成
            msg = MIMEMultipart()
            msg['From'] = gmail_address
            msg['To'] = gmail_address
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # グラフを添付
            if chart_file and os.path.exists(chart_file):
                try:
                    with open(chart_file, 'rb') as attachment:
                        image = MIMEImage(attachment.read())
                        image.add_header('Content-ID', '<chart>')
                        image.add_header('Content-Disposition', 'attachment', filename=chart_file)
                        msg.attach(image)
                except Exception as e:
                    logger.warning(f"グラフ添付エラー: {e}")
            
            # Gmail で送信
            smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            smtp_server.login(gmail_address, gmail_password)
            smtp_server.send_message(msg)
            smtp_server.quit()
            
            logger.info(f"[OK] メール送信成功: {gmail_address}")
            return True
        
        except Exception as e:
            logger.error(f"メール送信エラー: {e}")
            logger.info("対策: Gmail のアプリケーションパスワードを確認してください")
            return False


class EnhancedSignalGenerator:
    """改良版シグナル生成エンジン"""
    
    def __init__(self):
        self.ta = EnhancedTechnicalAnalysis()
        self.va = VolatilityAnalysis()
        self.tcp = TargetPriceCalculator()
        self.signals_history: List[Signal] = []
    
    def generate_signal(self, symbol: str, df: pd.DataFrame, 
                       timeframe: str = "SHORT_TERM") -> Optional[Signal]:
        """改良版シグナル生成"""
        
        if df is None or len(df) < 50:
            return None
        
        close = df['close']
        
        # ボラティリティ分析
        atr = self.va.calculate_atr(df, period=14)
        hv = self.va.calculate_historical_volatility(close, period=20)
        vol_ratio = self.va.calculate_volatility_ratio(close, short_period=10, long_period=20)
        
        latest_atr = atr.iloc[-1]
        latest_hv = hv.iloc[-1]
        latest_vol_ratio = vol_ratio.iloc[-1]
        
        # パラメータ設定
        if timeframe == "SHORT_TERM":
            ma_fast_period = 5
            ma_slow_period = 10
            ema_period = 3
            rsi_overbuy = 75
            rsi_oversell = 25
            parameter_name = "短期(高感度)"
        else:
            ma_fast_period = 20
            ma_slow_period = 50
            ema_period = 12
            rsi_overbuy = 70
            rsi_oversell = 30
            parameter_name = "長期(低感度)"
        
        # テクニカル指標を計算
        ma_fast = self.ta.calculate_sma(close, ma_fast_period)
        ma_slow = self.ta.calculate_sma(close, ma_slow_period)
        ema = self.ta.calculate_ema(close, ema_period)
        rsi = self.ta.calculate_rsi(close, 14)
        macd_line, signal_line, histogram = self.ta.calculate_macd(close, 12, 26, 9)
        upper_bb, middle_bb, lower_bb = self.ta.calculate_bollinger_bands(close, 20, 2.0)
        k_stoch, d_stoch = self.ta.calculate_stochastic(df, period=14)
        
        # 最新値を取得
        latest_close = close.iloc[-1]
        latest_ma_fast = ma_fast.iloc[-1]
        latest_ma_slow = ma_slow.iloc[-1]
        latest_ema = ema.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        latest_macd = macd_line.iloc[-1]
        latest_signal_line = signal_line.iloc[-1]
        latest_hist = histogram.iloc[-1]
        latest_upper_bb = upper_bb.iloc[-1]
        latest_lower_bb = lower_bb.iloc[-1]
        latest_k_stoch = k_stoch.iloc[-1]
        latest_d_stoch = d_stoch.iloc[-1]
        
        prev_hist = histogram.iloc[-2] if len(histogram) > 1 else 0
        
        # シグナル判定
        buy_count = 0
        sell_count = 0
        reasoning = []
        
        if latest_close > latest_ma_fast > latest_ma_slow:
            buy_count += 1
            reasoning.append(f"MA: 上昇トレンド({parameter_name})")
        elif latest_close < latest_ma_fast < latest_ma_slow:
            sell_count += 1
            reasoning.append(f"MA: 下降トレンド({parameter_name})")
        
        if latest_close > latest_ema:
            buy_count += 1
            reasoning.append("EMA: 価格がEMA上")
        else:
            sell_count += 1
            reasoning.append("EMA: 価格がEMA下")
        
        if latest_rsi < rsi_oversell:
            buy_count += 1
            reasoning.append(f"RSI: 過売状態 (<{rsi_oversell})")
        elif latest_rsi > rsi_overbuy:
            sell_count += 1
            reasoning.append(f"RSI: 過買状態 (>{rsi_overbuy})")
        
        if latest_hist > 0 and prev_hist <= 0:
            buy_count += 1
            reasoning.append("MACD: 正転")
        elif latest_hist < 0 and prev_hist >= 0:
            sell_count += 1
            reasoning.append("MACD: 負転")
        
        if latest_macd > latest_signal_line:
            buy_count += 1
            reasoning.append("MACD: ライン>シグナル")
        else:
            sell_count += 1
            reasoning.append("MACD: ライン<シグナル")
        
        if latest_close < latest_lower_bb:
            buy_count += 1
            reasoning.append("BB: 下部以下")
        elif latest_close > latest_upper_bb:
            sell_count += 1
            reasoning.append("BB: 上部以上")
        
        if latest_k_stoch < 20:
            buy_count += 1
            reasoning.append("Stochastic: 過売")
        elif latest_k_stoch > 80:
            sell_count += 1
            reasoning.append("Stochastic: 過買")
        
        # ボラティリティ判定
        if latest_vol_ratio > 1.2:
            reasoning.append(f"WARNING: ボラティリティ上昇({latest_vol_ratio:.2f})")
            if buy_count > 0:
                buy_count = max(buy_count - 0.5, 0)
            if sell_count > 0:
                sell_count = max(sell_count - 0.5, 0)
        
        if latest_vol_ratio < 0.8:
            reasoning.append(f"INFO: ボラティリティ低い(安定{latest_vol_ratio:.2f})")
            if buy_count > 0:
                buy_count += 0.5
            if sell_count > 0:
                sell_count += 0.5
        
        # シグナル強度を計算
        total_indicators = buy_count + sell_count
        if total_indicators > 0:
            signal_strength = max(buy_count, sell_count) / total_indicators * 100
        else:
            signal_strength = 0
        
        # シグナル判定
        if buy_count > sell_count:
            signal_type = "BUY"
        elif sell_count > buy_count:
            signal_type = "SELL"
        else:
            signal_type = "NEUTRAL"
        
        # ターゲット価格を計算
        targets = self.tcp.calculate_targets(latest_close, latest_atr, signal_type)
        
        # 指標情報
        indicators_dict = {
            '【トレンド】': '',
            'MA_Fast': round(latest_ma_fast, 4),
            'MA_Slow': round(latest_ma_slow, 4),
            'EMA': round(latest_ema, 4),
            '【モメンタム】': '',
            'RSI': round(latest_rsi, 2),
            'Stochastic_K': round(latest_k_stoch, 2),
            'Stochastic_D': round(latest_d_stoch, 2),
            '【トレンド転換】': '',
            'MACD': round(latest_macd, 6),
            'Signal_Line': round(latest_signal_line, 6),
            'Histogram': round(latest_hist, 6),
            '【オシレータ】': '',
            'BB_Upper': round(latest_upper_bb, 4),
            'BB_Middle': round(middle_bb.iloc[-1], 4),
            'BB_Lower': round(latest_lower_bb, 4),
            '【ボラティリティ】': '',
            'ATR': round(latest_atr, 4),
            'Historical_Volatility': round(latest_hv, 4),
            'Volatility_Ratio': round(latest_vol_ratio, 2),
        }
        
        signal = Signal(
            timestamp=datetime.now(),
            pair=symbol,
            signal_type=signal_type,
            strength=signal_strength,
            price=latest_close,
            timeframe=timeframe,
            indicators=indicators_dict,
            reasoning=reasoning,
            target_price=targets['entry_price_high'] if signal_type == "BUY" else targets['entry_price_low'],
            stop_loss=targets['stop_loss'],
            take_profit_1=targets['take_profit_1'],
            take_profit_2=targets['take_profit_2']
        )
        
        self.signals_history.append(signal)
        return signal


class CompleteTradingMonitor:
    """完全統合型トレーディング監視"""
    
    def __init__(self, symbol: str = "USDJPY=X", 
                 gmail_address: Optional[str] = None,
                 gmail_password: Optional[str] = None):
        self.symbol = symbol
        self.gmail_address = gmail_address
        self.gmail_password = gmail_password
        self.engine = EnhancedSignalGenerator()
        self.fetcher = RealDataFetcher()
        self.visualizer = GraphVisualizer()
        self.notifier = EmailNotifier()
    
    def run_analysis(self):
        """完全分析を実行"""
        
        print("\n" + "=" * 80)
        print("FX Trading Signal Generator - 完全統合版")
        print("=" * 80)
        
        # データを取得
        print("\n【Yahoo Finance からデータ取得中...】")
        df = self.fetcher.get_real_data(symbol=self.symbol, period="100d", interval="1h")
        
        if df is None or len(df) < 50:
            print("[ERROR] データが不足しています")
            return
        
        print(f"[OK] 取得完了: {len(df)}本のキャンドル")
        print(f"  最新価格: {df['close'].iloc[-1]:.4f}")
        
        # 短期シグナルを生成
        print("\n【短期シグナル(高感度 - 5分足相当)】")
        print("-" * 80)
        short_signal = self.engine.generate_signal(self.symbol, df, timeframe="SHORT_TERM")
        self._display_complete_signal(short_signal)
        
        # 長期シグナルを生成
        print("\n【長期シグナル(低感度 - 1時間足相当)】")
        print("-" * 80)
        long_signal = self.engine.generate_signal(self.symbol, df, timeframe="LONG_TERM")
        self._display_complete_signal(long_signal)
        
        # 比較
        print("\n" + "=" * 80)
        print("【シグナル比較】")
        print("=" * 80)
        self._compare_signals(short_signal, long_signal)
        
        # グラフを作成
        if short_signal and short_signal.signal_type != "NEUTRAL":
            print("\n【グラフを作成中...】")
            chart_file = self.visualizer.create_chart(df, short_signal, symbol="USDJPY")
            
            # メール通知を送信
            if self.gmail_address and self.gmail_password and chart_file:
                print("\n【メール通知を送信中...】")
                self.notifier.send_signal_email(
                    short_signal,
                    self.gmail_address,
                    self.gmail_password,
                    chart_file
                )
    
    def _display_complete_signal(self, signal):
        """詳細なシグナル情報を表示"""
        if signal is None:
            print("[ERROR] シグナルを生成できませんでした")
            return
        
        signal_mark = "[BUY]" if signal.signal_type == "BUY" else "[SELL]" if signal.signal_type == "SELL" else "[NEUTRAL]"
        
        print(f"\n{signal_mark} {signal.signal_type} シグナル (強度: {signal.strength:.1f}%)")
        print(f"時刻: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"価格: {signal.price:.4f}")
        
        # ターゲット価格を表示
        if signal.signal_type != "NEUTRAL":
            print(f"\n【推奨売買額】")
            if signal.signal_type == "BUY":
                print(f"推奨買値:    {signal.target_price:.4f}円")
            else:
                print(f"推奨売値:    {signal.target_price:.4f}円")
            print(f"ストップロス: {signal.stop_loss:.4f}円")
            print(f"利確目標1:   {signal.take_profit_1:.4f}円")
            print(f"利確目標2:   {signal.take_profit_2:.4f}円")
            
            print(f"\n【リスク管理】")
            risk = abs(signal.stop_loss - signal.price)
            reward_1 = abs(signal.take_profit_1 - signal.price)
            reward_2 = abs(signal.take_profit_2 - signal.price)
            print(f"リスク:      {risk:.4f}円")
            print(f"リワード1:   {reward_1:.4f}円")
            print(f"リワード2:   {reward_2:.4f}円")
            print(f"リスク・リワード比: {reward_1/risk:.2f}:1")
        
        print(f"\n【指標情報】")
        for key, value in signal.indicators.items():
            if value == '':
                print(f"\n{key}")
            else:
                print(f"  {key:25}: {value}")
        
        print(f"\n【判定理由】")
        for reason in signal.reasoning:
            print(f"  - {reason}")
    
    def _compare_signals(self, short_signal, long_signal):
        """シグナルを比較"""
        if short_signal and long_signal:
            print(f"\n短期: {short_signal.signal_type:6} (強度: {short_signal.strength:6.1f}%)")
            print(f"長期: {long_signal.signal_type:6} (強度: {long_signal.strength:6.1f}%)")
            
            if short_signal.signal_type == long_signal.signal_type:
                print("\n[OK] 両者が一致しています(強いシグナル)")
            else:
                print("\n[WARNING] 短期と長期でシグナルが異なります")


def main():
    """メイン実行"""
    
    # Gmail 設定（オプション）
    gmail_address = "perfectearthtn@gmail.com"
    gmail_password = "cmes mqar mtys eppt"

    
    # メール通知を有効にする場合は、ここに設定してください
    # gmail_address = "perfectearthtn@gmail.com"
    # gmail_password = "cmes mqar mtys eppt"  # Gmailアプリケーションパスワード
    
    print("\n" + "=" * 80)
    print("Gmail 通知設定")
    print("=" * 80)
    print("\nメール通知機能を使用するには:")
    print("1. fx_trading_complete.py を開く")
    print("2. main() 関数内の以下の部分を編集:")
    print("   gmail_address = 'your-email@gmail.com'")
    print("   gmail_password = 'your-app-password'")
    print("3. 保存して再度実行")
    print("\n詳細: https://support.google.com/accounts/answer/185833")
    
    monitor = CompleteTradingMonitor(
        symbol="USDJPY=X",
        gmail_address=gmail_address,
        gmail_password=gmail_password
    )
    monitor.run_analysis()


if __name__ == "__main__":
    main()
