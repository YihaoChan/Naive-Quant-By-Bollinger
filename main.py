import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date
import pyfolio as pf
from joblib import Parallel, delayed
import seaborn as sns
from tqdm import tqdm
import warnings
import multiprocessing

warnings.filterwarnings('ignore')
pd.set_option('display.width', None)


class BackTestByBollinger(object):
    def __init__(self,
                 future_data_path=None,
                 begin_time=None,
                 end_time=None,
                 init_cap=10000):
        self.future_data_path = future_data_path
        self.__check_path(future_data_path)

        # 合约名称
        self.contract_code = os.path.basename(self.future_data_path).split('.csv')[0]

        # 输出目录
        self.output_path = os.path.join('results', self.contract_code)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # 初始资金
        self.init_cap = init_cap

        # 回测起止时间
        self.begin_time = begin_time
        self.end_time = end_time

    def process_data(self):
        # 读取csv文件
        df = pd.read_csv(self.future_data_path, index_col=0)

        # 回测起止时间
        self.begin_time = self.__check_date(df, self.begin_time, begin=True)
        self.end_time = self.__check_date(df, self.end_time, begin=False)

        # 筛选出时间范围内的数据
        df['CLOCK'] = pd.to_datetime(df['CLOCK'])
        return df[
            (self.begin_time <= df['CLOCK']) & (df['CLOCK'] <= self.end_time)].reset_index().drop(
            ['SYMBOL'], axis=1)

    @staticmethod
    def __check_path(path):
        assert path is not None
        if not os.path.exists(path):
            raise ValueError("File path is not correct!")

    @staticmethod
    def __check_date(df, date_str, begin=True):
        if date_str is None:
            return df['CLOCK'].iloc[0] if begin else df['CLOCK'].iloc[-1]

        try:
            date.fromisoformat(date_str)
        except:
            raise ValueError("Datetime is not correct!")

        return date_str

    @staticmethod
    def calculate_bollinger_bands(df, window=20, no_std=2):
        """
        计算布林线
        """
        column_name = 'CLOSE'

        df['Observation'] = df[column_name]
        df['RollingMean'] = df[column_name].rolling(window).mean()

        std = df[column_name].rolling(window).std(ddof=0)
        df['UpperBound'] = df['RollingMean'] + no_std * std
        df['LowerBound'] = df['RollingMean'] - no_std * std

        return df

    @staticmethod
    def calculate_strategy_position(bollinger_df):
        """
        根据策略实现仓位
        趋势策略：
        当收盘价由下向上穿过上轨的时候，做多；然后由上向下穿过中轨的时候，平仓。
        当收盘价由上向下穿过下轨的时候，做空；然后由下向上穿过中轨的时候，平仓。
        """
        bollinger_df = bollinger_df.copy()

        # position: 持仓头寸，多仓为1，不持仓为0，空仓为-1
        bollinger_df['position'] = np.nan

        # 上穿上轨，做多
        bollinger_df['position'] = np.where(bollinger_df.Observation > bollinger_df.UpperBound, 1,
                                            bollinger_df['position'])

        # 下穿下轨，做空
        bollinger_df['position'] = np.where(bollinger_df.Observation < bollinger_df.LowerBound, -1,
                                            bollinger_df['position'])

        # 平仓时机，上下异号为穿越点
        cross = (bollinger_df.Observation - bollinger_df.RollingMean) / (
                bollinger_df.Observation.shift(1) - bollinger_df.RollingMean.shift(1))
        bollinger_df['position'] = np.where(cross < 0, 0, bollinger_df['position'])
        bollinger_df['position'] = bollinger_df['position'].ffill().fillna(0)

        # 做多开仓：当前指示多头头寸，且上一个交易日无仓位
        long_entry_condition = (bollinger_df['position'] == 1) & (bollinger_df['position'].shift(1) == 0)
        bollinger_df.loc[long_entry_condition, 'signal_long'] = 1

        # 做多平仓：当前指示无仓位，但上一个交易日为多仓
        long_exit_condition = (bollinger_df['position'] == 0) & (bollinger_df['position'].shift(1) == 1)
        bollinger_df.loc[long_exit_condition, 'signal_long'] = 0  # 不持仓

        # 做空开仓：当前指示空头头寸，且上一个交易日无仓位
        short_entry_condition = (bollinger_df['position'] == -1) & (bollinger_df['position'].shift(1) == 0)
        bollinger_df.loc[short_entry_condition, 'signal_short'] = -1

        # 做空平仓：当前指示无仓位，但上一个交易日为空仓
        short_exit_condition = (bollinger_df['position'] == 0) & (bollinger_df['position'].shift(1) == -1)
        bollinger_df.loc[short_exit_condition, 'signal_short'] = 0  # 不持仓

        return bollinger_df

    @staticmethod
    def calculate_returns(position_df):
        """
        计算收益率
        """
        # 指数环比昨日收益率
        position_df['market_changes'] = position_df['Observation'].pct_change().fillna(0)

        # 策略环比昨日收益率：交易信号出现后第二天交易
        # 收益率有正有负，乘上头寸1/-1，或者是0，如果有头寸就能产生收益，如果没有头寸说明策略没发出持仓信号
        position_df['strategy_returns'] = position_df['position'].shift(1) * position_df['market_changes']
        position_df['strategy_returns'] = position_df['strategy_returns'].fillna(0)

        # 策略累积收益率
        position_df['strategy_cum_returns'] = position_df['strategy_returns'].cumsum()

        return position_df

    def calculate_indicators(self, returns_df):
        # 每日收益率集合的均值
        avg = returns_df['strategy_returns'].mean()

        # 每日收益率集合的标准差
        std = returns_df['strategy_returns'].std()

        # 绝对收益率
        abs_return = returns_df['strategy_cum_returns'].iloc[-1]

        # 年化收益率
        dt = returns_df['CLOCK'].iloc[-1] - returns_df['CLOCK'].iloc[0]
        years = dt.days / 365
        annual_return = abs_return / years

        # 年化波动率
        annual_volatility = np.sqrt(365) * std

        # 夏普比率
        sharpe = np.sqrt(365) * (avg / std)

        # 最大回撤
        # 滚动最大值
        rolling_max = returns_df['strategy_cum_returns'].cummax()
        # 发生最大回撤的位置
        max_drawdown_down_idx = (rolling_max - returns_df['strategy_cum_returns']).argmax()
        # 最大回撤发生前取得最大收益的位置
        max_drawdown_up_idx = returns_df['strategy_cum_returns'][:max_drawdown_down_idx].argmax()
        # 最大回撤率
        max_drawdown_rate = round(
            (returns_df['strategy_cum_returns'].iloc[max_drawdown_up_idx] - returns_df['strategy_cum_returns'].iloc[
                max_drawdown_down_idx]) /
            returns_df['strategy_cum_returns'].iloc[max_drawdown_up_idx],
            6)
        # 最大回撤持续天数
        max_drawdown_span = (
                returns_df.iloc[max_drawdown_down_idx]['CLOCK'] - returns_df.iloc[max_drawdown_up_idx]['CLOCK']
        ).days

        # 最大回撤修复
        max_drawdown_recover_span = None
        # 最大回撤后的数据
        df_after_max_drawdown = returns_df.iloc[max_drawdown_down_idx:]
        # 超过发生最大回撤时的收益
        cmp = df_after_max_drawdown['strategy_cum_returns'] > returns_df['strategy_cum_returns'].iloc[
            max_drawdown_up_idx]
        df_larger_max_drawdown_up_returns = df_after_max_drawdown[cmp == False]
        # 最大回撤修复时间
        if not df_larger_max_drawdown_up_returns.empty:
            recover_time = df_larger_max_drawdown_up_returns.iloc[0]['CLOCK']
            max_drawdown_recover_span = (recover_time - returns_df.iloc[max_drawdown_down_idx]['CLOCK']).days

        # 卡玛比率
        calmar = round((abs_return / years) / max_drawdown_rate, 6)

        # 指标写入表格
        indicator_df = pd.DataFrame(
            columns=['ContractCode', 'begin', 'end', 'slippage', 'initCap',
                     'stddev', 'sharpe', 'calmar',
                     'AbsReturn', 'AnnualReturn', 'AnnualVolatility',
                     'MaxDrawdownRate', 'MaxDrawdownSpan', 'MaxDrawdownRecoverSpan',
                     'DataVersion'])
        indicator_df.loc[len(indicator_df.index)] = [
            self.contract_code, self.begin_time, self.end_time, 1, self.init_cap,
            round(std, 6), round(sharpe, 6), round(calmar, 6),
            round(abs_return, 6), round(annual_return, 6), round(annual_volatility, 6),
            round(max_drawdown_rate, 6), round(max_drawdown_span, 6), round(max_drawdown_recover_span, 6),
            '20210701'
        ]

        return indicator_df

    def plot_bollinger(self, bollinger_df):
        # 布林线
        plt.plot(bollinger_df['CLOCK'], bollinger_df[['Observation', 'RollingMean', 'UpperBound', 'LowerBound']],
                 linewidth=1)
        plt.title('Observation with Bollinger Bands')
        plt.savefig(os.path.join(self.output_path, 'bollinger.pdf'), dpi=1200, bbox_inches='tight', format='pdf',
                    pad_inches=0)
        plt.close()

    def plot_strategy_signal(self, position_df):
        # 做多开仓
        long_entry = position_df.loc[position_df['signal_long'] == 1][['CLOCK', 'Observation']]
        # 做多平仓
        long_exit = position_df.loc[position_df['signal_long'] == 0][['CLOCK', 'Observation']]

        # 做空开仓
        short_entry = position_df.loc[position_df['signal_short'] == -1][['CLOCK', 'Observation']]
        # 做空平仓
        short_exit = position_df.loc[position_df['signal_short'] == 0][['CLOCK', 'Observation']]

        # 布林线
        plt.plot(position_df['CLOCK'], position_df['Observation'], label='Observation', linewidth=1)
        plt.plot(position_df['CLOCK'], position_df['RollingMean'], label='RollingMean', linewidth=1)
        plt.plot(position_df['CLOCK'], position_df['UpperBound'], linewidth=1)
        plt.plot(position_df['CLOCK'], position_df['LowerBound'], linewidth=1)
        plt.fill_between(position_df['CLOCK'], position_df['UpperBound'], position_df['LowerBound'],
                         alpha=0.3, label='Bollinger Band')

        # 信号
        plt.scatter(long_entry['CLOCK'], long_entry['Observation'], color='r',
                    s=10, marker='^', label='Long Entry',
                    zorder=10)
        plt.scatter(long_exit['CLOCK'], long_exit['Observation'], color='r',
                    s=10, marker='x', label='Long Exit',
                    zorder=10)
        plt.scatter(short_entry['CLOCK'], short_entry['Observation'], color='b',
                    s=10, marker='^', label='Short Entry',
                    zorder=10)
        plt.scatter(short_exit['CLOCK'], short_exit['Observation'], color='b',
                    s=10, marker='x', label='Short Exit',
                    zorder=10)

        plt.title('Bollinger Band Strategy Trading Signals')
        plt.legend()

        plt.savefig(os.path.join(self.output_path, 'signals.pdf'),
                    dpi=1200, bbox_inches='tight', format='pdf', pad_inches=0)
        plt.close()

    def plot_holdings(self, returns_df):
        # 仓位模拟图
        returns_df['cash'] = self.init_cap
        pf.plot_long_short_holdings(
            returns=pd.Series(returns_df['strategy_returns'].values, index=returns_df['CLOCK']),
            positions=returns_df[['position', 'cash']].set_index(returns_df['CLOCK']))
        plt.savefig(os.path.join(self.output_path, 'holdings.pdf'), dpi=1200, bbox_inches='tight', format='pdf',
                    pad_inches=0)
        plt.close()

    def plot_pnl(self, returns_df):
        # PNL曲线
        plt.plot(returns_df['CLOCK'], returns_df['strategy_cum_returns'])
        plt.title('Cumulative Returns')
        plt.savefig(os.path.join(self.output_path, 'pnl.pdf'), dpi=1200, bbox_inches='tight', format='pdf',
                    pad_inches=0)
        plt.close()

    def record_indicators(self, indicator_df):
        # 回测指标
        indicator_df.to_csv(os.path.join(self.output_path, 'backtest.csv'))

    def optimize(self, df, window_candidate, no_std_candidate):
        sharpes = np.zeros([len(window_candidate), len(no_std_candidate)])

        for i in range(len(window_candidate)):
            for j in range(len(no_std_candidate)):
                bollinger_df = self.calculate_bollinger_bands(df,
                                                              window=window_candidate[i],
                                                              no_std=no_std_candidate[j])
                position_df = self.calculate_strategy_position(bollinger_df)
                returns_df = self.calculate_returns(position_df)
                indicator_df = self.calculate_indicators(returns_df)
                sharpes[i][j] = indicator_df['sharpe'].iloc[0]

        df = pd.DataFrame(sharpes, index=window_candidate, columns=no_std_candidate)
        sns.heatmap(df, annot=True, cmap='YlGnBu', cbar=False,
                    annot_kws={'size': 8, 'weight': 'bold', 'color': 'black'},
                    fmt='.3f')

        plt.xlabel('Window')
        plt.ylabel('#Std')
        plt.title('Sharpe w.r.t Parameters')

        plt.savefig(os.path.join(self.output_path, 'parameters_sensitivity.pdf'), dpi=1200, bbox_inches='tight',
                    format='pdf',
                    pad_inches=0)
        plt.close()


def backtest_one(future_data_path):
    bollinger = BackTestByBollinger(future_data_path=future_data_path,
                                    begin_time='2010-01-01',
                                    end_time=None,
                                    init_cap=100000)

    df = bollinger.process_data()
    bollinger_df = bollinger.calculate_bollinger_bands(df, window=20, no_std=2)
    position_df = bollinger.calculate_strategy_position(bollinger_df)
    returns_df = bollinger.calculate_returns(position_df)
    indicator_df = bollinger.calculate_indicators(returns_df)

    bollinger.plot_bollinger(bollinger_df)
    bollinger.plot_strategy_signal(position_df)
    bollinger.plot_holdings(returns_df)
    bollinger.plot_pnl(returns_df)
    bollinger.record_indicators(indicator_df)

    bollinger.optimize(df, window_candidate=range(10, 50, 5), no_std_candidate=range(1, 3, 1))


def backtest_parallel(path):
    future_data_paths = []
    for root, dirs, names in os.walk(path):
        for filename in names:
            future_data_paths.append(os.path.join(root, filename))

    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(backtest_one)(code_path) for code_path in tqdm(future_data_paths))


def main():
    dir_path = './futures_data/day_20220611/'
    backtest_parallel(dir_path)


if __name__ == '__main__':
    main()
