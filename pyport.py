import os
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import csv
from collections import Counter
from scipy.cluster.hierarchy import dendrogram
from labellines import labelLine, labelLines
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py
import json
from mlfinlab.portfolio_optimization.herc import HierarchicalEqualRiskContribution
from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity
from mlfinlab.portfolio_optimization.mean_variance import MeanVarianceOptimisation, ReturnsEstimators
from mlfinlab.portfolio_optimization import CriticalLineAlgorithm
from mlfinlab.portfolio_optimization.nco import NCO
from mlfinlab.online_portfolio_selection.rmr import RMR
from mlfinlab.online_portfolio_selection.olmar import OLMAR
from mlfinlab.online_portfolio_selection.fcornk import FCORNK
from mlfinlab.online_portfolio_selection.scorn import SCORN
from mlfinlab.microstructural_features.third_generation import get_vpin
from mlfinlab.data_structures import standard_data_structures
from mlfinlab.backtest_statistics import sharpe_ratio, drawdown_and_time_under_water
import matplotlib.pyplot as plt
import yfinance as yf

with open('config.json') as config_file:
    config = json.load(config_file)
test_mode = config['test_mode']
time_period_in_yrs = config['time_period_in_yrs']
input_files = config['input_files']
ignored_symbols = config['ignored_symbols']
use_cached_data = config['use_cached_data']
min_weight = config['min_weight']
verbose = config['verbose']
plot_returns = config['plot_returns']
use_plotly = config['use_plotly']
models = config['models']
optimization_config = config['optimization_config']
sort_by_weights = config['sort_by_weights']
stk = {}

for times in time_period_in_yrs:
    FOLDER = '{}yr'.format(times)
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
    CWD = os.getcwd() + '/'
    PATH = CWD + FOLDER + '/'
    DATE = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    TODAY = datetime.today()
    START_DATE = (
        TODAY +
        relativedelta(
            months=-
            round(
                times *
                12))).strftime('%Y-%m-%d')
    END_DATE = (TODAY + relativedelta(days=1)).strftime('%Y-%m-%d')

    # get ticker symbols
    symbols = []
    for input_file in input_files:
        input_file += '.csv'
        with open(CWD + input_file) as file:
            for line in file:
                name = line.rstrip()
                # name = name.split('.')[0]
                if not name.startswith("#") and name[:1].isalpha() and name.upper() not in [
                        x.upper() for x in ignored_symbols]:
                    symbols.append(name.upper())
    symbols = list(set(symbols))
    if test_mode:
        print(symbols)

    def get_stock_data(sym):
        df_sym = yf.download(
            sym,
            start=START_DATE,
            end=END_DATE)
        df_sym.to_csv(sym_file)
        return df_sym

    df = pd.DataFrame()
    for sym in symbols:
        if not sym:
            continue
        sym_file = PATH + '{}.csv'.format(sym)

        if use_cached_data:
            try:
                mod_time = datetime.fromtimestamp(os.path.getmtime(sym_file))
                time_elapsed = TODAY - \
                    mod_time.replace(hour=0, minute=0, second=0, microsecond=0)
                needs_refresh = time_elapsed > timedelta(
                    days=max(7, int(0.07 * times)))
            except BaseException:
                use_cached_data = False

        if not use_cached_data:
            print(
                '{} local data cache out of date. downloading latest price data...'.format(sym))
            df_sym = get_stock_data(sym)
        else:
            df_sym = pd.read_csv(sym_file, parse_dates=True, index_col="Date")
            if df_sym.empty:
                df_sym = get_stock_data(sym)

        df_sym.rename(columns={'Adj Close': sym}, inplace=True)
        df_sym.drop(['Open', 'High', 'Low', 'Close',
                     'Volume'], 1, inplace=True)

        if df.empty:
            df = df_sym
        else:
            df = df.join(df_sym, how='outer')

    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df = df.reindex(sorted(df.columns), axis=1)

    if test_mode:
        df = df.head(int(len(df) * 0.72))
        print(df.head(), df.tail(), 'Null check: ', df.isnull().values.any())

    def output(
            weights,
            sort_by_weights=False,
            optimization_method=None,
            time_period=times):
        if isinstance(weights, dict):
            clean_weights = weights
        else:
            clean_weights = weights.to_dict('records')[0]
        if test_mode:
            print('raw weights', clean_weights)

        clipped = {k: v for k, v in clean_weights.items() if v > min_weight}
        scaled = scale_to_one(clipped)
        stk[optimization_method + str(times)] = scaled

        if len(scaled) > 0:
            portfolio = df[scaled.keys()]
            asset_returns = np.log(portfolio) - np.log(portfolio.shift(1))
            asset_returns = asset_returns.iloc[1:, :]

            def apply_weights(row, weights):
                for i, _ in enumerate(row):
                    row[i] *= weights[i]
                return row

            portfolio_returns = asset_returns.apply(
                lambda row: apply_weights(
                    row,
                    list(scaled.values())
                ), axis=1).sum(axis=1)

            cumulative_returns = portfolio_returns.add(1).cumprod()
            sharpe = sharpe_ratio(portfolio_returns)

            dd, tuw = drawdown_and_time_under_water(cumulative_returns)
            mdd, dd_time = dd.max(), tuw[dd.idxmax()] * 252

            print(
                '\n{} to {} ({} yrs)'.format(
                    START_DATE,
                    END_DATE,
                    time_period))
            print('optimization method:', optimization_method)
            print('sharpe ratio:', round(sharpe, 2))
            print('drawdown: -{}, {} days to recover after {}'.format(
                round(mdd, 2),
                round(dd_time, 2),
                dd.idxmax().date()
            ))
            print('run at:', datetime.now())
            print('portfolio allocation weights (min {}):'.format(min_weight))
        else:
            print('max diversification recommended')

        if sort_by_weights:
            for sym, weight in sorted(
                scaled.items(), key=lambda kv: (
                    kv[1], kv[0]), reverse=True):
                print(sym, '\t% 5.3f' % (weight))
        else:
            for sym, weight in sorted(scaled.items()):
                print(sym, '\t% 5.3f' % (weight))

    def scale_to_one(weights):
        total_alloc = sum(weights.values())
        scaled = {
            k: v /
            total_alloc for k,
            v in weights.items() if v /
            total_alloc > min_weight}
        return scaled

    for optimization_method in models:
        temp = None
        print('\nCalculating...', optimization_method)
        optimization_method = optimization_method.lower()

        if (optimization_method == 'hrp'):
            temp = HierarchicalRiskParity()
            temp.allocate(
                asset_prices=df,
                linkage=optimization_config[optimization_method]['linkage'],
            )
        elif (optimization_method.find('herc') != -1):
            temp = HierarchicalEqualRiskContribution()
            temp.allocate(
                asset_prices=df,
                risk_measure=optimization_config[optimization_method]['risk_measure'],
                linkage=optimization_config[optimization_method]['linkage'])
        elif (optimization_method.find('nco') != -1):
            asset_returns = np.log(df) - np.log(df.shift(1))
            asset_returns = asset_returns.iloc[1:, :] 
            temp = NCO()
            if optimization_config[optimization_method]['sharpe']:
                mu_vec = np.array(asset_returns.mean())
            else:
                mu_vec = np.ones(len(df.columns))
            weights = temp.allocate_nco(
                cov=np.array(asset_returns.cov()),
                mu_vec=mu_vec.reshape(-1, 1)
            )
            temp_dict = dict(zip(df.columns, weights))
            temp.weights = temp_dict
        elif (optimization_method.find('cla') != -1):
            temp = CriticalLineAlgorithm()
            solution = optimization_config[optimization_method]['solution']
            temp.allocate(
                asset_prices=df,
                solution=solution)
        elif (optimization_method == 'olmar'):
            temp = OLMAR(
                reversion_method=optimization_config[optimization_method]['method'],
                epsilon=optimization_config[optimization_method]['epsilon'],
                window=optimization_config[optimization_method]['window'],
                alpha=optimization_config[optimization_method]['alpha'])
            temp.allocate(asset_prices=df, verbose=verbose)
            temp_dict = dict(zip(df.columns, temp.weights))
            temp.weights = temp_dict
        elif (optimization_method == 'rmr'):
            temp = RMR(
                epsilon=optimization_config[optimization_method]['epsilon'],
                n_iteration=optimization_config[optimization_method]['n_iteration'],
                tau=optimization_config[optimization_method]['tau'],
                window=optimization_config[optimization_method]['window'])
            temp.allocate(
                asset_prices=df,
                resample_by=optimization_config[optimization_method]["resample"],
                verbose=verbose)
            temp_dict = dict(zip(df.columns, temp.weights))
            temp.weights = temp_dict
        elif(optimization_method == 'scorn'):
            temp = SCORN(
                window=optimization_config[optimization_method]['window'],
                rho=optimization_config[optimization_method]['rho'])
            temp.allocate(
                asset_prices=df,
                resample_by=optimization_config[optimization_method]["resample"],
                verbose=verbose)
            temp_dict = dict(zip(df.columns, temp.weights))
            temp.weights = temp_dict
        elif(optimization_method == 'fcornk'):
            temp = FCORNK(
                window=optimization_config[optimization_method]['window'],
                rho=optimization_config[optimization_method]['rho'],
                lambd=optimization_config[optimization_method]['lambd'],
                k=optimization_config[optimization_method]['k'],
            )
            temp.allocate(
                asset_prices=df,
                resample_by=optimization_config[optimization_method]["resample"],
                verbose=verbose)
            temp_dict = dict(zip(df.columns, temp.weights[0]))
            temp.weights = temp_dict
        else:
            temp = MeanVarianceOptimisation()
            expected_returns = ReturnsEstimators(
            ).calculate_mean_historical_returns(asset_prices=df)
            covariance = ReturnsEstimators().calculate_returns(asset_prices=df).cov()
            temp.allocate(asset_names=df.columns,
                          asset_prices=df,
                          expected_asset_returns=expected_returns,
                          covariance_matrix=covariance,
                          solution=optimization_method,
                          target_return=optimization_config['efficient_risk'],
                          target_risk=optimization_config['efficient_return'],
                          risk_aversion=optimization_config['risk_aversion'],
                          )
            temp.get_portfolio_metrics()

        output(temp.weights, sort_by_weights, optimization_method)

if len(stk) > 1:
    t = [v for k, v in stk.items()]
    total = sum(map(Counter, t), Counter())
    N = float(len(stk))
    avg = {k: v / N for k, v in total.items()}
    print('input files:', input_files)
    output(avg, sort_by_weights=True, optimization_method=', '.join(
        models), time_period=', '.join([str(t) for t in time_period_in_yrs]))

if plot_returns:
    daily_returns = df.pct_change()
    if test_mode:
        print(daily_returns.head())
    cumulative_returns = daily_returns.add(1).cumprod().sub(1).mul(100)
    if test_mode:
        print(cumulative_returns.head())
    if sort_by_weights:
        sorted_cols = cumulative_returns.sort_values(
            cumulative_returns.index[-1],
            ascending=False,
            axis=1
        ).columns
        cumulative_returns = cumulative_returns[sorted_cols]

        if not use_plotly:
            ax1 = daily_returns.plot(
                kind='box',
                title='daily returns',
                grid=True,
                legend=None
            )

            try:
                labelLines(plt.gca().get_lines(), align=False, zorder=2.5)
            except BaseException:
                pass

            ax2 = cumulative_returns.plot(
                colormap='rainbow',
                title='cumulative returns',
                grid=True,
                legend=None
            )

            for line, name in zip(ax2.lines, cumulative_returns.columns):
                y = line.get_ydata()[-1]
                percent = "{} {:.2f}%".format(name, y)
                ax2.annotate(percent,
                             xy=(1, y),
                             xytext=(-25, 0),
                             color="w",
                             xycoords=ax2.get_yaxis_transform(),
                             textcoords="offset points",
                             bbox=dict(
                                 boxstyle="round, pad=0.5",
                                 fc=line.get_color(),
                                 edgecolor="none"),
                             fontsize=8,
                             va="center", ha="center"
                             )
            try:
                labelLines(plt.gca().get_lines(), align=False, zorder=2.5)
            except BaseException:
                pass
            plt.tight_layout()
            plt.show(block=False)

        else:
            fig = px.line(cumulative_returns, title="Cumulative Returns")
            c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0,
                                                                         360, len(daily_returns.columns))]
            fig2 = go.Figure(data=[go.Box(
                y=daily_returns[col],
                marker_color=c[i],
                name=col
            ) for i, col in enumerate(daily_returns.columns)])

            fig2.update_layout(
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False),
                yaxis=dict(
                    zeroline=False,
                    gridcolor='white'),
                paper_bgcolor='rgb(233,233,233)',
                plot_bgcolor='rgb(233,233,233)',
            )
        fig.show()
        fig2.show()
    plt.show()
