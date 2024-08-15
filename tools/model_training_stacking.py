import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn import metrics
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, LassoLars, 
                                  PassiveAggressiveRegressor, TheilSenRegressor, HuberRegressor, 
                                  OrthogonalMatchingPursuit, TweedieRegressor, PoissonRegressor, 
                                  GammaRegressor)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import LinearSVR, SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, 
                              AdaBoostRegressor, HistGradientBoostingRegressor)
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, Lasso, ElasticNet, SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from itertools import combinations
import argparse

# 初始化评估指标列表
models_name = []
RR = []
MAE = []
MSE = []
RMSE = []
MAE_B = []
RMSE_B = []
n_model = []


# 定义模型名称缩写字典
#  model_abbreviations = {
#     'ExtraTreesRegressor': 'ET',
#     'RandomForestRegressor': 'RF',
#     'LGBMRegressor': 'LGBM',
#     'BayesianRidge': 'BR',
#     'Lasso': 'LS',
#     'AdaBoostRegressor': 'AB',
#     'HistGradientBoostingRegressor': 'HGBR',
#     'SVR': 'SVR',
#     'DecisionTreeRegressor': 'DT'
#  }

# 生成模型名称的函数
def get_model_name(models):
    # 使用自定义缩写生成堆叠模型的名称
    model_names = []
    for model in models:
        model_name = type(model).__name__
        short_name = {
    'ExtraTreesRegressor': 'ET',
    'RandomForestRegressor': 'RF',
    'LGBMRegressor': 'LGBM',
    'BayesianRidge': 'BR',
    'Lasso': 'LS',
    'AdaBoostRegressor': 'AB',
    'HistGradientBoostingRegressor': 'HGBR',
    'SVR': 'SVR',
    'DecisionTreeRegressor': 'DT'
}.get(model_name, model_name)  # 获取缩写
        model_names.append(short_name)
        print(f"原始模型名称: {model_name}, 缩写: {short_name}")  # 打印每个模型的原始名称和缩写
    
    model_name = '_'.join(model_names)
    if not model_name.endswith('_stacking'):
        model_name += '_stacking'
    
    print(f"最终生成的模型名称: {model_name}")  # 打印生成的模型全名
    return model_name


# 根据命令行参数选择模型
def select_models(selected_names):
    base_models = {
        'ExtraTrees': ExtraTreesRegressor(n_estimators=40, random_state=45),
        'RandomForest': RandomForestRegressor(n_estimators=20, random_state=45),
        'LGBM': lgb.LGBMRegressor(n_estimators=20, random_state=45),
        'BayesianRidge': BayesianRidge(),
        'Lasso': Lasso(random_state=45),
        'AdaBoost': AdaBoostRegressor(n_estimators=20, random_state=45),
        'HistGBR': HistGradientBoostingRegressor(random_state=45),
        'SVR': SVR(),
        'DecisionTree': DecisionTreeRegressor(random_state=45)
    }
    selected_models = [base_models[name] for name in selected_names if name in base_models]
    return selected_models

# 打印模型评估结果的函数
def bf_print(model_name, acc_rf, adjr2_rf, MAE_rf, MSE_rf, RMSE_rf, MAE_b, RMSE_b):
    col_widths = {
        "Model": max(len(model_name), 35),
        "R^2": 18,
        "Adj R^2": 18,
        "MAE": 18,
        "MSE": 18,
        "RMSE": 18,
        "%MAE": 18,
        "%RMSE": 18
    }

    def print_separator():
        print('+' + '-'*(col_widths["Model"] + 2) +
              '+' + '-'*(col_widths["R^2"] + 2) +
              '+' + '-'*(col_widths["Adj R^2"] + 2) +
              '+' + '-'*(col_widths["MAE"] + 2) +
              '+' + '-'*(col_widths["MSE"] + 2) +
              '+' + '-'*(col_widths["RMSE"] + 2) +
              '+' + '-'*(col_widths["%MAE"] + 2) +
              '+' + '-'*(col_widths["%RMSE"] + 2) + '+')

    def print_header():
        print(f'| {"Model":<{col_widths["Model"]}} | {"R^2":<{col_widths["R^2"]}} | {"Adj R^2":<{col_widths["Adj R^2"]}} | '
              f'{"MAE":<{col_widths["MAE"]}} | {"MSE":<{col_widths["MSE"]}} | {"RMSE":<{col_widths["RMSE"]}} | '
              f'{"%MAE":<{col_widths["%MAE"]}} | {"%RMSE":<{col_widths["%RMSE"]}} |')

    print_separator()
    print_header()
    print_separator()
    
    print(f'| {model_name:<{col_widths["Model"]}} | {acc_rf:<{col_widths["R^2"]}.4f} | {adjr2_rf:<{col_widths["Adj R^2"]}.4f} | '
          f'{MAE_rf:<{col_widths["MAE"]}.4f} | {MSE_rf:<{col_widths["MSE"]}.4f} | {RMSE_rf:<{col_widths["RMSE"]}.4f} | '
          f'{MAE_b:<{col_widths["%MAE"]}.2f} | {RMSE_b:<{col_widths["%RMSE"]}.2f} |')

    print_separator()

# 训练堆叠模型并保存结果的函数
def T_stacking(reg, model_name, X_train, y_train, X_test, y_test, data, model_save_path, output_dir):
    y_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)

    res = pd.DataFrame()
    res['pointID'] = list(X_test.index)
    res['真实值'] = list(y_test)
    res['预测值'] = y_test_pred
    y_test.index = list(range(len(y_test.index)))
    loss = [abs(y_test[i] - y_test_pred[i]) for i in range(len(y_test))]
    res['差值'] = loss

    point = data.loc[:, ['pointID']]
    res = pd.merge(res, point)
    res.set_index(['pointID'])

    os.makedirs(output_dir, exist_ok=True)
    
    # 打印保存文件的路径和名称
    csv_path = f'{output_dir}/{model_name}_stacking.csv'
    print(f"保存CSV文件路径: {csv_path}")
    res.to_csv(csv_path, index=False, float_format='%.2f', encoding='gbk')

    acc_rf = metrics.r2_score(y_test, y_test_pred)
    adjr2_rf = 1 - (1 - metrics.r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    MAE_rf = metrics.mean_absolute_error(y_test, y_test_pred)
    MSE_rf = metrics.mean_squared_error(y_test, y_test_pred)
    RMSE_rf = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    MAE_b = (MAE_rf / np.mean(np.abs(y_test))) * 100
    RMSE_b = (RMSE_rf / np.mean(np.abs(y_test))) * 100

    bf_print(
        model_name=model_name,  # 这里传入的是字符串
        acc_rf=acc_rf,
        adjr2_rf=adjr2_rf,
        MAE_rf=MAE_rf,
        MSE_rf=MSE_rf,
        RMSE_rf=RMSE_rf,
        MAE_b=MAE_b,
        RMSE_b=RMSE_b
    )

    models_name.append(f'{model_name}')
    RR.append(acc_rf)
    MAE.append(MAE_rf)
    MSE.append(MSE_rf)
    RMSE.append(RMSE_rf)
    MAE_B.append(MAE_b)
    RMSE_B.append(RMSE_b)
    n_model.append(model_name)
    
    # 打印保存模型的路径和名称
    model_save_full_path = f'{model_save_path}/{model_name}.pkl'
    print(f"保存模型路径: {model_save_full_path}")
    os.makedirs(model_save_path, exist_ok=True)
    joblib.dump(reg, model_save_full_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stacking regression models")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save CSV outputs")
    parser.add_argument("--comparison_save_path", type=str, required=True, help="Path to save the model comparison results")
    parser.add_argument("--min_len", type=int, default=2, help="Minimum number of models to combine")
    parser.add_argument("--max_len", type=int, default=4, help="Maximum number of models to combine")
    parser.add_argument("--selected_models", type=str, nargs='+', required=False, help="List of selected models for stacking")
    args = parser.parse_args()

    # 根据命令行选择模型
    if args.selected_models:
        base_models = select_models(args.selected_models)
    else:
        # 使用所有模型
        base_models = select_models([
            'ExtraTrees', 
            'RandomForest', 
            'LGBM',
            'BayesianRidge', 
            'Lasso', 
            'AdaBoost', 
            'HistGBR', 
            'SVR', 
            'DecisionTree'
        ])

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # 加载数据集
    data = pd.read_csv(args.data_path, encoding='utf-8')

    # 删除不需要的列
    X = data.drop(['pointID', 'Y'], axis=1)
    y = data['Y']

    # 数据标准化
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    X_standardized = pd.DataFrame(X_standardized, columns=X.columns)

    # 数据归一化
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X_standardized)
    X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

    # 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=30)

    print("X_train 的前几行：")
    print(X_train.head())
    print("y_train 的前几行：")
    print(y_train.head())

    # 创建堆叠回归模型
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression
    from mlxtend.regressor import StackingCVRegressor
    from itertools import combinations

    cv = KFold(n_splits=10)
    n_jobs = 1

    model_combinations = []
    for r in range(args.min_len, args.max_len + 1):
        combinations_r = combinations(base_models, r)
        model_combinations.extend(combinations_r)

    for i, models in enumerate(model_combinations, start=1):
        model_name = get_model_name(models)
        print(f"训练Stacking模型 {i}/{len(model_combinations)}: {model_name}")  # 输出当前训练的模型组合
        stacking_reg = StackingCVRegressor(regressors=models, meta_regressor=LinearRegression(), cv=cv, verbose=0, random_state=45, n_jobs=n_jobs)
        stacking_reg.fit(X_train, y_train)
        T_stacking(stacking_reg, model_name, X_train, y_train, X_test, y_test, data, args.model_save_path, args.output_dir)

    # 生成模型评估结果表格
    df = pd.DataFrame({
        'Model': models_name,
        'R^2': RR,
        'MAE': MAE,
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE (%)': MAE_B,
        'RMSE (%)': RMSE_B,
        'n_model': n_model
    })

    # 根据 R^2 对模型进行排序
    df = df.sort_values(by='R^2', ascending=False)

    print(df)
    os.makedirs(os.path.dirname(args.comparison_save_path), exist_ok=True)
    df.to_csv(args.comparison_save_path, index=False, float_format='%.6f', encoding='utf-8')
    print('完成操作')

































