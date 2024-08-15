import os
import joblib
import pandas as pd
import numpy as np
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import StandardScaler


def bf_print(model_name, acc_rf, adjr2_rf, MAE_rf, MSE_rf, RMSE_rf, MAE_b, RMSE_b):
    # 计算每列的宽度
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

    # 打印表格的顶部边框
    def print_separator():
        print('+' + '-'*(col_widths["Model"] + 2) +
              '+' + '-'*(col_widths["R^2"] + 2) +
              '+' + '-'*(col_widths["Adj R^2"] + 2) +
              '+' + '-'*(col_widths["MAE"] + 2) +
              '+' + '-'*(col_widths["MSE"] + 2) +
              '+' + '-'*(col_widths["RMSE"] + 2) +
              '+' + '-'*(col_widths["%MAE"] + 2) +
              '+' + '-'*(col_widths["%RMSE"] + 2) + '+')

    # 打印表格的标题行
    def print_header():
        print(f'| {"Model":<{col_widths["Model"]}} | {"R^2":<{col_widths["R^2"]}} | {"Adj R^2":<{col_widths["Adj R^2"]}} | '
              f'{"MAE":<{col_widths["MAE"]}} | {"MSE":<{col_widths["MSE"]}} | {"RMSE":<{col_widths["RMSE"]}} | '
              f'{"%MAE":<{col_widths["%MAE"]}} | {"%RMSE":<{col_widths["%RMSE"]}} |')

    # 打印表格的分隔行
    print_separator()
    print_header()
    print_separator()
    
    # 打印表格的内容行
    print(f'| {model_name:<{col_widths["Model"]}} | {acc_rf:<{col_widths["R^2"]}.4f} | {adjr2_rf:<{col_widths["Adj R^2"]}.4f} | '
          f'{MAE_rf:<{col_widths["MAE"]}.4f} | {MSE_rf:<{col_widths["MSE"]}.4f} | {RMSE_rf:<{col_widths["RMSE"]}.4f} | '
          f'{MAE_b:<{col_widths["%MAE"]}.2f} | {RMSE_b:<{col_widths["%RMSE"]}.2f} |')

    # 打印表格的底部边框
    print_separator()





def train_and_evaluate_models_bagging(data_path, model_save_path, comparison_save_path, target, test_size=0.2, random_state=30, n_estimators=10, model_list=None):
    # 初始化评估指标列表
    models_name = []
    RR = []
    MAE = []
    MSE = []
    RMSE = []
    MAE_B = []
    RMSE_B = []

    # 加载数据集
    data = pd.read_csv(data_path, encoding="utf-8")

    # 验证数据是否包含 pointID 和目标列
    if 'pointID' in data.columns and target in data.columns:
        print("pointID 和目标列已经存在，将删除这两列以确保它们不作为特征。")
    else:
        raise ValueError("数据集不包含 pointID 或目标列，请检查数据集。")

    # 删除 pointID 和目标列
    X = data.drop([target, 'pointID'], axis=1)
    y = data[target]

    # 数据标准化
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    X_standardized = pd.DataFrame(X_standardized, columns=X.columns)

    # 数据归一化
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X_standardized)
    X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

    # 输出标准化和归一化后的特征数据以进行验证
    print("标准化后的特征数据的前几行：")
    print(X_standardized.head())
    print("归一化后的特征数据的前几行：")
    print(X_normalized.head())

    # 设置索引
    X_normalized.set_index(data['pointID'], inplace=True)
    y.index = data['pointID']

    # 检查是否仍然存在 pointID 或目标列
    if 'pointID' in X_normalized.columns or target in X_normalized.columns:
        raise ValueError("pointID 或目标列仍然存在于特征中，不能用于训练。")
    else:
        print("pointID 和目标列已成功从特征中删除。")

    # 切分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=test_size, random_state=random_state)

    # 初始化回归模型字典
    all_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=random_state),
        "Lasso": Lasso(random_state=random_state),
        "ElasticNet": ElasticNet(random_state=random_state),
        "BayesianRidge": BayesianRidge(),
        "LassoLars": LassoLars(random_state=random_state),
        "PassiveAggressiveRegressor": PassiveAggressiveRegressor(random_state=random_state),
        "TheilSenRegressor": TheilSenRegressor(random_state=random_state),
        "HuberRegressor": HuberRegressor(),
        "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit(),
        "TweedieRegressor": TweedieRegressor(),
        "PoissonRegressor": PoissonRegressor(),
        "GammaRegressor": GammaRegressor(),
        "RandomForestRegressor": RandomForestRegressor(random_state=random_state),
        "ExtraTreesRegressor": ExtraTreesRegressor(random_state=random_state),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
        "XGBRegressor": XGBRegressor(),
        "lgb.LGBMRegressor": lgb.LGBMRegressor(random_state=random_state),
        "AdaBoostRegressor": AdaBoostRegressor(random_state=random_state),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=random_state),
        "LinearSVR": LinearSVR(random_state=random_state),
        "SVR": SVR(),
        "NuSVR": NuSVR(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=random_state),
        "ExtraTreeRegressor": ExtraTreeRegressor(random_state=random_state),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "GaussianProcessRegressor": GaussianProcessRegressor(random_state=random_state)
    }

    # 根据 model_list 选择模型，或使用所有模型
    if model_list is None:
        selected_models = all_models
    else:
        selected_models = {name: all_models[name] for name in model_list if name in all_models}

    # 创建模型保存路径目录
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(os.path.dirname(comparison_save_path), exist_ok=True)
    
    # 创建 Bagging 模型并评估
    def evaluate_bagging_model(model, model_name):
        y_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

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
        os.makedirs(f'{model_save_path}/csv/', exist_ok=True)
        res.to_csv(f'{model_save_path}/csv/{model_name}_bagging.csv', index=False, float_format='%.2f', encoding='gbk')

        acc_rf = metrics.r2_score(y_test, y_test_pred)
        adjr2_rf = 1 - (1 - metrics.r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
        MAE_rf = metrics.mean_absolute_error(y_test, y_test_pred)
        MSE_rf = metrics.mean_squared_error(y_test, y_test_pred)
        RMSE_rf = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
        MAE_b = (MAE_rf / np.mean(np.abs(y_test))) * 100
        RMSE_b = (RMSE_rf / np.mean(np.abs(y_test))) * 100

        # # 打印模型评估结果
        # print(
        #     f"模型名称: {model_name}\n"
        #     f"R^2: {acc_rf:.4f}\n"
        #     f"调整后的 R^2: {adjr2_rf:.4f}\n"
        #     f"MAE: {MAE_rf:.4f}\n"
        #     f"MSE: {MSE_rf:.4f}\n"
        #     f"RMSE: {RMSE_rf:.4f}\n"
        #     f"百分比 MAE: {MAE_b:.2f}%\n"
        #     f"百分比 RMSE: {RMSE_b:.2f}%\n"
        # )
        
        
        # 打印模型评估结果
        bf_print(
            model_name=model_name,
            acc_rf=acc_rf,
            adjr2_rf=adjr2_rf,
            MAE_rf=MAE_rf,
            MSE_rf=MSE_rf,
            RMSE_rf=RMSE_rf,
            MAE_b=MAE_b,
            RMSE_b=RMSE_b
        )
        
        
        
        models_name.append(f'{model_name}_bagging')
        RR.append(acc_rf)
        MAE.append(MAE_rf)
        MSE.append(MSE_rf)
        RMSE.append(RMSE_rf)
        MAE_B.append(MAE_b)
        RMSE_B.append(RMSE_b)

        # 保存模型
        os.makedirs(f'{model_save_path}/pkl/', exist_ok=True)
        joblib.dump(model, f'{model_save_path}/pkl/{model_name}_bagging.pkl')

    for model_name, model in selected_models.items():
        print(f"训练 Bagging 模型: {model_name}")
        # print("----------------------------------------------------------------")
        bagging_model = BaggingRegressor(estimator=model, n_estimators=n_estimators, random_state=random_state, n_jobs=1)
        bagging_model.fit(X_train, y_train)
        evaluate_bagging_model(bagging_model, model_name)
        
    # 创建 DataFrame 保存评估结果
    df = pd.DataFrame({'模型名称': models_name, 'R^2': RR, 'MAE': MAE, 'MSE': MSE, 'RMSE': RMSE, '%MAE': MAE_B, '%RMSE': RMSE_B})

    # 根据 R^2 对模型进行排序
    df = df.sort_values(by='R^2', ascending=False)

    # 确保保存路径存在
    os.makedirs(os.path.dirname(comparison_save_path), exist_ok=True)
    print(df)
    # 保存评估结果到指定路径的 CSV 文件
    df.to_csv(comparison_save_path, index=True, float_format='%.6f', encoding='gbk')
    print('Bagging袋装法集成模型训练完成操作')
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Bagging ensemble learning for regression models.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file.')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save trained models.')
    parser.add_argument('--comparison_save_path', type=str, required=True, help='Path to save model comparison results.')
    parser.add_argument('--target', type=str, required=True, help='Target column name for regression.')
    parser.add_argument('--n_estimators', type=int, default=10, help='Number of Bagging estimators.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to be used as test set.')
    parser.add_argument('--random_state', type=int, default=30, help='Random seed for reproducibility.')
    parser.add_argument('--model_list', type=str, nargs='*', default=None, help='List of model names to train, e.g. "Ridge Lasso"')

    args = parser.parse_args()

    train_and_evaluate_models_bagging(
        data_path=args.data_path,
        model_save_path=args.model_save_path,
        comparison_save_path=args.comparison_save_path,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        model_list=args.model_list
    )