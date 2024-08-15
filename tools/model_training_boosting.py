import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn import metrics

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



def train_and_evaluate_models_boosting(data_path, model_save_path, comparison_save_path, num, target, test_size, random_state):
    # 初始化评估指标列表
    models_name = []
    RR = []
    MAE = []
    MSE = []
    RMSE = []
    MAE_B = []
    RMSE_B = []

    # 加载数据集
    try:
        train = pd.read_csv(data_path, encoding="utf-8")
        data = pd.read_csv(data_path, encoding="utf-8")
        print("成功使用 utf-8 编码读取数据。")
    except UnicodeDecodeError:
        # 如果 utf-8 读取失败，则使用 gbk 编码
        train = pd.read_csv(data_path, encoding="gbk")
        data = pd.read_csv(data_path, encoding="gbk")
        print("utf-8 读取失败，成功使用 gbk 编码读取数据。")
        
    X = train.drop([target,"pointID"], axis=1)
    y = train[target]

    for column in X.columns:
        mean = X[column].mean()
        std = X[column].std()
        X[column] = (X[column] - mean) / std

    X.set_index(data['pointID'], inplace=True)
    y.index = data['pointID']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    

    # 定义 Boosting 回归模型
    regression_models = {
        "AdaBoost": AdaBoostRegressor(n_estimators=num, learning_rate=2.0, loss='linear', random_state=45),
        "GradientBoosting": GradientBoostingRegressor(loss='squared_error', learning_rate=0.2, n_estimators=num, random_state=random_state),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=45),
        "XGBoost_boosting": xgb.XGBRegressor(n_estimators=num),
        "LGBM_boosting": lgb.LGBMRegressor(random_state=45, n_estimators=num)
    }

    def evaluate_boosting_model(model, model_name):
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
        res.to_csv(f'{model_save_path}/csv/{model_name}_boosting.csv', index=False, float_format='%.2f', encoding='gbk')

        acc_rf = metrics.r2_score(y_test, y_test_pred)
        adjr2_rf = 1 - (1 - metrics.r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
        MAE_rf = metrics.mean_absolute_error(y_test, y_test_pred)
        MSE_rf = metrics.mean_squared_error(y_test, y_test_pred)
        RMSE_rf = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
        MAE_b = (MAE_rf / np.mean(np.abs(y_test))) * 100
        RMSE_b = (RMSE_rf / np.mean(np.abs(y_test))) * 100

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

        models_name.append(f'{model_name}_boosting')
        RR.append(acc_rf)
        MAE.append(MAE_rf)
        MSE.append(MSE_rf)
        RMSE.append(RMSE_rf)
        MAE_B.append(MAE_b)
        RMSE_B.append(RMSE_b)

        # 保存模型
        os.makedirs(f'{model_save_path}/pkl/', exist_ok=True)
        joblib.dump(model, f'{model_save_path}/pkl/{model_name}_boosting.pkl')

    for model_name, model in regression_models.items():
        print(f"训练 {model_name} 模型")
        model.fit(X_train, y_train)
        evaluate_boosting_model(model, model_name)

    # 创建 DataFrame 保存评估结果
    df = pd.DataFrame({'模型名称': models_name, 'R^2': RR, 'MAE': MAE, 'MSE': MSE, 'RMSE': RMSE, '%MAE': MAE_B, '%RMSE': RMSE_B})

    # 根据 R^2 对模型进行排序
    df = df.sort_values(by='R^2', ascending=False)

    print(df)
    os.makedirs(os.path.dirname(comparison_save_path), exist_ok=True)
    df.to_csv(comparison_save_path, index=True, float_format='%.6f', encoding='gbk')
    print('boosting提升法集成模型训练完成操作')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Boosting ensemble learning for regression models.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file.')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save trained models.')
    parser.add_argument('--comparison_save_path', type=str, required=True, help='Path to save model comparison results.')
    parser.add_argument('--target', type=str, required=True, help='Target column name for regression.')
    parser.add_argument('--num', type=int, required=True, help='Number of boosting estimators.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to be used as test set.')
    parser.add_argument('--random_state', type=int, default=30, help='Random seed for reproducibility.')

    args = parser.parse_args()

    train_and_evaluate_models_boosting(
        data_path=args.data_path,
        model_save_path=args.model_save_path,
        comparison_save_path=args.comparison_save_path,
        num=args.num,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state
    )
