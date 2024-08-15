#导⼊相关包
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
# 拆分预测数据和训练数据
from sklearn.model_selection import train_test_split
#缺失值处理
from sklearn.impute import SimpleImputer
#聚类
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# 模型
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
#其他模型
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, Lars, LassoLars, PassiveAggressiveRegressor, TheilSenRegressor, HuberRegressor, OrthogonalMatchingPursuit, TweedieRegressor, PoissonRegressor, GammaRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, HistGradientBoostingRegressor
from sklearn.svm import LinearSVR, SVR, NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.mixture import BayesianGaussianMixture
#随机寻参
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
#交叉验证
from sklearn.model_selection import KFold 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
#模型评估
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
# 绘图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import seaborn as sns
sns.set(style='white',context='notebook',palette='muted',font='SimHei')  # 解决Seaborn中文显示问题并调整字体大小) # 设置sns样式

# 可视化树形图
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

# 预测情况绘制地图
import geopandas
import shapefile  # 使用pyshp
import numpy as np
from random import random
from pathlib import Path
import os
import matplotlib as mpl
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import StandardScaler
import argparse
from scipy import stats
from scipy.stats import norm, skew #for some statistics
    
os.environ["PATH"] += os.pathsep + './graphviz/bin/'
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

    


class ModelTrainer:
    def __init__(self, data_path, 
                 model_save_path, 
                 comparison_save_path, 
                 target, 
                 test_size=0.2, 
                 random_state=42, 
                 model_list=None):
        
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.comparison_save_path = comparison_save_path
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        
        # Define all available models
        self.all_models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=45),
            "Lasso": Lasso(random_state=45),
            "ElasticNet": ElasticNet(random_state=45),
            "BayesianRidge": BayesianRidge(),
            "LassoLars": LassoLars(random_state=45),
            "PassiveAggressiveRegressor": PassiveAggressiveRegressor(random_state=45),
            "TheilSenRegressor": TheilSenRegressor(random_state=45),
            "HuberRegressor": HuberRegressor(),
            "OrthogonalMatchingPursuit": OrthogonalMatchingPursuit(),
            "TweedieRegressor": TweedieRegressor(),
            "PoissonRegressor": PoissonRegressor(),
            "GammaRegressor": GammaRegressor(),
            "RandomForestRegressor": RandomForestRegressor(random_state=45),
            "ExtraTreesRegressor": ExtraTreesRegressor(random_state=45),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=45),
            "XGBRegressor": XGBRegressor(),
            "lgb.LGBMRegressor": lgb.LGBMRegressor(random_state=45),
            "AdaBoostRegressor": AdaBoostRegressor(random_state=45),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=45),
            "LinearSVR": LinearSVR(random_state=45),
            "SVR": SVR(),
            "NuSVR": NuSVR(),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=45),
            "ExtraTreeRegressor": ExtraTreeRegressor(random_state=45),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "GaussianProcessRegressor": GaussianProcessRegressor(random_state=45),
    }
        
        # Select models based on model_list, or use all models if None
        if model_list is None:
            self.models = self.all_models
        else:
            self.models = {name: self.all_models[name] for name in model_list if name in self.all_models}
    
    def load_data(self):
        self.data = pd.read_csv(self.data_path, encoding="utf-8")
        print("数据导入成功")
        print("原始数据的前几行：")
        print(self.data.head())
    
    def preprocess_data(self):
        # 将数据读取并设置 'pointID' 为索引
        self.data = pd.read_csv(self.data_path, encoding="utf-8", index_col='pointID')
        print("数据导入成功")

        # 删除不需要的列，'pointID' 已经作为索引，不需要再次删除
        X = self.data.drop([self.target], axis=1)
        y = self.data[self.target]

        # 数据标准化
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X)
        X_standardized = pd.DataFrame(X_standardized, columns=X.columns, index=self.data.index)
        
        # 数据归一化
        normalizer = MinMaxScaler()
        X_normalized = normalizer.fit_transform(X_standardized)
        X_normalized = pd.DataFrame(X_normalized, columns=X.columns, index=self.data.index)

        print("标准化后的特征数据的前几行：")
        print(X_standardized.head())
        print("归一化后的特征数据的前几行：")
        print(X_normalized.head())

        # 分割数据集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_normalized, y, test_size=self.test_size, random_state=self.random_state
        )

    
    # def train_and_evaluate(self):
    #     # Define paths for saving CSV and PKL files
    #     csv_save_path = os.path.join(self.model_save_path, "csv")
    #     pkl_save_path = os.path.join(self.model_save_path, "pkl")

    #     # Create directories if they don't exist
    #     os.makedirs(csv_save_path, exist_ok=True)
    #     os.makedirs(pkl_save_path, exist_ok=True)
    #     os.makedirs(os.path.dirname(self.comparison_save_path), exist_ok=True)

    #     results = []
    #     for model_name, model in self.models.items():
    #         model.fit(self.X_train, self.y_train)
    #         y_train_pred = model.predict(self.X_train)
    #         y_test_pred = model.predict(self.X_test)

    #         # 创建一个结果数据框，其中 'pointID' 已经作为索引，不再需要从 self.data 中提取
    #         res = pd.DataFrame({
    #             'pointID': self.X_test.index,  # 使用索引作为 pointID
    #             '真实值': self.y_test,
    #             '预测值': y_test_pred,
    #             '差值': abs(self.y_test - y_test_pred)
    #         })

    #          # 保存模型为csv文件
    #         res.to_csv(os.path.join(csv_save_path, f'{model_name}.csv'), index=False, float_format='%.2f', encoding='gbk')
            
    #         # 保存模型为pkl文件
    #         joblib.dump(model, os.path.join(pkl_save_path, f'{model_name}.pkl'))

    #         # 模型评估
    #         acc = metrics.r2_score(self.y_test, y_test_pred)
    #         adjr2 = 1 - (1 - acc) * (len(self.y_test) - 1) / (len(self.y_test) - self.X_test.shape[1] - 1)
    #         mae = metrics.mean_absolute_error(self.y_test, y_test_pred)
    #         mse = metrics.mean_squared_error(self.y_test, y_test_pred)
    #         rmse = np.sqrt(mse)
    #         mae_b = (mae / np.mean(np.abs(self.y_test))) * 100
    #         rmse_b = (rmse / np.mean(np.abs(self.y_test))) * 100

    #         # 打印每个模型的评估结果
    #         bf_print(
    #             model_name=model_name,
    #             acc_rf=acc,
    #             adjr2_rf=adjr2,
    #             MAE_rf=mae,
    #             MSE_rf=mse,
    #             RMSE_rf=rmse,
    #             MAE_b=mae_b,
    #             RMSE_b=rmse_b
    #         )

    #         results.append({
    #             'Model': model_name,
    #             'R^2': acc,
    #             'Adjusted R^2': adjr2,
    #             'MAE': mae,
    #             'MSE': mse,
    #             'RMSE': rmse,
    #             'MAE (%)': mae_b,
    #             'RMSE (%)': rmse_b
    #         })

    #     # 保存评估结果
    #     pd.DataFrame(results).to_csv(self.comparison_save_path, index=False, float_format='%.4f', encoding='gbk')
    #     print(f"模型评估结果已保存到 {self.comparison_save_path}")
    def train_and_evaluate(self):
        # Define paths for saving CSV and PKL files
        csv_save_path = os.path.join(self.model_save_path, "csv")
        pkl_save_path = os.path.join(self.model_save_path, "pkl")

        # Create directories if they don't exist
        os.makedirs(csv_save_path, exist_ok=True)
        os.makedirs(pkl_save_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.comparison_save_path), exist_ok=True)

        results = []
        for model_name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)

            # 创建一个结果数据框，其中 'pointID' 已经作为索引，不再需要从 self.data 中提取
            res = pd.DataFrame({
                'pointID': self.X_test.index,  # 使用索引作为 pointID
                '真实值': self.y_test,
                '预测值': y_test_pred,
                '差值': abs(self.y_test - y_test_pred)
            })

            # 保存模型为csv文件
            res.to_csv(os.path.join(csv_save_path, f'{model_name}.csv'), index=False, float_format='%.2f', encoding='gbk')
            
            # 保存模型为pkl文件
            joblib.dump(model, os.path.join(pkl_save_path, f'{model_name}.pkl'))

            # 模型评估
            acc = metrics.r2_score(self.y_test, y_test_pred)
            adjr2 = 1 - (1 - acc) * (len(self.y_test) - 1) / (len(self.y_test) - self.X_test.shape[1] - 1)
            mae = metrics.mean_absolute_error(self.y_test, y_test_pred)
            mse = metrics.mean_squared_error(self.y_test, y_test_pred)
            rmse = np.sqrt(mse)
            mae_b = (mae / np.mean(np.abs(self.y_test))) * 100
            rmse_b = (rmse / np.mean(np.abs(self.y_test))) * 100

            results.append({
                'Model': model_name,
                'R^2': acc,
                'Adjusted R^2': adjr2,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAE (%)': mae_b,
                'RMSE (%)': rmse_b
            })

        # 对结果按照 R^2 进行降序排列
        results = sorted(results, key=lambda x: x['R^2'], reverse=True)

        # 打印排序后的评估结果
        for result in results:
            bf_print(
                model_name=result['Model'],
                acc_rf=result['R^2'],
                adjr2_rf=result['Adjusted R^2'],
                MAE_rf=result['MAE'],
                MSE_rf=result['MSE'],
                RMSE_rf=result['RMSE'],
                MAE_b=result['MAE (%)'],
                RMSE_b=result['RMSE (%)']
            )

        # 保存排序后的评估结果
        pd.DataFrame(results).to_csv(self.comparison_save_path, index=False, float_format='%.4f', encoding='gbk')
        print(f"模型评估结果已保存到 {self.comparison_save_path}")



def main():
    parser = argparse.ArgumentParser(description="Train and evaluate regression models.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data CSV file.")
    parser.add_argument("--model_save_path", type=str, required=True, help="Directory to save trained model predictions.")
    parser.add_argument("--comparison_save_path", type=str, required=True, help="File to save the comparison results.")
    parser.add_argument("--target", type=str, required=True, help="Target column name in the dataset.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the data to be used as test set.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument("--model_list", type=str, nargs='*', default=None, help="List of model names to train, e.g. 'LinearRegression Ridge'.")

    args = parser.parse_args()
    
    model_trainer = ModelTrainer(
        data_path=args.data_path,
        model_save_path=args.model_save_path,
        comparison_save_path=args.comparison_save_path,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        model_list=args.model_list
    )
    
    model_trainer.load_data()
    model_trainer.preprocess_data()
    model_trainer.train_and_evaluate()

if __name__ == "__main__":
    main()
