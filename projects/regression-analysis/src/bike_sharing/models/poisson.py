"""Poisson 回归：用 log 链接建模非负计数目标的条件均值。"""

from sklearn.linear_model import PoissonRegressor

from bike_sharing.preprocessing import make_pipeline


def build_pipeline(parameters, seed=42):
    """预测均值为正，可为小数；不是分类，也不要求把预测取整。"""
    # alpha 是 L2 正则强度；与 Ridge 同名参数不能直接比较数值大小。
    # 对数链接没有自动引入特征交互，特征工程仍需单独设计实验。
    return make_pipeline(PoissonRegressor(**parameters), scale_numeric=True)


def main():
    from bike_sharing.train import main as train_main

    train_main(model_name="poisson")


if __name__ == "__main__":
    main()
