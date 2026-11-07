"""直方图梯度提升树回归：逐步增加树来优化预测损失，不是深度学习。"""

from sklearn.ensemble import HistGradientBoostingRegressor

from bike_sharing.preprocessing import make_pipeline


def build_pipeline(parameters, seed=42):
    """学习率、迭代次数、叶节点数和 L2 共同控制模型复杂度。"""
    parameters = dict(parameters)
    # 自动早停可能随机划出验证子集，与本项目的时间验证不一致。
    # B 阶段固定轮数；C 若要早停，应显式构造靠后的时间验证段。
    if parameters.get("early_stopping", False) is not False:
        raise ValueError("Baseline disables random early stopping; use time validation in C.")
    parameters["early_stopping"] = False
    parameters.setdefault("random_state", seed)
    # 此估计器需要稠密输入；类别编码含义与其他模型保持一致。
    return make_pipeline(
        HistGradientBoostingRegressor(**parameters), scale_numeric=False, dense=True
    )


def main():
    from bike_sharing.train import main as train_main

    train_main(model_name="gradient_boosting")


if __name__ == "__main__":
    main()
