"""决策树回归：通过条件分裂学习非线性和特征交互。"""

from sklearn.tree import DecisionTreeRegressor

from bike_sharing.preprocessing import make_pipeline


def build_pipeline(parameters, seed=42):
    """max_depth、min_samples_leaf 控制树复杂度和拟合程度。"""
    parameters = dict(parameters)
    parameters.setdefault("random_state", seed)
    # 树按阈值分裂，不需要标准化；仍复用相同的填充和类别独热编码。
    return make_pipeline(DecisionTreeRegressor(**parameters), scale_numeric=False)


def main():
    from bike_sharing.train import main as train_main

    train_main(model_name="decision_tree")


if __name__ == "__main__":
    main()
