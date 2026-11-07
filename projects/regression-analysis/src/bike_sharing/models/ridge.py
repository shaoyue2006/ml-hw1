"""Ridge 回归：平方误差 + L2 正则化，作为加性线性基线。"""

from sklearn.linear_model import Ridge

from bike_sharing.preprocessing import make_pipeline


def build_pipeline(parameters, seed=42):
    """alpha 控制系数收缩；当前 lsqr 求解器不使用随机种子。"""
    # 连续变量标准化，避免尺度差异使 L2 惩罚对不同特征不公平。
    # 小时等类别采用独热编码；不会自动构造“小时×工作日”交互。
    return make_pipeline(Ridge(**parameters), scale_numeric=True)


def main():
    """独立训练入口，复用公共流程保证数据划分和评价规则一致。"""
    from bike_sharing.train import main as train_main

    train_main(model_name="ridge")


if __name__ == "__main__":
    main()
