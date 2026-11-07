"""模型注册入口：保留 build_model 接口，具体算法各自独立实现。"""

from importlib import import_module

MODEL_NAMES = ("ridge", "poisson", "decision_tree", "gradient_boosting")


def build_model(name: str, parameters: dict, seed: int = 42):
    """返回未拟合的 Pipeline，C 可使用 model__ 参数进行搜索。"""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    # 延迟导入使各算法可以独立 python -m 执行，避免提前加载对应模块。
    module = import_module(f"{__name__}.{name}")
    return module.build_pipeline(dict(parameters), seed)
