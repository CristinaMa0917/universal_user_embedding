# -*- coding: utf-8 -*-#
from collections import namedtuple


# 训练配置
TrainConfigs = namedtuple("TrainConfigs", (
    # 学习率
    "learning_rate",
    # Batch大小
    "batch_size",
    "dropout_rate"
))

# 预测配置项
PredictConfigs = namedtuple("PredictConfigs", (
    # 是否输出分类前的特征向量
    "separator",
))

# 模型配置
RunConfigs = namedtuple("RunConfigs", (
    # 输出日志间隔
    "log_every"
))