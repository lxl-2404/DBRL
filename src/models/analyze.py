import numpy as np
# 加载数据
loaded_data = np.load("F:/lxl/28_dogfight/.dogfight/Lib/site-packages/jsbsim/DBRL/log/decision_data.npy", allow_pickle=True)

# 分析数据
for step_data in loaded_data:
    print(step_data["obs"])  # 观察
    print(step_data["action"])  # 动作
    print(step_data["reward"]) # 奖励
    print(step_data["done"])
    print(step_data["truncated"])
    # ... 其他分析 ...