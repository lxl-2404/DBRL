#这个程序是为了重写stable_baseline3的callback（）函数，以此可以在训练开始、step、end时执行我们想要的动作。
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class SaveDecisionCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose=verbose)
        self.save_path = save_path
        self.data = []

    def _on_step(self) -> bool:
        obs = self.locals["new_obs"]
        action = self.locals["actions"]
        reward = self.locals["rewards"]
        done = self.locals["dones"]
        info = self.locals["infos"][0] # Access the first element of infos list.

        truncated = info.get("TimeLimit.truncated", False) # Extract truncated info, default to false if not found.

        self.data.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "done": done,
            "truncated": truncated
        })

        return True

    def _on_training_end(self) -> None:
      np.save(self.save_path, self.data)
      print(f"Decision data saved to {self.save_path}")
    
    def _on_training_end(self) -> None:
        print("Training begins----------")