```bash
[ERROR] [1744958530.098430, 1232.099000]: bad callback: <bound method PPOSkillSelector.state_callback of <__main__.PPOSkillSelector object at 0x7f79f08a7cd0>>
Traceback (most recent call last):
  File "/opt/ros/melodic/lib/python2.7/dist-packages/rospy/topics.py", line 750, in _invoke_callback
    cb(msg)
  File "src/ppo-indoor-nav/scripts/ppo_skill_selector_py3.py", line 107, in state_callback
    self.process_state()
  File "src/ppo-indoor-nav/scripts/ppo_skill_selector_py3.py", line 139, in process_state
    self.update_policy()
  File "src/ppo-indoor-nav/scripts/ppo_skill_selector_py3.py", line 261, in update_policy
    critic_loss.backward()
  File "/home/lyb/.conda/envs/scene/lib/python3.8/site-packages/torch/_tensor.py", line 492, in backward
    torch.autograd.backward(
  File "/home/lyb/.conda/envs/scene/lib/python3.8/site-packages/torch/autograd/__init__.py", line 251, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: Found dtype Double but expected Float


```
