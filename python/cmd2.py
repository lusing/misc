python train.py --algo a2c --env ALE/Pong-v5 --eval-episodes 10 --eval-freq 10000
python train.py --algo ppo --env ALE/Pong-v5 --eval-episodes 10 --eval-freq 10000

python train.py --algo a2c --env BreakoutNoFrameskip-v4 -i rl-trained-agents/a2c/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip -n 5000


D:\app\Python310\python.exe train.py --algo ppo --env ALE/Pong-v5 -i logs/ppo/ALE-Pong-v5_4/ALE-Pong-v5

noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
