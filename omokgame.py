import omokenv
from omok import DQN, select_action
import os
import torch

env = omokenv.Omokenv()
n_observations, n_actions = 100, 100

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net_1p, policy_net_2p = DQN(n_observations, n_actions).to(
    device), DQN(n_observations, n_actions).to(device)
if 'model_1p.pth' in os.listdir('./'):
    policy_net_1p.load_state_dict(
        torch.load('model_1p.pth', map_location=device))
    print('loaded_1p')
if 'model_2p.pth' in os.listdir('./'):
    policy_net_2p.load_state_dict(
        torch.load('model_2p.pth', map_location=device))
    print('loaded_2p')

state_1p = env.reset()
state_1p = torch.tensor(state_1p, dtype=torch.float32,
                        device=device).view(-1).unsqueeze(0)
while True:
    # player 1
    action_1p = tuple(
        [int(i) for i in (input('(x, y) 형태로 입력하세요')) if i.isnumeric()])
    observation_1p, reward_1p, terminated = env.step(action_1p)
    action_1p = torch.tensor(
        [[action_1p[0]*9+action_1p[1]]], device=device)
    # 다음 상태로 이동
    next_state_2p = torch.tensor(
        observation_1p, dtype=torch.float32, device=device).view(-1).unsqueeze(0)
    state_2p = next_state_2p
    env.render()
    if terminated:
        next_state_1p = None
        break

    else:
        # player 2
        action_2p = select_action(state_2p, policy_net_2p)
        observation_2p, reward_2p, terminated = env.step(action_2p)
        action_2p = torch.tensor(
            [[action_2p[0]*9+action_2p[1]]], device=device)
        # 다음 상태로 이동
        next_state_1p = torch.tensor(
            observation_2p, dtype=torch.float32, device=device).view(-1).unsqueeze(0)
    # 메모리에 변이 저장
    state_1p = next_state_1p
    env.render()
    if terminated:
        next_state_2p = None
        break
