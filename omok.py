import omokenv
import os
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, n_actions)

    # 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호촐됩니다.
    # ([[left0exp,right0exp]...]) 를 반환합니다.
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
# BATCH_SIZE는 리플레이 버퍼에서 샘플링된 트랜지션의 수입니다.
# GAMMA는 이전 섹션에서 언급한 할인 계수입니다.
# EPS_START는 엡실론의 시작 값입니다.
# EPS_END는 엡실론의 최종 값입니다.
# EPS_DECAY는 엡실론의 지수 감쇠(exponential decay) 속도 제어하며, 높을수록 감쇠 속도가 느립니다.
# TAU는 목표 네트워크의 업데이트 속도입니다.
# LR은 ``AdamW`` 옵티마이저의 학습율(learning rate)입니다.
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.01
LR = 1e-2
STEPS_DONE = 0

env = omokenv.Omokenv()
n_observations, n_actions = 100, 100

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
target_net_1p, target_net_2p = DQN(n_observations, n_actions).to(
    device), DQN(n_observations, n_actions).to(device)
target_net_1p.load_state_dict(policy_net_1p.state_dict())
target_net_2p.load_state_dict(policy_net_2p.state_dict())

optimizer_1p, optimizer_2p = optim.Adam(policy_net_1p.parameters(
), lr=LR, amsgrad=True), optim.Adam(policy_net_2p.parameters(), lr=LR, amsgrad=True)
memory_1p, memory_2p = ReplayMemory(100000), ReplayMemory(100000)

average_reward = []
winning_rate = []

win = 0
rewards = 0


def select_action(state, policy_net, eval=False):
    global STEPS_DONE
    empty_loc = []
    for i in range(len(state[0])):
        if state[0][i] == 0:
            empty_loc.append(i)
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * STEPS_DONE / EPS_DECAY)
    action = random.choice(empty_loc)
    if sample > eps_threshold or eval == True:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            result = policy_net(state)
            action = torch.argmax(result[0][torch.tensor([empty_loc])])
    loc = (action % 10, action//10)
    STEPS_DONE += 1
    return loc


def plot_durations(show_result=False):
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.plot(winning_rate, label='Winning rate')
    plt.plot(average_reward, label='average reward')

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
    # 전환합니다.
    batch = Transition(*zip(*transitions))

    # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
    # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    state_action_values = policy_net(
        state_batch).gather(1, action_batch)

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1)[0]으로 최고의 보상을 선택하십시오.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1)[0]
    # 기대 Q 값 계산
    expected_state_action_values = (
        next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()

    # 변화도 클리핑 바꿔치기
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if __name__ == '__main__':
    if torch.cuda.is_available():
        num_episodes = 400
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
        # 환경과 상태 초기화
        terminated, done = False, False
        state_1p = env.reset()
        state_1p = torch.tensor(state_1p, dtype=torch.float32,
                                device=device).view(-1).unsqueeze(0)
        for t in count():
            if terminated:  # player 2 승리
                done = True
                next_state_2p = None
            else:
                # player 1
                action_1p = select_action(state_1p, policy_net_1p)
                observation_1p, reward_1p, terminated = env.step(action_1p)
                action_1p = torch.tensor(
                    [[action_1p[0]*9+action_1p[1]]], device=device)
                # 다음 상태로 이동
                next_state_2p = None if terminated else torch.tensor(
                    observation_1p, dtype=torch.float32, device=device).view(-1).unsqueeze(0)
            if t > 0:  # player 2는 3번째 턴부터 계산
                reward_2p = reward_2p-reward_1p
                if torch.is_tensor(reward_2p):
                    rewards += reward_2p.item()
                else:
                    rewards += reward_2p  # 2p만 리워드 기록
                reward_2p = torch.tensor([reward_2p], device=device)
                # 메모리에 변이 저장
                memory_2p.push(state_2p, action_2p,
                               next_state_2p, reward_2p)
            state_2p = next_state_2p
            if t > 0:
                # (정책 네트워크에서) 최적화 한단계 수행
                optimize_model(memory_2p, policy_net_2p,
                               target_net_2p, optimizer_2p)
                # 목표 네트워크의 가중치를 소프트 업데이트
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_2p_state_dict = target_net_2p.state_dict()
                policy_net_2p_state_dict = policy_net_2p.state_dict()
                for key in policy_net_2p_state_dict:
                    target_net_2p_state_dict[key] = policy_net_2p_state_dict[key] * \
                        TAU + target_net_2p_state_dict[key]*(1-TAU)
                target_net_2p.load_state_dict(target_net_2p_state_dict)

            if not done:  # player 2 승리 시 한 번 더 반복되지 않기 위해
                if terminated:  # player 1 승리
                    done = True
                    next_state_1p = None
                else:
                    # player 2
                    action_2p = select_action(state_2p, policy_net_2p)
                    observation_2p, reward_2p, terminated = env.step(action_2p)
                    action_2p = torch.tensor(
                        [[action_2p[0]*9+action_2p[1]]], device=device)
                    # 다음 상태로 이동
                    next_state_1p = None if terminated else torch.tensor(
                        observation_2p, dtype=torch.float32, device=device).view(-1).unsqueeze(0)
                reward_1p = reward_1p-reward_2p
                # if torch.is_tensor(reward_1p):
                #     rewards += reward_1p.item()
                # else:
                #     rewards += reward_1p # 1p만 리워드 기록
                reward_1p = torch.tensor([reward_1p], device=device)
                # 메모리에 변이 저장
                memory_1p.push(state_1p, action_1p,
                               next_state_1p, reward_1p)
                state_1p = next_state_1p
                # (정책 네트워크에서) 최적화 한단계 수행
                optimize_model(memory_1p, policy_net_1p,
                               target_net_1p, optimizer_1p)
                # 목표 네트워크의 가중치를 소프트 업데이트
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_1p_state_dict = target_net_1p.state_dict()
                policy_net_1p_state_dict = policy_net_1p.state_dict()
                for key in policy_net_1p_state_dict:
                    target_net_1p_state_dict[key] = policy_net_1p_state_dict[key] * \
                        TAU + target_net_1p_state_dict[key]*(1-TAU)
                target_net_1p.load_state_dict(target_net_1p_state_dict)

            if done:
                win += 2-env.player
                average_reward.append(rewards/(i_episode+1))
                winning_rate.append(win/(i_episode+1))
                plot_durations()
                break

    print('Complete',
          f'player 1의 승률 {winning_rate[-1]:.2%}')
    print(f'player 1의 평균 reward {average_reward[-1]:.3f}')
    torch.save(policy_net_1p.state_dict(), 'model_1p.pth')
    torch.save(policy_net_2p.state_dict(), 'model_2p.pth')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()
