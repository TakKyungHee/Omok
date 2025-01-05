import omokenv
import omok
import os
import torch
from itertools import count


def play(loc=None):
    global done, states
    if not done:
        player = 1-env.turn % 2  # player 1 -> 0, player 2 -> 1
        # player는 턴의 홀짝으로 구분
        if playables[player] == True:
            if loc == None or len(loc) != 2 or env.board[loc[1]][loc[0]] != 0:
                return
            else:
                action = loc
        else:
            action = omok.select_action(
                states[player], policys[player], eval=True)
        observation, reward, terminated = env.step(action)
        # 다음 상태로 이동
        player = 1-env.turn % 2
        next_state = None if terminated else torch.tensor(
            observation, dtype=torch.float32, device=omok.device).view(-1).unsqueeze(0)
        states[player] = next_state
        if terminated:  # player 승리
            done = True
            print(f'Game Over, player {2-player%2} win!')
        env.render()
        if not done and playables[player] == False:
            play()


env = omok.env

save_states = []  # 상호 참조 방지

policys = [omok.policy_net_1p.eval(), omok.policy_net_2p]

if len(save_states) > 0:
    state_1p, state_2p = save_states[0], save_states[1]
else:
    state_1p, state_2p = env.reset(), env.reset()
state_1p, state_2p = torch.tensor(state_1p, dtype=torch.float32,
                                  device=omok.device).view(-1).unsqueeze(0), torch.tensor(state_2p, dtype=torch.float32,
                                                                                          device=omok.device).view(-1).unsqueeze(0)
states = [state_1p, state_2p]

# playables = [True, False]
playables = [False, True]

done = False

if __name__ == '__main__':
    play()
    save_states.append(state_1p)
    save_states.append(state_2p)
    omokenv.tk.mainloop()
