import torch


class OmokAlgorithm():
    def __init__(self, player):
        self.player = player

    def __call__(self, state):
        board = [list(map(lambda x: int(x), state[0]))[10*i:10*i+10]
                 for i in range(10)]
        result = [[0 for i in range(10)] for i in range(10)]
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == self.player:
                    if j+1 < 10:
                        result[i][j+1] += 0.1
                    if j-1 >= 0:
                        result[i][j-1] += 0.1
                    if i+1 < 10:
                        result[i+1][j] += 0.1
                    if i-1 >= 0:
                        result[i-1][j] += 0.1
        # 가로 체크
        for i in range(len(board)):
            transarray = [[], []]
            for j in range(len(board[0])):
                if board[i][j] != 0:
                    transarray[board[i][j]-1].append(j)
            for eacharray in transarray:
                for t in eacharray:
                    if t+1 in eacharray and t+2 in eacharray:
                        if t-1 >= 0:
                            result[i][t-1] += 0.5
                        if t+3 < 10:
                            result[i][t+3] += 0.5
                    if t+1 in eacharray and t+2 in eacharray and\
                            t+3 in eacharray:
                        if t-1 >= 0:
                            result[i][t-1] += 2
                        if t+4 < 10:
                            result[i][t+4] += 2
        # 세로 체크
        for j in range(len(board[0])):
            longiarray = [[], []]
            for i in range(len(board[0])):
                if board[i][j] != 0:
                    longiarray[board[i][j]-1].append(i)
            for eacharray in longiarray:
                for t in eacharray:
                    if t+1 in eacharray and t+2 in eacharray:
                        if t-1 >= 0:
                            result[t-1][j] += 0.5
                        if t+3 < 10:
                            result[t+3][j] += 0.5
                    if t+1 in eacharray and t+2 in eacharray and\
                            t+3 in eacharray:
                        if t-1 >= 0:
                            result[t-1][j] += 2
                        if t+4 < 10:
                            result[t+4][j] += 2
        # 대각선 체크 (\ 방향)
        for i in range(len(board)):
            diagnarray = [[], []]
            for j in range(len(board[0])):
                if 0 <= i+j < len(board):
                    if board[i+j][j] != 0:
                        diagnarray[board[i+j][j]-1].append(j)
            for eacharray in diagnarray:
                for t in eacharray:
                    if t+1 in eacharray and t+2 in eacharray:
                        if t-1 >= 0:
                            result[i+t-1][t-1] += 0.5
                        if i+t+3 < 10:
                            result[i+t+3][t+3] += 0.5
                    if t+1 in eacharray and t+2 in eacharray and\
                            t+3 in eacharray:
                        if t-1 >= 0:
                            result[i+t-1][t-1] += 2
                        if i+t+4 < 10:
                            result[i+t+4][t+4] += 2
        for i in range(len(board)):
            diagnarray = [[], []]
            for j in range(len(board[0])):
                if 0 <= j-i < len(board):
                    if board[j-i][j] != 0:
                        diagnarray[board[j-i][j]-1].append(j)
            for eacharray in diagnarray:
                for t in eacharray:
                    if t+1 in eacharray and t+2 in eacharray:
                        if t-1-i >= 0:
                            result[t-1-i][t-1] += 0.5
                        if t+3 < 10:
                            result[t+3-i][t+3] += 0.5
                    if t+1 in eacharray and t+2 in eacharray and\
                            t+3 in eacharray:
                        if t-1-i >= 0:
                            result[t-1-i][t-1] += 2
                        if t+4 < 10:
                            result[t+4-i][t+4] += 2
        # 대각선 체크 (/ 방향)
        for i in range(len(board)):
            diagnarray = [[], []]
            for j in range(len(board[0])):
                if 0 <= i+j < len(board):
                    if board[i+j][-1-j] != 0:
                        diagnarray[board[i+j][-1-j]-1].append(j)
            for eacharray in diagnarray:
                for t in eacharray:
                    if t+1 in eacharray and t+2 in eacharray:
                        if t-1 >= 0:
                            result[i+t-1][-t] += 0.5
                        if i+t+3 < 10:
                            result[i+t+3][-t-4] += 0.5
                    if t+1 in eacharray and t+2 in eacharray and\
                            t+3 in eacharray:
                        if t-1 >= 0:
                            result[i+t-1][-t] += 2
                        if i+t+4 < 10:
                            result[i+t+4][-t-5] += 2
        for i in range(len(board)):
            diagnarray = [[], []]
            for j in range(len(board[0])):
                if 0 <= j-i < len(board):
                    if board[j-i][-1-j] != 0:
                        diagnarray[board[j-i][-1-j]-1].append(j)
            for eacharray in diagnarray:
                for t in eacharray:
                    if t+1 in eacharray and t+2 in eacharray:
                        if t-1-i >= 0:
                            result[t-1-i][-t] += 0.5
                        if t+3 < 10:
                            result[t+3-i][-t-4] += 0.5
                    if t+1 in eacharray and t+2 in eacharray and\
                            t+3 in eacharray:
                        if t-1-i >= 0:
                            result[t-1-i][-t] += 2
                        if t+4 < 10:
                            result[t+4-i][-t-5] += 2

        return torch.tensor(result).view(1, -1)
