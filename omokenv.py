# GUI
from tkinter import *
from PIL import ImageTk, Image

tk = Tk()
tk.title('오목')
images = [ImageTk.PhotoImage(Image.open("unit_default.png")), ImageTk.PhotoImage(
    Image.open("unit_black.png")), ImageTk.PhotoImage(Image.open("unit_white.png"))]
units = []


class OmokUnit():
    def __init__(self, loc):
        self.x, self.y = loc
        self.unit = Button(tk, image=images[0], command=self.click)
        self.unit.grid(row=self.y, column=self.x)

    def click(self):
        from omokgame import play
        play((self.x, self.y))


for i in range(10):
    for j in range(10):
        unit = OmokUnit((j, i))  # (x, y)
        units.append(unit)
message_label = Label(
    tk, text=f'player {1} 차례, {1:>3d}번째 턴')
message_label.grid(column=i+1)


# ENV
class Omokenv():
    def init(self):
        self.turn = int
        self.player = int
        self.board = []
        self.reward = int
        self.neighbor = int
        self.three = []
        self.four = []
        self.five = bool
        self.done = bool
        self.draw = bool

    def reset(self):
        self.turn = 1
        self.player = 1  # 1:흑, 2:백
        self.board = [[0 for i in range(10)] for i in range(10)]
        self.reward = 0
        self.neighbor = 0
        self.three = [0, 0]
        self.four = [0, 0]
        self.five = False
        self.done = False
        self.draw = False
        return self.board

    def step(self, loc):
        x, y = loc
        self.board[y][x] = self.player
        self.check_game_status()
        self.player = 1+self.player % 2
        self.turn += 1
        return self.board, self.reward, self.done

    def check_game_status(self):
        # 지도학습용 파라미터
        neighbor = 0
        # three = 0
        four = 0

        # 무승부 체크
        self.draw = True
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == 0:
                    self.draw = False

        # 가로 체크
        for i in range(len(self.board)):
            transarray = []
            neighborarray = []
            for j in range(len(self.board[0])):
                if self.board[i][j] == self.player:
                    transarray.append(j)
                if self.board[i][j] != 0:
                    neighborarray.append(j)
            for t in neighborarray:
                if t+1 in neighborarray:
                    neighbor += 1
            for t in transarray:
                # if t+1 in transarray and t+2 in transarray:
                #     three += 1
                if t+1 in transarray and t+2 in transarray and\
                        t+3 in transarray:
                    four += 1
                if t+1 in transarray and t+2 in transarray and\
                        t+3 in transarray and t+4 in transarray:
                    self.five = True
        # 세로 체크
        for j in range(len(self.board[0])):
            longiarray = []
            for i in range(len(self.board)):
                if self.board[i][j] == self.player:
                    longiarray.append(i)
                if self.board[i][j] != 0:
                    neighborarray.append(i)
            for t in neighborarray:
                if t+1 in neighborarray:
                    neighbor += 1
            for t in longiarray:
                # if t+1 in longiarray and t+2 in longiarray:
                #     three += 1
                if t+1 in longiarray and t+2 in longiarray and\
                        t+3 in longiarray:
                    four += 1
                if t+1 in longiarray and t+2 in longiarray and\
                        t+3 in longiarray and t+4 in longiarray:
                    self.five = True
        # 대각선 체크 (\ 방향)
        for i in range(len(self.board)):
            diagnarray = []
            for j in range(len(self.board[0])):
                if 0 <= i+j < len(self.board):
                    if self.board[i+j][j] == self.player:
                        diagnarray.append(j)
                    if self.board[i+j][j] != 0:
                        neighborarray.append(j)
            for t in neighborarray:
                if t+1 in neighborarray:
                    neighbor += 1
            for t in diagnarray:
                # if t+1 in diagnarray and t+2 in diagnarray:
                #     three += 1
                if t+1 in diagnarray and t+2 in diagnarray and\
                        t+3 in diagnarray:
                    four += 1
                if t+1 in diagnarray and t+2 in diagnarray and\
                        t+3 in diagnarray and t+4 in diagnarray:
                    self.five = True
        for i in range(len(self.board)):
            diagnarray = []
            for j in range(len(self.board[0])):
                if 0 <= j-i < len(self.board):
                    if self.board[j-i][j] == self.player:
                        diagnarray.append(j)
                    if self.board[j-i][j] != 0:
                        neighborarray.append(j)
            for t in neighborarray:
                if t+1 in neighborarray:
                    neighbor += 1
            for t in diagnarray:
                # if t+1 in diagnarray and t+2 in diagnarray:
                #     three += 1
                if t+1 in diagnarray and t+2 in diagnarray and\
                        t+3 in diagnarray:
                    four += 1
                if t+1 in diagnarray and t+2 in diagnarray and\
                        t+3 in diagnarray and t+4 in diagnarray:
                    self.five = True
        # 대각선 체크 (/ 방향)
        for i in range(len(self.board)):
            diagnarray = []
            for j in range(len(self.board[0])):
                if 0 <= i+j < len(self.board):
                    if self.board[i+j][-1-j] == self.player:
                        diagnarray.append(j)
                    if self.board[i+j][-1-j] != 0:
                        neighborarray.append(j)
            for t in neighborarray:
                if t+1 in neighborarray:
                    neighbor += 1
            for t in diagnarray:
                # if t+1 in diagnarray and t+2 in diagnarray:
                #     three += 1
                if t+1 in diagnarray and t+2 in diagnarray and\
                        t+3 in diagnarray:
                    four += 1
                if t+1 in diagnarray and t+2 in diagnarray and\
                        t+3 in diagnarray and t+4 in diagnarray:
                    self.five = True
        for i in range(len(self.board)):
            diagnarray = []
            for j in range(len(self.board[0])):
                if 0 <= j-i < len(self.board):
                    if self.board[j-i][-1-j] == self.player:
                        diagnarray.append(j)
                    if self.board[j-i][-1-j] != 0:
                        neighborarray.append(j)
            for t in neighborarray:
                if t+1 in neighborarray:
                    neighbor += 1
            for t in diagnarray:
                # if t+1 in diagnarray and t+2 in diagnarray:
                #     three += 1
                if t+1 in diagnarray and t+2 in diagnarray and\
                        t+3 in diagnarray:
                    four += 1
                if t+1 in diagnarray and t+2 in diagnarray and\
                        t+3 in diagnarray and t+4 in diagnarray:
                    self.five = True
        if self.five == True:
            self.reward = 2
        else:
            self.reward = 0
            if neighbor > self.neighbor:
                # self.reward += 0.01*(neighbor-self.neighbor)
                self.neighbor = neighbor
            else:
                self.reward -= 0.1
            # if three > self.three[self.player-1]:
            #     self.reward += 0.1 * \
            #         (three-self.three[self.player-1])
            #     self.three[self.player-1] = three
            if four > self.four[self.player-1]:
                self.reward += 0.1 * \
                    (four-self.four[self.player-1])
                self.four[self.player-1] = four
        self.done = self.five or self.draw

    def render(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                units[i*10+j].unit['image'] = images[self.board[i][j]]
        message_label['text'] = f'player {self.player} 차례, {self.turn:>3d}번째 턴'
        tk.update()
