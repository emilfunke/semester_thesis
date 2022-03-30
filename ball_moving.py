from tkinter import *
from screeninfo import get_monitors

class Ball:
    def __init__(self, canvas, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.canvas = canvas
        self.ball = canvas.create_oval(self.x1, self.y1, self.x2, self.y2, fill='yellow')

    def move_ball(self, delta_x, delta_y):
        self.canvas.move(self.ball, delta_x, delta_y)
        self.canvas.after(50, self.move_ball(delta_x, delta_y))

root = Tk()
root.title("Ball moving for eye tracking")
root.resizable(True, True)

for m in get_monitors():
    height = m.height
    width = m.width

canvas = Canvas(root, height=height, width=width, background='black')
canvas.pack()

ball = Ball(canvas, 10, 10, 20, 20)

ball.move_ball(5,5)



root.mainloop()
