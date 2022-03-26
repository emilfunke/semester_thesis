from tkinter import *
from screeninfo import get_monitors

root = Tk()
root.title("Ball moving for eye tracking")
root.resizable(True, True)

for m in get_monitors():
    height = m.height
    width = m.width

canvas = Canvas(root, height=height, width=width, background='black')
canvas.pack()

canvas.create_oval(10,10,100,100, fill='yellow')



root.mainloop()
