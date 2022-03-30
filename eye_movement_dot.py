#idea found on https://www.geeksforgeeks.org/draw-moving-object-using-turtle-in-python/

import turtle
from screeninfo import get_monitors


def moving_obj(move):
    move.fillcolor('yellow')
    move.begin_fill()
    move.circle(25)
    move.end_fill()


if __name__ == "__main__":
    screen = turtle.Screen()
    for m in get_monitors():
        height = m.height
        width = m.width
    print(height, width)
    screen.setup(width, height)
    screen.bgcolor('black')
    screen.tracer(0)
    move = turtle.Turtle()
    move.color('yellow')
    move.speed(10)
    move.width(5)
    move.hideturtle()
    move.penup()
    move.goto(-width/2 + 30, height/2 -80)
    move.pendown()



while True:
    move.clear()
    moving_obj(move)
    screen.update()
    print(move.pos())
    move.forward(50)
    move.speed(5)

