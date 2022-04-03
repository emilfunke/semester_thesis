import turtle
import time
from screeninfo import get_monitors

mode = input("Please choose the ball moving mode: ")
# mode 1 --> outline screen
# mode 2 --> circle
# mode 3 --> cross
wait_time = 2

time.sleep(wait_time)

screen = turtle.Screen()
for m in get_monitors():
    height = m.height
    width = m.width
screen.setup(width=width, height=height)
screen.bgcolor('black')

ball = turtle.Turtle()
ball.color('yellow')
ball.pensize(10)
ball.speed(2)
ball.shapesize(3)
ball.shape('circle')


def ball_reset():
    ball.color('yellow')
    ball.speed(2)
    ball.shapesize(3)
    ball.shape('circle')


running = True
while running:
    screen_width = screen.window_width() * 0.9
    screen_height = screen.window_height() * 0.9

    # outline screen
    if mode == '1':
        ball.penup()
        ball.goto(-screen_width / 2, screen_height / 2)
        ball.forward(screen_width)
        ball.right(90)
        ball.forward(screen_height)
        ball.right(90)
        ball.forward(screen_width)
        ball.right(90)
        ball.forward(screen_height)
        mode = input("new mode or stop: ")
        ball_reset()

    # circle
    elif mode == '2':
        ball.penup()
        ball.goto(0, -screen_height / 2)
        ball.circle(screen_height / 2)
        mode = input("new mode or stop: ")
        ball_reset()

    # cross
    elif mode == '3':
        ball.penup()
        ball.goto(-screen_width / 2, screen_height / 2)
        ball.forward(screen_width)
        ball.goto(-screen_width / 2, -screen_height / 2)
        ball.forward(screen_width)
        ball.goto(0, 0)
        mode = input("new mode or stop: ")
        ball_reset()

    # stop
    elif mode == 'stop':
        running = False

    # wrong number and not stop
    else:
        mode = input("Please enter a number between within 1 and 3: ")
