import pygame
from screeninfo import get_monitors

for m in get_monitors():
    height = m.height
    width = m.width

pygame.init()
window = pygame.display.set_mode((width - 5, height - 100))
pygame.display.set_caption("Ball moving")

#global
ballx = 50
bally = 50
ballW = 30
ballH = 30

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.draw.circle(window, 'yellow', (ballx, bally), ballW, 0)
    pygame.display.flip()


pygame.display.quit()
pygame.quit()