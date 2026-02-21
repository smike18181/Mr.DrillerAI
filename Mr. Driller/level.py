from random import randint
import block
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generateLvl(colors, lines, width, background, pillP=5, PillPL=7, PillMLE=20, SoloP=10, UnbreakableP=5, DelayedP=10):
    level = []
    LineRemainBeforePill = 0
    for i in range(lines + 5):
        line = []

        if i in range(5):   # Prime 5 righe vuote
            for j in range(width):
                newBlock = block.Classic(j, i, 1, 0)
                newBlock.changeBG(background)
                line.append(newBlock)

        elif i in range(lines):
            LineRemainBeforePill -= 1
            for j in range(width):

                PillRn        = randint(0, 100)
                UnbreakableRn = randint(0, 100)
                DelayedRn     = randint(0, 100)
                SoloRn        = randint(0, 100)

                if PillRn < pillP and PillPL != 0 and LineRemainBeforePill < 0:
                    PillPL -= 1
                    newBlock = block.Pill(j, i)
                    newBlock.changeBG(background)
                    line.append(newBlock)
                    LineRemainBeforePill = PillMLE

                elif SoloRn < SoloP:
                    newBlock = block.Solo(j, i)
                    newBlock.changeBG(background)
                    line.append(newBlock)

                elif UnbreakableRn < UnbreakableP:
                    newBlock = block.Unbreakable(j, i)
                    newBlock.changeBG(background)
                    line.append(newBlock)

                elif DelayedRn < DelayedP:
                    newBlock = block.Delayed(j, i)
                    newBlock.changeBG(background)
                    line.append(newBlock)

                else:
                    newBlock = block.Classic(j, i, randint(1, colors), 1)
                    newBlock.changeBG(background)
                    line.append(newBlock)

        else:
            for j in range(width):
                newBlock = block.End(j, i)
                newBlock.changeBG(background)
                line.append(newBlock)

        level.append(line)

    return level


def resetLevel(level):
    for row in level:
        for cell in row:
            cell.reset()


def render(surface, level, currOffset):
    """
    FIX: la versione originale chiamava element.display(surface) senza currOffset
    quando currOffset==0, causando il render a coordinate assolute sbagliate
    al cambio livello. Ora currOffset viene SEMPRE passato.
    Si disegnano le 9 righe visibili a partire da currOffset.
    """
    start = currOffset
    end   = min(currOffset + 9, len(level))   # protezione out-of-bounds

    for i in range(start, end):
        for element in level[i]:
            element.display(surface, currOffset)