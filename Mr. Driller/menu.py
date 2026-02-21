import pygame
from os import path, makedirs # Aggiunto makedirs
import level

# --- GESTIONE PUNTEGGI ---
def storeScore(playerScore):
    # Creiamo la cartella se non esiste, per evitare crash
    folder_path = path.join("Assets", "Score")
    if not path.exists(folder_path):
        makedirs(folder_path)

    file_path = path.join(folder_path, "score.txt")

    # Apre o crea il file
    if path.isfile(file_path):
        scoreFile = open(file_path, "r+")
    else:
        scoreFile = open(file_path, "w+")

    # Gestione file vuoto
    scoreFile.seek(0)
    if not scoreFile.read(1):
        for i in range(100, 400, 100):
            scoreFile.write(str(i) + "\n")
    
    scoreFile.seek(0)
    lines = scoreFile.readlines()
    
    # Pulizia e conversione
    scores = []
    for line in lines:
        try:
            scores.append(int(line.strip()))
        except ValueError:
            pass

    scores.append(playerScore)
    scores.sort(reverse=True)
    
    # Mantieni solo i top 3
    scores = scores[:3]

    # Scrittura
    scoreFile.seek(0)
    scoreFile.truncate()
    for s in scores:
        scoreFile.write(str(s) + "\n")

    scoreFile.close()


def readScore(surface):
    file_path = path.join("Assets", "Score", "score.txt")
    
    # Se il file non esiste ancora, non disegnare nulla o disegna placeholder
    if not path.exists(file_path):
        return

    try:
        scoreFile = open(file_path, "r")
        FontUi = pygame.font.Font("Assets/Misc/police/Act_Of_Rejection.ttf", 48) # Controlla il nome del font!

        lines = scoreFile.readlines()
        lines.sort(reverse=True, key=lambda x: int(x.strip()) if x.strip().isdigit() else 0)
        
        dispQyt = min(len(lines), 3)
        
        for i in range(dispQyt):
            val = lines[i].strip()
            ttd = f"{i + 1} : {val}"
            scoreDisp = FontUi.render(ttd, 1, (220, 0, 255))
            surface.blit(scoreDisp, (i*240+30, 535))

        scoreFile.close()
    except Exception as e:
        print(f"Errore lettura score: {e}")


# --- MENU GRAFICO ---
def mainMenu(surface, optionIM):
    # Assicurati che i percorsi siano corretti
    try:
        bg = pygame.image.load(path.join("Assets", "Menu", "menu.png"))
        
        if optionIM == 1:
            playImg = pygame.image.load(path.join("Assets", "Buttons", "play_s.png"))
            quitImg = pygame.image.load(path.join("Assets", "Buttons", "quit_u.png"))
        elif optionIM == 2:
            playImg = pygame.image.load(path.join("Assets", "Buttons", "play_u.png"))
            quitImg = pygame.image.load(path.join("Assets", "Buttons", "quit_s.png"))

        surface.blit(bg, (0, 0))
        surface.blit(playImg, (310, 300))
        surface.blit(quitImg, (310, 400))
        readScore(surface)
        pygame.display.update()
    except FileNotFoundError as e:
        print(f"Errore caricamento asset menu: {e}")


def changeLvl(currentLvl, player, is_ai=False):
    if currentLvl < 10:   #PRIMA 10
        currentLvl += 1
        player.resetCoord(currentLvl)
        
        colors = 2 if currentLvl in [2, 7] else 4

        bg_name = f"{currentLvl}" 

        if currentLvl > 4:
            lvl = level.generateLvl(colors, 155, 7, bg_name, 5, 7, 20, 10, 10, 12)
        else:
            lvl = level.generateLvl(colors, 80, 7, bg_name)
        # -----------------------

        for row in lvl:
            for block in row:
                block.updCoText(lvl)

        return lvl, currentLvl, False

    else:
        # GIOCO FINITO (Livello 10 superato)
        if not is_ai: 
            storeScore(player.scoreAcc())
        
        game_completed = True
        return [], currentLvl, game_completed


def restart(player):
    # Riavvia dal livello 0 -> 1
    lvl, currentlvl, won = changeLvl(0, player, is_ai=True) # is_ai qui non importa perché lvl < 10
    player.resetScore()
    return lvl, currentlvl, won