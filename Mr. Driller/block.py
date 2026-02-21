from connectCorrect import *
from os import path
import pygame
from menu import *
import random

# ─────────────────────────────────────────────────────────────────────
# CACHE TEXTURE
# Carica ogni immagine una sola volta per ridurre le letture disco.
# clear_cache() deve essere chiamato dal main ad ogni cambio livello
# per evitare che sfondi vecchi rimangano in memoria.
# ─────────────────────────────────────────────────────────────────────
_texture_cache = {}

def load_cached(filepath):
    if filepath not in _texture_cache:
        try:
            _texture_cache[filepath] = pygame.image.load(filepath).convert_alpha()
        except Exception:
            surf = pygame.Surface((64, 64), pygame.SRCALPHA)
            surf.fill((200, 0, 200, 180))
            _texture_cache[filepath] = surf
    return _texture_cache[filepath]

def clear_cache():
    """Svuota la cache. Chiamare ad ogni cambio livello."""
    _texture_cache.clear()


class Block:
    """Classe madre di tutti i blocchi."""

    def __init__(self, posX, posY, forceHP, chain_reaction, colors=0):
        self._posX       = posX
        self._posY       = posY
        self._currOffset = 0   # Posizione della telecamera (scrolling) rispetto alla griglia di gioco.
        self._hp         = forceHP
        self._blockType  = "neutral"

        # ── Fisica caduta ─────────────────────────────────────────────
        # Sistema semplice come l'originale:
        # _shakeTicks: quante volte è stato chiamato applyGravity da quando trema.
        #              Quando raggiunge SHAKE_TICKS_BEFORE_FALL → cade.
        # _isFalling:  True = il blocco scende di una cella ad ogni chiamata
        #              di applyGravity (senza ritardi extra).
        # Nessun pygame.time.get_ticks(): il timing dipende solo dalla
        # frequenza con cui il main chiama applyGravity (timer EVTIC_GRAVITY).
        self._isShaking        = False
        self._shakeTicks       = 0
        self._isFalling        = False
        # Numero di chiamate ad applyGravity prima di passare da shake a fall.
        # Con EVTIC_GRAVITY a 400ms × GAME_SPEED, SHAKE_TICKS=5 = 2s di tremore.
        self.SHAKE_TICKS_BEFORE_FALL = 5

        if chain_reaction:
            self._colors = colors
        self._chain_reaction = chain_reaction

        self._texturePath = path.join("Assets", "Textures", "Blocks", "Neutral", "b_s.png")
        self._bg          = path.join("Assets", "Textures", "Background", "bg_1.png")

        self._brkSound = pygame.mixer.Sound(path.join("Assets", "Sounds", "pop.wav"))
        self._brkSound.set_volume(0.3)

    # ── Accessors ─────────────────────────────────────────────────────
    def hpAccess(self):    return self._hp
    def ColorAccess(self): return self._colors
    def posAcc(self):      return self._posY, self._posX
    def typeAccess(self):  return self._blockType

    # ── Logica ────────────────────────────────────────────────────────
    def changeBG(self, bg):
        self._bg = path.join("Assets", "Textures", "Background",
                             "bg_" + str(bg) + ".png")

    def updOffset(self, currentOffset):
        self._currOffset = currentOffset

    def hit(self, surface, level, player, nochain=0, instakill=0, delayedTimeout=0):
        if instakill:
            self._hp = 0

        if self._blockType == "unbreakable":
            self._hp -= 1
            self.updTexture()
            if self._hp == 0:
                self._brkSound.play()
                player.updateOxygen(2, surface, level)
                player.AddScore(10)
            else:
                self._hitSound.play()

        elif self._blockType == "pill":
            player.updateOxygen(3, surface, level)
            player.AddScore(20)
            self._brkSound.play()
            self._hp -= 1

        elif self._blockType == "delayed":
            if not self.idAcc():
                player.AddScore(10)
                self._brkSound.play()
            self._isDisappearing = True
            self.updTexture()

        elif self._blockType == "end":
            # Il blocco End NON viene distrutto: resta visibile.
            # Posta solo l'evento cambio livello.
            self.changeLvl()

        else:
            player.AddScore(10)
            self._brkSound.play()
            self._hp -= 1

        # Chain reaction
        if self._chain_reaction == 1 and nochain == 0 and self._blockType == "classic":
            if level[self._posY + 1][self._posX].hpAccess() != 0 \
                    and level[self._posY + 1][self._posX].typeAccess() == "classic":
                if level[self._posY + 1][self._posX].ColorAccess() == self._colors:
                    level[self._posY + 1][self._posX].hit(surface, level, player)   # Chiamata ricorsiva

            if level[self._posY - 1][self._posX].hpAccess() != 0 \
                    and level[self._posY - 1][self._posX].typeAccess() == "classic":
                if level[self._posY - 1][self._posX].ColorAccess() == self._colors:
                    level[self._posY - 1][self._posX].hit(surface, level, player)

            if self._posX < len(level[0]) - 1 \
                    and level[self._posY][self._posX + 1].hpAccess() != 0 \
                    and level[self._posY][self._posX + 1].typeAccess() == "classic":
                if level[self._posY][self._posX + 1].ColorAccess() == self._colors:
                    level[self._posY][self._posX + 1].hit(surface, level, player)

            if self._posX > 0 \
                    and level[self._posY][self._posX - 1].hpAccess() != 0 \
                    and level[self._posY][self._posX - 1].typeAccess() == "classic":
                if level[self._posY][self._posX - 1].ColorAccess() == self._colors:
                    level[self._posY][self._posX - 1].hit(surface, level, player)

        self.display(surface, self._currOffset)

    # ── Fisica caduta (sistema a tick, non a get_ticks) ───────────────
    def isShaking(self):
        return self._isShaking

    def isFalling(self):
        return self._isFalling

    def startShaking(self):
        """Inizia il tremore. Chiamato da applyGravity quando il blocco sotto è vuoto."""
        if not self._isShaking and not self._isFalling:
            self._isShaking  = True
            self._shakeTicks = 0

    def tickShake(self):
        """
        Avanza di un tick il tremore. Chiamato da applyGravity ad ogni ciclo.
        Ritorna True se il blocco è pronto a cadere.
        """
        if self._isShaking:
            self._shakeTicks += 1
            if self._shakeTicks >= self.SHAKE_TICKS_BEFORE_FALL:
                self._isShaking  = False
                self._isFalling  = True
                self._shakeTicks = 0
                return True
        return False

    def stopFalling(self):
        """Atterraggio: resetta tutto."""
        self._isFalling  = False
        self._isShaking  = False
        self._shakeTicks = 0

    def updatePos(self, newX, newY):
        self._posX = newX
        self._posY = newY

    # ── Texture ───────────────────────────────────────────────────────
    def updCoText(self, level):
        """Ricalcola la texture di connessione per i blocchi Classic."""
        if not (hasattr(self, '_blockType') and self._blockType == "classic"):
            return

        max_y = len(level) - 1
        max_x = len(level[0]) - 1

        def same_color(y, x):
            if not (0 <= y <= max_y and 0 <= x <= max_x):
                return False
            t = level[y][x]
            return (t.typeAccess() == "classic"
                    and t.hpAccess() > 0
                    and hasattr(t, '_colors')
                    and t._colors == self._colors)

        has_down  = same_color(self._posY + 1, self._posX)
        has_up    = same_color(self._posY - 1, self._posX)
        has_right = same_color(self._posY,     self._posX + 1)
        has_left  = same_color(self._posY,     self._posX - 1)

        total = 0
        if has_down:  total += 32
        if has_up:    total += 128
        if has_right: total += 64
        if has_left:  total += 16

        if has_down  and has_left  and same_color(self._posY + 1, self._posX - 1): total += 1
        if has_down  and has_right and same_color(self._posY + 1, self._posX + 1): total += 2
        if has_up    and has_left  and same_color(self._posY - 1, self._posX - 1): total += 4
        if has_up    and has_right and same_color(self._posY - 1, self._posX + 1): total += 8

        try:
            from connectCorrect import correct
            total = correct(total)
        except Exception:
            pass

        base   = path.join("Assets", "Textures", "Blocks", str(self._colors))
        p      = path.join(base, str(total) + ".png")
        if path.isfile(p):
            self._texturePath = p
        else:
            p2 = path.join(base, str(total & 240) + ".png")
            self._texturePath = p2 if path.isfile(p2) else path.join(base, "0.png")

    def display(self, surface, currentOffset=0):
        shakeX, shakeY = 0, 0
        if self._isShaking:
            shakeX = random.randint(-2, 2)
            shakeY = random.randint(-1, 1)

        pos = (self._posX * 64 + 26 + shakeX,
               (self._posY * 64 + 12) - currentOffset * 64 + shakeY)

        try:
            surface.blit(load_cached(self._bg), pos)
            if self._hp > 0:
                surface.blit(load_cached(self._texturePath), pos)
        except Exception as e:
            print(f"ERRORE GRAFICO blocco ({self._posX},{self._posY}): {e}")


# ─────────────────────────────────────────────────────────────────────
# CLASSI FIGLIE
# ─────────────────────────────────────────────────────────────────────

class Classic(Block):
    def __init__(self, posX, posY, colors, forceHP):
        Block.__init__(self, posX, posY, forceHP, 1, colors)
        self._texturePath = path.join("Assets", "Textures", "Blocks", str(colors), "0.png")
        self._blockType   = "classic"


class Unbreakable(Block):
    def __init__(self, posX, posY):
        Block.__init__(self, posX, posY, 5, 0)
        self._texturePath = path.join("Assets", "Textures", "Blocks", "Unbreakable", "5.png")
        self._blockType   = "unbreakable"
        self._brkSound    = pygame.mixer.Sound(path.join("Assets", "Sounds", "unbreakable.wav"))
        self._hitSound    = pygame.mixer.Sound(path.join("Assets", "Sounds", "tac.wav"))
        self._hitSound.set_volume(0.4)

    def updTexture(self):
        self._texturePath = path.join("Assets", "Textures", "Blocks", "Unbreakable",
                                      str(self.hpAccess()) + ".png")


class Solo(Block):
    def __init__(self, posX, posY):
        Block.__init__(self, posX, posY, 1, 0)
        self._texturePath = path.join("Assets", "Textures", "Blocks", "Solo", "b_s.png")
        self._blockType   = "solo"


class Delayed(Block):
    """
    Blocco con countdown visivo prima di sparire.
    Sistema a counter come nell'originale: timeout() viene chiamato dal
    timer EVTICSEC del main (ogni secondo × GAME_SPEED).
    __seconds parte da 2 → 2 chiamate a timeout() → 2 secondi → sparisce.
    """

    def __init__(self, posX, posY):
        Block.__init__(self, posX, posY, 5, 0)
        self._texturePath    = path.join("Assets", "Textures", "Blocks", "Delayed", "0.png")
        self._isDisappearing = False
        self._brkSound       = pygame.mixer.Sound(path.join("Assets", "Sounds", "crystal.wav"))
        self._brkSound.set_volume(0.35)
        self._blockType      = "delayed"
        self.__seconds       = 2    # step visivi prima della scomparsa (come originale)

    def getTimer(self):
        return self.__seconds  # Restituisce 2, 1 o 0

    def idAcc(self):
        return self._isDisappearing

    def updTexture(self):
        if self._isDisappearing and self._hp > 0:
            self._texturePath = path.join("Assets", "Textures", "Blocks", "Delayed",
                                          str(self.__seconds) + ".png")

    def timeout(self):
        """
        Chiamato dal timer EVTICSEC nel main (ogni secondo × GAME_SPEED).
        Decrementa il counter visivo e ritorna True quando il blocco muore.
        """
        if self._isDisappearing and self._hp > 0:
            self.__seconds -= 1
            self.updTexture()
            if self.__seconds <= 0:
                self._hp = 0
                self._brkSound.play()
                return True
        return False


class Pill(Block):
    def __init__(self, posX, posY):
        Block.__init__(self, posX, posY, 1, 0)
        self._texturePath = path.join("Assets", "Textures", "Blocks", "Pill", "pill_1.png")
        self._brkSound    = pygame.mixer.Sound(path.join("Assets", "Sounds", "pill.wav"))
        self._brkSound.set_volume(0.15)
        self._blockType   = "pill"

    def changeBG(self, bg):
        self._texturePath = path.join("Assets", "Textures", "Blocks", "Pill",
                                      "pill_" + str(bg) + ".png")
        self._bg          = path.join("Assets", "Textures", "Background",
                                      "bg_" + str(bg) + ".png")


class End(Block):
    def __init__(self, posX, posY):
        Block.__init__(self, posX, posY, 1, 0)
        self._texturePath = path.join("Assets", "Textures", "Blocks", "End", "b_s.png")
        self._blockType   = "end"
        self._nextLvl     = False

    def nextLvlAcc(self):
        return self._nextLvl

    def changeLvl(self):
        evChgLvl = pygame.event.Event(pygame.USEREVENT)
        pygame.event.post(evChgLvl)