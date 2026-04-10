from connectCorrect import *   # Importa tutte le funzioni dal modulo connectCorrect,
                               # che gestisce la logica di connessione visiva tra blocchi
                               # adiacenti dello stesso colore (es. quale texture di bordo usare)
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
# Dizionario globale che funge da cache per le texture già caricate.
# Chiave: percorso del file immagine (stringa).
# Valore: oggetto pygame.Surface già caricato e convertito.

def load_cached(filepath):
    # Funzione che carica un'immagine dal disco SOLO se non è già in cache.
    # Se è già stata caricata in precedenza, restituisce direttamente la versione
    # già in memoria, evitando costose operazioni di I/O su disco ad ogni frame.

    if filepath not in _texture_cache:
        # L'immagine NON è ancora in cache: bisogna caricarla

        try:
            _texture_cache[filepath] = pygame.image.load(filepath).convert_alpha()
            # pygame.image.load(filepath) legge il file immagine dal disco
            # .convert_alpha() converte la surface nel formato ottimale per pygame,
            # preservando il canale alpha (trasparenza). Questo velocizza notevolmente
            # il rendering rispetto a usare l'immagine nel suo formato originale.
            # Il risultato viene salvato nel dizionario con il percorso come chiave.

        except Exception:
            # Se il file non esiste o non è leggibile (es. percorso sbagliato),
            # crea una surface di fallback: un quadrato fucsia semitrasparente
            # facilmente riconoscibile durante il debug.
            surf = pygame.Surface((64, 64), pygame.SRCALPHA)
            # pygame.Surface((64, 64), ...) crea una superficie vuota 64×64 pixel
            # pygame.SRCALPHA abilita il canale alpha (trasparenza per pixel)

            surf.fill((200, 0, 200, 180))
            # Riempie la surface con colore RGBA: rosso=200, verde=0, blu=200 (fucsia)
            # alpha=180 (su 255): semitrasparente, così si vede che è un placeholder

            _texture_cache[filepath] = surf
            # Salva anche il placeholder in cache per non ripeterne la creazione

    return _texture_cache[filepath]
    # Restituisce la texture (originale o placeholder) dalla cache


def clear_cache():
    """Svuota la cache. Chiamare ad ogni cambio livello."""
    _texture_cache.clear()
    # .clear() rimuove tutti gli elementi dal dizionario.
    # Va chiamato quando si cambia livello per liberare la memoria RAM occupata
    # dalle texture del livello precedente (sfondi, blocchi, ecc.)


class Block:
    """Classe madre di tutti i blocchi."""
    # Questa è la classe base (superclasse) da cui ereditano tutti i tipi di blocco
    # (Classic, Unbreakable, Delayed, ecc.).
    # Contiene la logica comune: posizione, HP, fisica di caduta, display.

    def __init__(self, posX, posY, forceHP, chain_reaction, colors=0):
        # Costruttore: viene chiamato automaticamente quando si crea un nuovo blocco.
        # Parametri:
        #   posX           — colonna nella griglia di gioco (0 = sinistra)
        #   posY           — riga nella griglia di gioco (0 = alto)
        #   forceHP        — punti vita iniziali del blocco
        #   chain_reaction — 1 se il blocco partecipa alle reazioni a catena, 0 altrimenti
        #   colors         — indice del colore (0 = neutro, usato per i blocchi Classic)

        self._posX = posX           # Colonna del blocco nella griglia (es. 3 = quarta colonna)
        self._posY = posY           # Riga del blocco nella griglia (es. 5 = sesta riga)
        self._currOffset = 0        # Offset verticale di scorrimento della telecamera
                                    # (quante righe sono scomparse in fondo allo schermo)
        self._hp = forceHP          # Punti vita del blocco. Quando hp==0 il blocco è "aria" (vuoto)
        self._blockType = "neutral" # Tipo del blocco come stringa. Le sottoclassi lo sovrascrivono.
                                    # Usato da altri metodi per distinguere il comportamento.

        # ── Fisica caduta ─────────────────────────────────────────────

        self._isShaking = False     # Flag: True = il blocco sta tremando prima di cadere
        self._shakeTicks = 0        # Contatore di tick di tremore trascorsi
        self._isFalling = False     # Flag: True = il blocco sta attualmente cadendo
        # Con EVTIC_GRAVITY a 400ms × GAME_SPEED=2 → 200ms, SHAKE_TICKS=5 = ~1s di tremore.
        self.SHAKE_TICKS_BEFORE_FALL = 5
        # Numero di tick di tremore prima che il blocco inizi a cadere.
        # Ogni tick corrisponde a un evento EVTIC_GRAVITY. Con GAME_SPEED=2
        # e un tick ogni 200ms, 5 tick = circa 1 secondo di tremore visivo.

        if chain_reaction:
            # Solo i blocchi con reazione a catena (es. Classic) hanno un colore.
            # Per gli altri (Unbreakable, Solo, ecc.) il colore non è necessario.
            self._colors = colors   # Salva il colore per confrontarlo con i vicini nella chain reaction

        self._chain_reaction = chain_reaction
        # Salva il flag di chain reaction per usarlo nel metodo hit()

        self._texturePath = path.join("Assets", "Textures", "Blocks", "Neutral", "b_s.png")
        # Percorso di default alla texture del blocco neutro.
        # Le sottoclassi lo sovrascrivono nel loro __init__ con la texture appropriata.
        # path.join costruisce il percorso in modo cross-platform:
        # su Windows → "Assets\Textures\Blocks\Neutral\b_s.png"
        # su Linux/Mac → "Assets/Textures/Blocks/Neutral/b_s.png"

        self._bg = path.join("Assets", "Textures", "Background", "bg_1.png")
        # Percorso alla texture di sfondo disegnata SOTTO il blocco.
        # Lo sfondo cambia con il progresso nel livello (bg_1, bg_2, ecc.)

        self._brkSound = pygame.mixer.Sound(path.join("Assets", "Sounds", "pop.wav"))
        # Carica il suono da riprodurre quando il blocco viene distrutto.
        # pygame.mixer.Sound crea un oggetto audio pronto per essere riprodotto.

        self._brkSound.set_volume(0.3)
        # Imposta il volume del suono di rottura al 30% del massimo (range 0.0 - 1.0).
        # Valore basso per non coprire altri suoni del gioco.

    # ── Accessors ─────────────────────────────────────────────────────
    # I metodi seguenti sono "getter": restituiscono il valore degli attributi privati
    # senza esporli direttamente.

    def hpAccess(self):
        return self._hp             # Restituisce i punti vita correnti del blocco.
                                    # hp==0 significa che il blocco è vuoto/aria.

    def ColorAccess(self):
        return self._colors         # Restituisce l'indice colore del blocco (per la chain reaction).

    def posAcc(self):
        return self._posY, self._posX   # Restituisce la posizione come tupla (riga, colonna).
                                        # Nota: prima Y (riga) poi X (colonna), non il contrario!

    def typeAccess(self):
        return self._blockType      # Restituisce il tipo del blocco come stringa
                                    # (es. "classic", "unbreakable", "pill", ecc.)

    # ── Logica ────────────────────────────────────────────────────────

    def changeBG(self, bg):
        # Aggiorna il percorso dello sfondo quando il gioco avanza a una nuova sezione.
        # 'bg' è un numero intero (es. 1, 2, 3) che corrisponde a diversi sfondi del livello.
        self._bg = path.join("Assets", "Textures", "Background",
                             "bg_" + str(bg) + ".png")
        # Costruisce es. "Assets/Textures/Background/bg_2.png"

    def updOffset(self, currentOffset):
        # Aggiorna l'offset di scorrimento della telecamera.
        # Quando il giocatore scende abbastanza, la telecamera scorre verso il basso:
        # tutti i blocchi devono sapere di quante righe sono visivamente "saliti".
        self._currOffset = currentOffset

    def hit(self, surface, level, player, nochain=0, instakill=0, delayedTimeout=0):
        # Metodo chiamato quando il giocatore (o una chain reaction) colpisce questo blocco.
        # Parametri:
        #   surface      — la superficie pygame su cui ridisegnare
        #   level        — la griglia 2D di tutti i blocchi del livello
        #   player       — oggetto Character del giocatore (per aggiornare punteggio/ossigeno)
        #   nochain      — se 1, non propagare la chain reaction (evita ricorsione infinita)
        #   instakill    — se 1, porta immediatamente gli HP a 0 (usato per revive/debug)
        #   delayedTimeout — parametro per i blocchi Delayed (non usato qui nella base)

        if instakill:
            self._hp = 0
            # Se instakill è attivo, azzera subito gli HP indipendentemente dal tipo di blocco.
            # Usato da revive() per distruggere il blocco sopra al giocatore che rinasce.

        if self._blockType == "unbreakable":
            # ── Blocco Unbreakable (X): ha 5 HP, si rompe solo colpendolo 5 volte ──
            self._hp -= 1           # Riduci di 1 l'HP ad ogni colpo
            self.updTexture()       # Aggiorna la texture in base all'HP rimanente
                                    # (mostra crepe progressive: 5, 4, 3, 2, 1)
            if self._hp == 0:
                # Blocco completamente distrutto
                self._brkSound.play()                   # Riproduci suono di rottura
                player.updateOxygen(2, surface, level)  # Funzione 2 = -20% ossigeno (penalità)
                player.AddScore(10)                     # Aggiungi 10 punti al punteggio
            else:
                # Blocco danneggiato ma non distrutto
                self._hitSound.play()   # Suono di colpo (diverso dal suono di rottura)

        elif self._blockType == "pill":
            # ── Blocco Pill (capsula dell'ossigeno): si raccoglie toccandola ──
            player.updateOxygen(3, surface, level)  # Funzione 3 = +20% ossigeno (o reset a 100%)
            player.AddScore(20)                     # Vale il doppio dei blocchi normali
            self._brkSound.play()                   # Suono di raccolta
            self._hp -= 1                           # Porta gli HP a 0 (la pill sparisce)

        elif self._blockType == "delayed":
            # ── Blocco Delayed (cristallo): non si rompe subito, inizia un countdown ──
            if not self.idAcc():
                # idAcc() restituisce True se il countdown è già partito.
                # Il punteggio si aggiunge SOLO al primo colpo, non ad ogni tick del countdown.
                player.AddScore(10)
                self._brkSound.play()       # Suono al primo colpo

            self._isDisappearing = True     # Avvia il countdown visivo (la texture cambia)
            self.updTexture()               # Mostra la prima frame del countdown

        elif self._blockType == "end":
            # ── Blocco End (uscita del livello): il giocatore ha completato il livello ──
            self.changeLvl()    # Lancia l'evento pygame per cambiare livello

        else:
            # ── Blocco standard (Classic, Solo, Neutral): comportamento base ──
            player.AddScore(10)     # 10 punti per ogni blocco normale distrutto
            self._brkSound.play()   # Suono di rottura
            self._hp -= 1           # Riduce gli HP di 1 (i blocchi normali hanno 1 HP)

        # ── Chain Reaction ────────────────────────────────────────────────────
        # Se questo blocco ha chain_reaction==1, nochain==0 ed è di tipo "classic",
        # colpisce automaticamente i blocchi adiacenti dello stesso colore.
        # Questo crea l'effetto "esplosione a catena" dei blocchi colorati.
        if self._chain_reaction == 1 and nochain == 0 and self._blockType == "classic":

            # Controlla il blocco SOTTO (posY+1, stesso posX)
            if level[self._posY + 1][self._posX].hpAccess() != 0 \
                    and level[self._posY + 1][self._posX].typeAccess() == "classic":
                # Il blocco sotto esiste (hp > 0) ed è di tipo classic
                if level[self._posY + 1][self._posX].ColorAccess() == self._colors:
                    # Ha lo stesso colore → propagate la chain reaction
                    level[self._posY + 1][self._posX].hit(surface, level, player)
                    # nochain=0 di default: la reazione si propaga ulteriormente

            # Controlla il blocco SOPRA (posY-1, stesso posX)
            if level[self._posY - 1][self._posX].hpAccess() != 0 \
                    and level[self._posY - 1][self._posX].typeAccess() == "classic":
                if level[self._posY - 1][self._posX].ColorAccess() == self._colors:
                    level[self._posY - 1][self._posX].hit(surface, level, player)

            # Controlla il blocco a DESTRA (stessa posY, posX+1)
            if self._posX < len(level[0]) - 1 \
                    and level[self._posY][self._posX + 1].hpAccess() != 0 \
                    and level[self._posY][self._posX + 1].typeAccess() == "classic":
                # Controlla anche i limiti della griglia (posX < larghezza - 1)
                # per evitare IndexError sul bordo destro
                if level[self._posY][self._posX + 1].ColorAccess() == self._colors:
                    level[self._posY][self._posX + 1].hit(surface, level, player)

            # Controlla il blocco a SINISTRA (stessa posY, posX-1)
            if self._posX > 0 \
                    and level[self._posY][self._posX - 1].hpAccess() != 0 \
                    and level[self._posY][self._posX - 1].typeAccess() == "classic":
                # self._posX > 0: evita IndexError sul bordo sinistro
                if level[self._posY][self._posX - 1].ColorAccess() == self._colors:
                    level[self._posY][self._posX - 1].hit(surface, level, player)

        self.display(surface, self._currOffset)
        # Ridisegna il blocco dopo il colpo per mostrare immediatamente
        # lo stato aggiornato (texture danneggiata o blocco sparito)

    # ── Fisica caduta (sistema a tick) ────────────────────────────────

    def isShaking(self):
        return self._isShaking  # Restituisce True se il blocco sta tremando.
                                # Usato da Character.fall() per evitare che il giocatore
                                # si appoggi su un blocco instabile.

    def isFalling(self):
        return self._isFalling  # Restituisce True se il blocco è in caduta libera.
                                # Analogamente usato da Character.fall() come guardia.

    def startShaking(self):
        """Inizia il tremore. Chiamato da applyGravity quando il blocco sotto è vuoto."""
        if not self._isShaking and not self._isFalling:
            # Avvia il tremore solo se il blocco non sta già tremando o cadendo.
            # Evita di azzerare il contatore di shake se il ciclo è già in corso.
            self._isShaking  = True     # Abilita la flag di tremore
            self._shakeTicks = 0        # Azzera il contatore di tick

    def tickShake(self):
        """
        Avanza di un tick il tremore. Chiamato da applyGravity ad ogni ciclo.
        Ritorna True se il blocco è pronto a cadere.
        """
        if self._isShaking:
            self._shakeTicks += 1       # Incrementa il contatore: un tick è trascorso

            if self._shakeTicks >= self.SHAKE_TICKS_BEFORE_FALL:
                # Il blocco ha tremato abbastanza: è il momento di cadere
                self._isShaking  = False    # Disabilita il tremore
                self._isFalling  = True     # Abilita la caduta
                self._shakeTicks = 0        # Resetta il contatore per uso futuro
                return True                 # Segnala al chiamante che la caduta può iniziare
        return False    # Il blocco sta ancora tremando, non è ancora pronto a cadere

    def stopFalling(self):
        """Atterraggio: resetta tutto."""
        self._isFalling  = False    # Il blocco non è più in caduta
        self._isShaking  = False    # Assicura che anche il tremore sia disattivato
        self._shakeTicks = 0        # Resetta il contatore tick

    def updatePos(self, newX, newY):
        # Aggiorna la posizione logica del blocco nella griglia.
        # Chiamato da applyGravity dopo che un blocco è caduto di una riga.
        self._posX = newX   # Nuova colonna (di solito invariata durante la caduta)
        self._posY = newY   # Nuova riga (incrementata di 1 ad ogni step di caduta)

    # ── Texture ───────────────────────────────────────────────────────

    def updCoText(self, level):
        """Ricalcola la texture di connessione per i blocchi Classic."""
        # Questo metodo determina quale texture mostrare per un blocco Classic
        # in base ai suoi vicini dello stesso colore (es. se ha un vicino sotto,
        # mostra un bordo aperto verso il basso; se è isolato, mostra tutti i bordi chiusi).
        # Il sistema usa un valore numerico bitmask per codificare le connessioni.

        if not (hasattr(self, '_blockType') and self._blockType == "classic"):
            return
            # Questo metodo ha senso solo per i blocchi Classic.
            # hasattr() verifica che l'attributo esista prima di leggerlo
            # (sicurezza extra, perché Block.__init__ non imposta sempre _blockType="classic")

        max_y = len(level) - 1      # Indice massimo valido per le righe (evita IndexError)
        max_x = len(level[0]) - 1   # Indice massimo valido per le colonne

        def same_color(y, x):
            # Funzione interna: verifica se il blocco in posizione (y,x) è Classic,
            # ha HP > 0 (esiste), e ha lo stesso colore di questo blocco.
            # Restituisce False se (y,x) è fuori dai limiti della griglia.
            if not (0 <= y <= max_y and 0 <= x <= max_x):
                return False    # Fuori dalla griglia: tratta come "nessun vicino"
            t = level[y][x]
            return (t.typeAccess() == "classic"         # È un blocco Classic
                    and t.hpAccess() > 0                # Esiste (non è aria)
                    and hasattr(t, '_colors')            # Ha un attributo colore
                    and t._colors == self._colors)       # Ha lo stesso colore

        # Controlla i 4 vicini diretti (N, S, E, W)
        has_down  = same_color(self._posY + 1, self._posX)     # Vicino in basso
        has_up    = same_color(self._posY - 1, self._posX)     # Vicino in alto
        has_right = same_color(self._posY,     self._posX + 1) # Vicino a destra
        has_left  = same_color(self._posY,     self._posX - 1) # Vicino a sinistra

        # Costruisce un valore bitmask che codifica quali lati sono "connessi":
        # Ogni direzione ha un valore di bit specifico (compatibile con il sistema connectCorrect)
        total = 0
        if has_down:  total += 32   # Bit per la connessione verso il basso
        if has_up:    total += 128  # Bit per la connessione verso l'alto
        if has_right: total += 64   # Bit per la connessione verso destra
        if has_left:  total += 16   # Bit per la connessione verso sinistra

        # Aggiunge i bit per gli angoli (diagonali), ma SOLO se entrambi i lati
        # adiacenti all'angolo sono connessi (evita connessioni angolari "false"):
        if has_down  and has_left  and same_color(self._posY + 1, self._posX - 1): total += 1   # Angolo in basso-sinistra
        if has_down  and has_right and same_color(self._posY + 1, self._posX + 1): total += 2   # Angolo in basso-destra
        if has_up    and has_left  and same_color(self._posY - 1, self._posX - 1): total += 4   # Angolo in alto-sinistra
        if has_up    and has_right and same_color(self._posY - 1, self._posX + 1): total += 8   # Angolo in alto-destra

        try:
            from connectCorrect import correct
            total = correct(total)
            # La funzione correct() normalizza il valore bitmask: verifica che
            # esista effettivamente un file texture per quel valore, e se no
            # lo mappa al valore più vicino disponibile.
        except Exception:
            pass    # Se connectCorrect non è disponibile, usa il valore grezzo

        # Costruisce il percorso alla texture corrispondente al bitmask calcolato
        base = path.join("Assets", "Textures", "Blocks", str(self._colors))
        # Es. "Assets/Textures/Blocks/3" per il colore 3 (arancione)

        p = path.join(base, str(total) + ".png")
        # Es. "Assets/Textures/Blocks/3/240.png" per il bitmask 240

        if path.isfile(p):
            self._texturePath = p   # La texture esatta esiste: usala
        else:
            # La texture esatta non esiste: prova la versione senza angoli
            # (azzera i 4 bit bassi che codificano gli angoli: total & 240 = total & 0b11110000)
            p2 = path.join(base, str(total & 240) + ".png")
            self._texturePath = p2 if path.isfile(p2) else path.join(base, "0.png")
            # Se neanche la versione senza angoli esiste, usa "0.png" (blocco isolato, tutti i bordi chiusi)

    def display(self, surface, currentOffset=0):
        # Disegna il blocco sullo schermo.
        # Parametri:
        #   surface       — la superficie pygame su cui disegnare
        #   currentOffset — quante righe la telecamera è scesa (per lo scorrimento verticale)

        base_pos = (
            self._posX * 64 + 26,                       # X in pixel: ogni cella è 64px larga, +26 di margine sinistro
            (self._posY * 64 + 12) - currentOffset * 64  # Y in pixel: ogni cella è 64px alta,
                                                          # +12 di margine superiore, -offset×64 per lo scorrimento
        )
        # base_pos è la posizione visiva del CENTRO della cella, senza effetti di shake.

        shakeX, shakeY = 0, 0   # Offset di shake inizialmente nullo
        if self._isShaking:
            shakeX = random.randint(-2, 2)  # Offset orizzontale casuale tra -2 e +2 pixel
            shakeY = random.randint(-1, 1)  # Offset verticale casuale tra -1 e +1 pixel
            # Questi piccoli spostamenti casuali creano l'effetto visivo di tremore

        block_pos = (base_pos[0] + shakeX, base_pos[1] + shakeY)
        # Posizione finale della texture del blocco: posizione base + shake

        try:
            # BG sempre alla posizione base — nessun artefatto di sfondo
            surface.blit(load_cached(self._bg), base_pos)
            # Disegna prima lo sfondo alla posizione BASE (senza shake).
            # Questo "pulisce" la cella ridisegnando lo sfondo corretto,
            # evitando artefatti visivi quando il blocco si sposta.

            if self._hp > 0:
                surface.blit(load_cached(self._texturePath), block_pos)
                # Disegna la texture del blocco SOPRA allo sfondo, con l'eventuale shake.
                # Se hp == 0 il blocco è vuoto (aria): si mostra solo lo sfondo.

        except Exception as e:
            print(f"ERRORE GRAFICO blocco ({self._posX},{self._posY}): {e}")
            # Se la visualizzazione fallisce (es. texture corrotta), stampa un errore
            # ma non interrompe il gioco. Questo evita crash per file mancanti.


# ─────────────────────────────────────────────────────────────────────
# CLASSI FIGLIE
# Ogni sottoclasse specializza Block per un tipo specifico di blocco.
# ─────────────────────────────────────────────────────────────────────

class Classic(Block):
    # Il blocco colorato standard del gioco. Ha chain_reaction=1 (partecipa alle reazioni a catena).
    def __init__(self, posX, posY, colors, forceHP):
        Block.__init__(self, posX, posY, forceHP, 1, colors)
        # Chiama il costruttore della classe madre con:
        #   chain_reaction=1 → partecipa alle reazioni a catena
        #   colors=colors    → ha un colore specifico per la chain reaction

        self._texturePath = path.join("Assets", "Textures", "Blocks", str(colors), "0.png")
        # Imposta la texture iniziale: "0.png" è la texture di un blocco isolato
        # (nessun vicino connesso). updCoText() la aggiornerà in base ai vicini.

        self._blockType = "classic"
        # Sovrascrive il tipo della classe madre (era "neutral")


class Unbreakable(Block):
    # Blocco con 5 HP che emette suoni diversi a ogni colpo. Ha una texture diversa per ogni HP.
    def __init__(self, posX, posY):
        Block.__init__(self, posX, posY, 5, 0)
        # HP=5: ci vogliono 5 colpi per distruggerlo
        # chain_reaction=0: non partecipa alle chain reaction

        self._texturePath = path.join("Assets", "Textures", "Blocks", "Unbreakable", "5.png")
        # Texture iniziale al massimo degli HP (5 = integro, senza crepe)

        self._blockType = "unbreakable"

        self._brkSound = pygame.mixer.Sound(path.join("Assets", "Sounds", "unbreakable.wav"))
        # Sovrascrive il suono di rottura della classe madre con uno specifico per l'Unbreakable

        self._hitSound = pygame.mixer.Sound(path.join("Assets", "Sounds", "tac.wav"))
        # Suono aggiuntivo (non presente nella classe madre) riprodotto ad ogni colpo
        # quando il blocco è danneggiato ma non ancora distrutto

        self._hitSound.set_volume(0.4)  # Volume al 40%

    def updTexture(self):
        # Aggiorna la texture in base all'HP corrente per mostrare lo stato di danno.
        # Con HP=5 → "5.png" (integro); con HP=1 → "1.png" (quasi distrutto)
        self._texturePath = path.join("Assets", "Textures", "Blocks", "Unbreakable",
                                      str(self.hpAccess()) + ".png")
        # hpAccess() restituisce l'HP corrente; la texture corrispondente mostra le crepe


class Solo(Block):
    # Blocco senza chain reaction (chain_reaction=0) e con 1 solo HP. Non si connette visivamente.
    def __init__(self, posX, posY):
        Block.__init__(self, posX, posY, 1, 0)
        # HP=1: si distrugge con un colpo solo
        # chain_reaction=0: nessuna reazione a catena

        self._texturePath = path.join("Assets", "Textures", "Blocks", "Solo", "b_s.png")
        self._blockType   = "solo"


class Delayed(Block):
    """
    Blocco con countdown visivo prima di sparire.
    Sistema a counter: timeout() viene chiamato dal timer EVTICSEC (ogni secondo × GAME_SPEED).
    """
    # Quando colpito, avvia un countdown di 2 secondi (visibile con texture che cambia).
    # Dopo il countdown, scompare automaticamente anche senza altri colpi.

    def __init__(self, posX, posY):
        Block.__init__(self, posX, posY, 5, 0)
        # HP=5 ma non viene consumato dai colpi: la logica del Delayed ignora
        # la riduzione standard di HP e usa invece _isDisappearing + countdown.

        self._texturePath    = path.join("Assets", "Textures", "Blocks", "Delayed", "0.png")
        # Texture iniziale: aspetto integro (nessun countdown attivo)

        self._isDisappearing = False
        # Flag: True quando il countdown è stato avviato (il blocco è stato colpito)

        self._brkSound = pygame.mixer.Sound(path.join("Assets", "Sounds", "crystal.wav"))
        # Suono cristallino specifico per i blocchi Delayed

        self._brkSound.set_volume(0.35)     # Volume al 35%
        self._blockType = "delayed"
        self.__seconds = 2                  # Countdown iniziale in secondi (2 → 1 → 0 → sparisce)
                                            # Il doppio underscore rende questo attributo "privato"
                                            # anche rispetto alle sottoclassi (name mangling Python)

    def getTimer(self):
        return self.__seconds   # Restituisce il countdown corrente (2, 1, o 0)

    def idAcc(self):
        return self._isDisappearing
        # "id" = "is Disappearing". Restituisce True se il blocco è già in countdown.
        # Usato da hit() per non aggiungere punti e suono più di una volta.

    def updTexture(self):
        # Aggiorna la texture per mostrare visivamente il countdown corrente.
        if self._isDisappearing and self._hp > 0:
            # Aggiorna solo se il countdown è attivo E il blocco esiste ancora (hp > 0)
            self._texturePath = path.join("Assets", "Textures", "Blocks", "Delayed",
                                          str(self.__seconds) + ".png")
            # Es. con __seconds=2 → "2.png" (2 secondi rimasti)
            #     con __seconds=1 → "1.png" (1 secondo rimanente)

    def timeout(self):
        # Chiamato ogni secondo dal timer EVTICSEC in main.py.
        # Avanza il countdown e distrugge il blocco quando arriva a 0.
        # Restituisce True quando il blocco si è appena distrutto, False altrimenti.
        if self._isDisappearing and self._hp > 0:
            self.__seconds -= 1         # Scala il countdown di 1 secondo
            self.updTexture()           # Mostra la nuova texture del countdown
            if self.__seconds <= 0:
                # Il countdown è esaurito: il blocco scompare
                self._hp = 0            # hp=0 lo rende "aria" per tutta la logica del gioco
                self._brkSound.play()   # Suono finale di rottura
                return True             # Segnala al chiamante che il blocco è sparito
        return False    # Il blocco è ancora in attesa (countdown non ancora a 0)


class Pill(Block):
    # Capsula dell'ossigeno: si raccoglie camminandoci sopra o colpendola.
    # Non partecipa alle chain reaction e ha 1 HP.
    def __init__(self, posX, posY):
        Block.__init__(self, posX, posY, 1, 0)
        # HP=1, chain_reaction=0

        self._texturePath = path.join("Assets", "Textures", "Blocks", "Pill", "pill_1.png")
        # Texture iniziale della pill (il numero corrisponde al tema visivo del livello)

        self._brkSound = pygame.mixer.Sound(path.join("Assets", "Sounds", "pill.wav"))
        # Suono specifico per la raccolta della capsula (diverso dal suono dei blocchi normali)

        self._brkSound.set_volume(0.15)     # Volume basso (0.15): suono delicato di raccolta
        self._blockType = "pill"

    def changeBG(self, bg):
        # Sovrascrive changeBG della classe madre: la Pill cambia sia la sua texture
        # che lo sfondo in base al tema visivo del livello corrente.
        # (A differenza degli altri blocchi, la texture della Pill dipende dallo sfondo)
        self._texturePath = path.join("Assets", "Textures", "Blocks", "Pill",
                                      "pill_" + str(bg) + ".png")
        # Es. con bg=2 → "pill_2.png" (pill con schema cromatico del secondo livello)

        self._bg = path.join("Assets", "Textures", "Background",
                             "bg_" + str(bg) + ".png")
        # Aggiorna anche lo sfondo (comportamento identico alla classe madre)


class End(Block):
    # Blocco di uscita del livello. Toccarlo fa avanzare al livello successivo.
    def __init__(self, posX, posY):
        Block.__init__(self, posX, posY, 1, 0)
        # HP=1, chain_reaction=0

        self._texturePath = path.join("Assets", "Textures", "Blocks", "End", "b_s.png")
        self._blockType   = "end"
        self._nextLvl     = False
        # Flag interno (non usato attivamente): il cambio livello avviene tramite evento pygame

    def nextLvlAcc(self):
        return self._nextLvl    # Getter per il flag _nextLvl (attualmente sempre False)

    def changeLvl(self):
        # Genera un evento pygame personalizzato (USEREVENT) per segnalare al main loop
        # che il giocatore ha raggiunto l'uscita e il livello deve cambiare.
        evChgLvl = pygame.event.Event(pygame.USEREVENT)
        # pygame.USEREVENT è un tipo di evento riservato agli sviluppatori per usi custom

        pygame.event.post(evChgLvl)
        # .post() inserisce l'evento nella coda degli eventi pygame.
        # Il main loop leggerà questo evento nel prossimo ciclo e cambierà livello.