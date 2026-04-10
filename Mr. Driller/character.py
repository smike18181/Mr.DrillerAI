import pygame               
from os import path 

# ─────────────────────────────────────────────────────────────────────
# CACHE TEXTURE PERSONAGGIO
# Analogamente ai blocchi, il personaggio carica ogni texture una sola
# volta per evitare disk I/O ad ogni frame.
# ─────────────────────────────────────────────────────────────────────

_char_texture_cache = {}
# Dizionario globale per la cache delle texture del personaggio.
# Chiave: percorso del file immagine. Valore: oggetto pygame.Surface.

def _load_char_img(filepath):
    # Funzione di supporto: carica l'immagine del personaggio dalla cache
    # se disponibile, altrimenti dal disco. Evita di caricare lo stesso file
    # più volte al secondo.

    if filepath not in _char_texture_cache:
        # L'immagine non è ancora in cache: bisogna caricarla dal disco

        try:
            _char_texture_cache[filepath] = pygame.image.load(filepath).convert_alpha()
            # Carica il file immagine e lo converte nel formato ottimale per pygame
            # convert_alpha() preserva la trasparenza e velocizza il rendering

        except FileNotFoundError:
            # Il file specificato non esiste nel filesystem
            surf = pygame.Surface((40, 60), pygame.SRCALPHA)
            # Crea una surface di fallback 40×60 pixel (dimensioni approssimative del personaggio)
            # pygame.SRCALPHA abilita il canale alpha per la trasparenza

            surf.fill((255, 0, 0, 220))
            # Riempie con rosso quasi opaco (RGBA: 255,0,0,220).
            # Il rosso brillante è facilmente riconoscibile come placeholder durante il debug.

            _char_texture_cache[filepath] = surf
            # Salva il placeholder in cache per non rieseguire questa logica di fallback

        except Exception:
            # Qualsiasi altro errore di caricamento (file corrotto, formato non supportato, ecc.)
            surf = pygame.Surface((40, 60), pygame.SRCALPHA)
            surf.fill((255, 0, 0, 220))
            _char_texture_cache[filepath] = surf
            # Stesso fallback del caso FileNotFoundError

    return _char_texture_cache[filepath]
    # Restituisce la texture dalla cache (originale o placeholder)


class Character:        # Important : directions list : Up = 1; Right = 2; Down = 3; Left = 4
    """Character class"""
    # Classe che rappresenta il personaggio giocante (o controllato dall'AI).
    # Gestisce: posizione nella griglia, animazione, ossigeno, punteggio, vite,
    # movimento, perforazione, caduta e respawn.
    # CONVENZIONE DIREZIONI: Su=1, Destra=2, Giù=3, Sinistra=4

    def __init__(self, posX, posY, bg, lives):
        # Costruttore del personaggio.
        # Parametri:
        #   posX  — colonna iniziale nella griglia di gioco
        #   posY  — riga iniziale nella griglia di gioco
        #   bg    — indice del tema visivo (1, 2, 3...) usato per scegliere lo sfondo
        #   lives — numero di vite iniziali

        # ── Posizione nella griglia ───────────────────────────────────
        self.__posX = posX          # Colonna corrente del personaggio (0 = colonna sinistra)
        self.__posY = posY          # Riga corrente del personaggio (0 = riga in alto)
        self.__blocksFallen = 0     # Numero di blocchi che la telecamera ha "scrollato" verso l'alto.
                                    # Usato per calcolare la posizione visiva sullo schermo:
                                    # più il giocatore scende, più blocksFallen aumenta.
        self.__climb = 0            # Contatore di "salite" pendenti.
                                    # Quando il giocatore scala un blocco (move su un gradino),
                                    # __climb viene incrementato. Poi fall() lo decrementa
                                    # per farlo scendere alla quota corretta con gravità.

        # ── Statistiche di gioco ─────────────────────────────────────
        self.__oxygen = 100         # Livello di ossigeno corrente (0-100).
                                    # Scende ogni tick (funzione 1: -1%), cala molto colpendo
                                    # blocchi X (funzione 2: -20%), sale raccogliendo pill (funzione 3: +20%)
        self.__lives  = lives       # Vite rimaste. Se scendono a -1 (o meno), game over.
        self.__score  = 0           # Punteggio accumulato (parte da 0 ad ogni partita)

        # ── Direzione corrente ────────────────────────────────────────
        self.__current_direction = 0
        # Indice numerico della direzione corrente per l'animazione.
        # 0=Giù, 1=Destra, 2=Su, 3=Sinistra
        # (NOTA: diverso dalle convenzioni del commento in cima che usa 1=Su, 2=Destra, ecc.)

        # ── Percorso texture di sfondo personale ─────────────────────
        file = "bg_"
        file += str(bg)     # Aggiunge il numero del tema (es. "1", "2")
        file += ".png"
        # Costruzione manuale della stringa del nome file (equivale a f"bg_{bg}.png")
        self.__bg = path.join("Assets", "Textures", "Background", file)
        # Es. "Assets/Textures/Background/bg_1.png"
        # Usato da backDownCleanup() per pulire i pixel sopra il personaggio.

        # ── Flag di animazione ────────────────────────────────────────
        # Ogni flag corrisponde a uno stato/animazione del personaggio.
        # Il metodo Anim() legge questi flag e imposta la texture appropriata,
        # poi resetta il flag a False. Un solo flag può essere True alla volta.

        self.__IsMovingRight       = False  # Sta camminando verso destra
        self.__IsMovingLeft        = False  # Sta camminando verso sinistra
        self.__IsFalling           = False  # Sta cadendo per gravità
        self.__IsDrillingRight     = False  # Sta attivamente perforando a destra (con forare)
        self.__IsDrillingLeft      = False  # Sta attivamente perforando a sinistra
        self.__IsDrillingRight_off = False  # Frame "off" della perforazione destra (prima/dopo il colpo)
        self.__IsDrillingLeft_off  = False  # Frame "off" della perforazione sinistra
        self.__IsDrillingDown      = False  # Sta perforando verso il basso
        self.__IsReviving          = False  # Sta rinascendo dopo aver perso una vita
        self.__IsIdling            = True   # È fermo/inattivo (stato di default)

        self.__texturePath = path.join("Assets", "Textures", "Character", "play_d_off.png")
        # Texture iniziale: personaggio rivolto verso il basso, in stato idle ("off" = trapano spento)

    # ── Accessors (getter) ────────────────────────────────────────────

    def imgIndAcc(self):
        return self.__current_direction     # Indice direzione corrente per uso esterno (es. AI)

    def posAcc(self):
        return self.__posY, self.__posX     # Posizione come tupla (riga, colonna)
                                            # NOTA: Y prima di X, convenzione usata anche in Block

    def NeedToIdle(self, surface):
        self.__IsIdling = True  # Forza lo stato idle
        self.Anim(surface)      # Ridisegna immediatamente con la texture idle

    def IdlingAcc(self):
        return self.__IsIdling  # True se il personaggio non sta facendo nulla di specifico

    def blocksFallenAcc(self):
        return self.__blocksFallen  # Offset verticale della telecamera (in celle)

    def climbAcc(self):
        return self.__climb     # Quante "salite" sono ancora in coda da processare

    def livesAcc(self):
        return self.__lives     # Vite rimaste (usato dall'AI e dal display UI)

    def oxyAcc(self):
        return self.__oxygen    # Livello ossigeno corrente in % (usato dall'AI e dalla UI)

    def scoreAcc(self):
        return self.__score     # Punteggio corrente (usato dall'AI e dalla UI)

    def fallAcc(self):
        """Ritorna True se il personaggio sta cadendo."""
        return self.__IsFalling  # Usato per sincronizzare la logica di gioco con l'animazione

    def airTimerAcc(self):
        """Ritorna l'ossigeno attuale (usato come proxy timer dall'AI)."""
        return self.__oxygen    # L'AI usa questo per decidere se cercare ossigeno urgentemente

    # ── Animazione ────────────────────────────────────────────────────

    def Anim(self, surface):
        # Metodo di animazione: legge i flag booleani degli stati e imposta la texture
        # corretta in base allo stato attivo. Alla fine chiama display() per ridisegnare.
        # Ogni blocco if controlla un flag; se attivo, imposta la texture e resetta il flag.
        # L'ordine dei controlli determina la priorità visiva (il fondo ha la precedenza più bassa).

        if self.__IsIdling:
            # Stato di default: personaggio fermo, rivolto verso il basso
            if self.__oxygen > 20:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_d_off.png")
                # Texture normale (ossigeno sufficiente)
            else:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_d_off_o2.png")
                # Texture con indicatore di basso ossigeno (es. personaggio ansimante)
            # NOTA: __IsIdling NON viene resettato qui; rimane True finché un altro flag lo sovrascrive.
            # Questo è il comportamento di default persistente.

        if self.__IsFalling:
            if self.__oxygen > 20:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_fall.png")
            else:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_fall_o2.png")
            self.__IsFalling = False    # Reset del flag: l'animazione di caduta dura un solo frame
            self.__IsIdling  = False    # Non è più in idle

        if self.__IsReviving:
            # Animazione di rinascita dopo la morte
            self.__texturePath = path.join("Assets", "Textures", "Character", "play_dead_y.png")
            # "dead_y" probabilmente sta per "dead → yellow" o "dead → yes (respawning)"
            self.__IsReviving = False
            self.__IsIdling   = False

        if self.__IsMovingLeft:
            if self.__oxygen > 20:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_l_mov.png")
                # "l_mov" = left moving: texture del personaggio che cammina verso sinistra
            else:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_l_mov_o2.png")
            self.__IsMovingLeft = False
            self.__IsIdling     = False

        if self.__IsMovingRight:
            if self.__oxygen > 20:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_r_mov.png")
                # "r_mov" = right moving
            else:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_r_mov_o2.png")
            self.__IsMovingRight = False
            self.__IsIdling      = False

        if self.__IsDrillingRight_off:
            # "off" = trapano spento: primo/ultimo frame della perforazione destra
            # (postura con trapano alzato ma non in azione)
            if self.__oxygen > 20:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_r_off.png")
            else:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_r_off_o2.png")
            self.__IsDrillingRight_off = False
            self.__IsIdling            = False

        if self.__IsDrillingLeft_off:
            # Analogo a IsDrillingRight_off, ma verso sinistra
            if self.__oxygen > 20:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_l_off.png")
            else:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_l_off_o2.png")
            self.__IsDrillingLeft_off = False
            self.__IsIdling           = False

        if self.__IsDrillingRight:
            # "on" = trapano in funzione: il blocco a destra sta per essere/è perforato
            if self.__oxygen > 20:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_r_on.png")
            else:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_r_on_o2.png")
            self.__IsDrillingRight = False
            self.__IsIdling        = False

        if self.__IsDrillingLeft:
            if self.__oxygen > 20:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_l_on.png")
            else:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_l_on_o2.png")
            self.__IsDrillingLeft = False
            self.__IsIdling       = False

        if self.__IsDrillingDown:
            # Perforazione verso il basso
            if self.__oxygen > 20:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_d_on.png")
                # "d_on" = down, trapano attivo
            else:
                self.__texturePath = path.join("Assets", "Textures", "Character", "play_d_on_o2.png")
            self.__IsDrillingDown = False
            self.__IsIdling       = False

        self.display(surface)
        # Dopo aver aggiornato la texture, ridisegna il personaggio sullo schermo
        # con la nuova immagine selezionata

    # ── Metodi logici ─────────────────────────────────────────────────

    def AddScore(self, x):
        self.__score += x   # Aggiunge x punti al punteggio accumulato
        return self.__score  # Restituisce il nuovo punteggio (utile per debug o UI)

    def move(self, surface, direction, level):
        # Gestisce il movimento laterale del personaggio (destra o sinistra).
        # Controlla se il movimento è possibile (cella libera) e se è necessario
        # "scalare" un gradino (climb). Non gestisce il movimento verso il basso
        # (quello è gestito dalla gravità in fall()).
        #
        # Logica per ogni direzione:
        #   1. Movimento normale: la cella target è vuota (hp==0) → sposta il personaggio
        #   2. Pill: la cella target è una pill → raccoglila e sposta il personaggio
        #   3. Climb: la cella target è occupata ma quella sopra è libera → scala il gradino
        #   4. Climb+Pill: scala e raccoglie una pill nella cella di arrivo in alto

        # --- RIGHT (Direzione 2) ---
        if direction == 2:
            self.__current_direction = 1    # Aggiorna la direzione interna (1 = destra per l'animazione)
            self.__IsMovingRight = True     # Attiva il flag per l'animazione di camminata a destra

            if self.__posX < len(level[0]) - 1 and level[self.__posY][self.__posX + 1].hpAccess() == 0:
                # CASO 1: La cella a destra esiste (non si esce dai bordi) ed è vuota (hp==0 → aria)
                level[self.__posY][self.__posX].display(surface, self.__blocksFallen)
                # Ridisegna la cella corrente per "cancellare" il personaggio dalla sua vecchia posizione

                self.__posX += 1            # Sposta il personaggio di una colonna a destra
                self.display(surface)       # Ridisegna il personaggio nella nuova posizione
                level[self.__posY][self.__posX-1].display(surface, self.__blocksFallen)
                # Ridisegna la cella appena lasciata per pulire eventuali artefatti visivi

            elif self.__posX < len(level[0]) - 1 \
                    and level[self.__posY][self.__posX + 1].typeAccess() == "pill" \
                    and level[self.__posY][self.__posX + 1].hpAccess() != 0:
                # CASO 2: La cella a destra è una pill ancora attiva (hp != 0)
                level[self.__posY][self.__posX + 1].hit(surface, level, self)
                # Raccoglie la pill (aggiunge ossigeno e punti, hp pill → 0)

                level[self.__posY][self.__posX].display(surface, self.__blocksFallen)
                self.__posX += 1            # Sposta nella cella dove c'era la pill (ora vuota)
                self.display(surface)
                level[self.__posY][self.__posX - 1].display(surface, self.__blocksFallen)

            # Right Climb: la cella target è occupata, ma sopra c'è spazio → scala il gradino
            elif self.__posX < len(level[0]) - 1 \
                    and level[self.__posY][self.__posX + 1].hpAccess() != 0 \
                    and level[self.__posY - 1][self.__posX].hpAccess() == 0 \
                    and level[self.__posY - 1][self.__posX + 1].hpAccess() == 0:
                # Condizioni:
                #   - la cella a destra (stessa riga) è BLOCCATA (hp != 0) → c'è un muro
                #   - la cella sopra il personaggio (posY-1, stessa colonna) è LIBERA
                #   - la cella in diagonale in alto a destra (posY-1, posX+1) è LIBERA
                # → Il personaggio può scalare il muro come un gradino

                level[self.__posY][self.__posX].display(surface, self.__blocksFallen)
                # Pulisce la vecchia posizione

                self.__posX += 1    # Si sposta di una colonna a destra...
                self.__posY -= 1    # ...e di una riga verso l'alto (scala il gradino)
                self.__climb += 1   # Incrementa il contatore di climb pendenti
                                    # (la gravità lo riporterà giù gradualmente)

                self.display(surface)
                level[self.__posY + 1][self.__posX - 1].display(surface, self.__blocksFallen)
                # Ridisegna la cella di partenza (posY+1 perché ora il personaggio è salito)

            # Right Climb Pill: scala il gradino e raccoglie una pill in cima
            elif self.__posX < len(level[0]) - 1 \
                    and level[self.__posY][self.__posX + 1].hpAccess() != 0 \
                    and level[self.__posY - 1][self.__posX].hpAccess() == 0 \
                    and level[self.__posY - 1][self.__posX + 1].typeAccess() == "pill" \
                    and level[self.__posY - 1][self.__posX + 1].hpAccess() != 0:
                # Come Right Climb, ma la cella in alto a destra contiene una pill
                level[self.__posY - 1][self.__posX + 1].hit(surface, level, self)
                # Raccoglie la pill prima di spostarsi

                level[self.__posY][self.__posX].display(surface, self.__blocksFallen)
                self.__posX += 1
                self.__posY -= 1
                self.__climb += 1
                self.display(surface)
                level[self.__posY + 1][self.__posX - 1].display(surface, self.__blocksFallen)

        # --- LEFT (Direzione 4) ---
        if direction == 4:
            # Logica speculare alla direzione destra: stessi 4 casi ma verso sinistra
            self.__current_direction = 3    # 3 = sinistra per l'animazione
            self.__IsMovingLeft = True

            if self.__posX > 0 and level[self.__posY][self.__posX - 1].hpAccess() == 0:
                # CASO 1: La cella a sinistra esiste (posX > 0) ed è vuota
                level[self.__posY][self.__posX].display(surface, self.__blocksFallen)
                self.__posX -= 1    # Sposta a sinistra
                self.display(surface)
                level[self.__posY][self.__posX + 1].display(surface, self.__blocksFallen)
                # Ridisegna la cella appena lasciata (posX+1 perché il personaggio si è spostato a sinistra)

            elif self.__posX > 0 \
                    and level[self.__posY][self.__posX - 1].typeAccess() == "pill" \
                    and level[self.__posY][self.__posX - 1].hpAccess() != 0:
                # CASO 2: La cella a sinistra è una pill attiva
                level[self.__posY][self.__posX - 1].hit(surface, level, self)
                level[self.__posY][self.__posX].display(surface, self.__blocksFallen)
                self.__posX -= 1
                self.display(surface)
                level[self.__posY][self.__posX + 1].display(surface, self.__blocksFallen)

            # Left Climb
            elif self.__posX > 0 \
                and level[self.__posY][self.__posX - 1].hpAccess() != 0 \
                    and level[self.__posY - 1][self.__posX].hpAccess() == 0 \
                    and level[self.__posY - 1][self.__posX - 1].hpAccess() == 0:
                # Scala a sinistra: cella a sinistra bloccata, ma sopra e in diagonale libere
                level[self.__posY][self.__posX].display(surface, self.__blocksFallen)
                self.__posX -= 1    # Si sposta a sinistra...
                self.__posY -= 1    # ...e sale di una riga
                self.__climb += 1
                self.display(surface)
                level[self.__posY + 1][self.__posX + 1].display(surface, self.__blocksFallen)
                # posX+1 perché ora il personaggio è a sinistra della vecchia posizione

            # Left Climb Pill
            elif self.__posX > 0 \
                    and level[self.__posY][self.__posX - 1].hpAccess() != 0 \
                    and level[self.__posY - 1][self.__posX].hpAccess() == 0 \
                    and level[self.__posY - 1][self.__posX - 1].typeAccess() == "pill" \
                    and level[self.__posY - 1][self.__posX - 1].hpAccess() != 0:
                # Scala a sinistra raccogliendo una pill in cima
                level[self.__posY - 1][self.__posX - 1].hit(surface, level, self)
                level[self.__posY][self.__posX].display(surface, self.__blocksFallen)
                self.__posX -= 1
                self.__posY -= 1
                self.__climb += 1
                self.display(surface)
                level[self.__posY + 1][self.__posX + 1].display(surface, self.__blocksFallen)

    def breakBlock(self, surface, direction, level, currentBotLine):
        # Gestisce la perforazione attiva dei blocchi (azione di "drill").
        # A differenza di move(), questa funzione NON sposta il personaggio:
        # colpisce il blocco adiacente senza muoversi.
        # Parametri:
        #   surface       — superficie pygame per il ridisegno
        #   direction     — direzione del colpo (2=destra, 3=basso, 4=sinistra)
        #   level         — griglia 2D dei blocchi
        #   currentBotLine — riga inferiore visibile (limite per la perforazione verso il basso)

        # Right
        if direction == 2:
            self.__current_direction = 1        # Aggiorna la direzione visiva (destra)
            self.__IsDrillingRight_off = True    # Attiva il frame "off" del trapano destra
                                                 # (postura di perforazione, anche se non colpisce nulla)

            if self.__posX < len(level[0])-1 \
                    and level[self.__posY][self.__posX + 1].hpAccess() > 0 \
                    and level[self.__posY][self.__posX + 1].typeAccess() != "pill":
                # Condizioni per poter perforare a destra:
                #   - non si è al bordo destro della griglia
                #   - il blocco a destra esiste (hp > 0)
                #   - il blocco a destra NON è una pill (le pill si raccolgono camminandoci, non perforandole)
                self.__IsDrillingRight = True           # Attiva il frame "on" (trapano in azione)
                level[self.__posY][self.__posX + 1].hit(surface, level, self)   # Colpisce il blocco

        # Down
        elif direction == 3 \
                and self.__posY < currentBotLine \
                and level[self.__posY + 1][self.__posX].hpAccess() > 0 \
                and level[self.__posY + 1][self.__posX].typeAccess() != "pill":
            # Condizioni per perforare verso il basso:
            #   - il personaggio non è già alla riga inferiore visibile
            #   - il blocco sotto esiste (hp > 0)
            #   - il blocco sotto non è una pill
            self.__current_direction = 0    # 0 = giù per l'animazione
            self.__IsDrillingDown = True
            level[self.__posY + 1][self.__posX].hit(surface, level, self)

        # Left
        elif direction == 4:
            self.__current_direction = 3        # 3 = sinistra
            self.__IsDrillingLeft_off = True    # Frame "off" del trapano sinistra

            if self.__posX > 0 \
                    and level[self.__posY][self.__posX - 1].hpAccess() > 0 \
                    and level[self.__posY][self.__posX - 1].typeAccess() != "pill":
                # Condizioni analoghe alla destra, ma verso sinistra
                self.__IsDrillingLeft = True
                level[self.__posY][self.__posX - 1].hit(surface, level, self)

    def fall(self, surface, level):
        """
        BUG FIX: il player NON deve cadere su blocchi in isFalling() o isShaking().
        Il controllo originale usava solo hpAccess()==0, ignorando che un blocco instabile
        ha ancora hp>0 ma si sta spostando. Il player ci cadeva sopra, il blocco finiva
        la sua animazione spostandosi, lasciando il player sospeso nel vuoto.

        Ora la condizione per cadere è:
          - la cella sotto ha hp==0 (aria), OPPURE
          - la cella sotto è una pill (raccoglibile camminandoci sopra)
        MA NON se:
          - la cella sotto è un blocco con hp>0 in isFalling() o isShaking()
            → in quel caso il player NON scende: aspetta che il blocco si stabilizzi.

        Caso speciale: se la cella sotto è falling/shaking, il player rimane fermo
        (la gravità verrà ri-applicata al prossimo tick di EVTIC_GRAVITY).
        """
        # Gestisce la gravità sul personaggio: se la cella sotto è vuota, scende di una riga.
        # Restituisce sempre __blocksFallen (può essere usato dal chiamante per sapere
        # quante righe la telecamera ha scrollato).

        if self.__posY >= len(level) - 2:
            return self.__blocksFallen
            # Se il personaggio è già alla penultima riga (o oltre), non può cadere oltre.
            # -2 invece di -1 per tenere un margine di sicurezza contro IndexError.

        cell_below = level[self.__posY + 1][self.__posX]
        # Recupera l'oggetto blocco direttamente sotto il personaggio

        # Il blocco sotto è instabile ma ancora presente → il player non cade
        if cell_below.hpAccess() > 0 and (cell_below.isFalling() or cell_below.isShaking()):
            return self.__blocksFallen
            # BUG FIX: il blocco sotto ha ancora HP (non è aria) ma è in movimento.
            # Il personaggio deve aspettare che si stabilizzi prima di appoggiarsi.

        if cell_below.hpAccess() == 0 or cell_below.typeAccess() == "pill":
            # La cella sotto è aria (hp==0) O è una pill → il personaggio può cadere

            if self.__climb == 0:
                # Il personaggio non ha climb pendenti: caduta normale con gravità

                if cell_below.typeAccess() == "pill":
                    cell_below.hit(surface, level, self)
                    # Raccoglie la pill sotto (aggiunge ossigeno/punti e svuota la cella)

                self.__blocksFallen += 1    # Incrementa l'offset della telecamera
                self.__posY += 1            # Scende di una riga nella griglia
                self.__IsFalling = True     # Attiva il flag di animazione caduta

            else:
                # Ha un climb pendente: non è una "vera" caduta, è la discesa
                # naturale dopo aver scalato un gradino (climb ritardato dalla gravità)
                if cell_below.typeAccess() == "pill":
                    cell_below.hit(surface, level, self)

                self.__climb -= 1   # Decrementa il contatore di climb pendenti
                self.__posY += 1    # Scende di una riga (senza animazione di caduta,
                                    # perché è solo il completamento della scalata)

            return self.__blocksFallen  # Restituisce l'offset aggiornato

    def revive(self, surface, level):
        # Gestisce la morte e il respawn del personaggio.
        # Chiamato da updateOxygen() quando l'ossigeno scende a 0.

        if self.__lives <= 0:
            # Non ci sono più vite: game over silenzioso
            # (il main loop rileverà lives < 0 per il game over screen)
            self.__lives -= 1       # Scende a -1 o meno: segnale definitivo di game over
            self.__oxygen = 100     # Resetta l'ossigeno (per evitare loop di morti consecutive)

        else:
            # Ci sono ancora vite: animazione di respawn
            self.__IsReviving = True    # Attiva il flag per la texture di respawn
            self.__oxygen = 100         # Ripristina l'ossigeno al massimo
            self.__lives -= 1           # Consuma una vita

            if self.__posY > 0:
                # Controlla il blocco sopra il personaggio
                block_above = level[self.__posY - 1][self.__posX]
                if block_above.hpAccess() > 0:
                    block_above.hit(surface, level, self, 1, 1)
                    # nochain=1: non propagare chain reaction durante il respawn
                    # instakill=1: distruggi immediatamente il blocco sopra
                    # Serve a liberare spazio per l'animazione di respawn
                    # (senza questo, il personaggio sarebbe "schiacciato" dal blocco sopra)

            self.__oxygen = 100     # Secondo reset dell'ossigeno (sicurezza: hit() potrebbe averlo modificato)

    def resetCoord(self, bg):
        # Riposiziona il personaggio alla sua posizione di partenza del livello.
        # Chiamato all'inizio di ogni nuovo livello.

        self.__posX = 3             # Colonna iniziale fissa (4ª colonna da sinistra)
        self.__posY = 4             # Riga iniziale fissa (5ª riga dall'alto)
        self.__blocksFallen = 0     # Resetta l'offset della telecamera
        self.__climb = 0            # Resetta i climb pendenti
        self.__oxygen = 100         # Ripristina l'ossigeno al massimo

        # Aggiorna il percorso dello sfondo al tema del nuovo livello
        file = "bg_"
        file += str(bg)
        file += ".png"
        self.__bg = path.join("Assets", "Textures", "Background", file)
        # Es. con bg=2 → "Assets/Textures/Background/bg_2.png"

    def resetScore(self):
        # Resetta le statistiche per una nuova partita (non solo un nuovo livello).
        # A differenza di resetCoord(), resetta anche punteggio e vite.
        self.__score = 0    # Azzera il punteggio
        self.__lives = 2    # Ripristina le vite al valore iniziale (2 vite di default)

    def updateOxygen(self, funct, surface, level):
        # Aggiorna il livello di ossigeno in base alla funzione specificata.
        # Parametri:
        #   funct=1 → decremento lento (-1%): chiamato ogni tick del timer ossigeno
        #   funct=2 → penalità grossa (-20%): chiamato quando si colpisce un blocco X
        #   funct=3 → ricarica (+20%, max 100%): chiamato raccogliendo una pill

        if funct == 1:
            self.__oxygen -= 1      # Consumo normale: -1% per tick
        elif funct == 2:
            self.__oxygen -= 20     # Penalità: -20% per blocco X colpito
        elif funct == 3:
            if self.__oxygen <= 70:
                self.__oxygen += 20     # Se l'ossigeno è basso (≤70%), aggiunge 20%
            else:
                self.__oxygen = 100     # Se l'ossigeno è già alto (>70%), ricarica al massimo
                                        # Evita di superare 100% senza clampare

        if self.__oxygen <= 0:
            # L'ossigeno è esaurito: il personaggio muore
            snd_path = path.join("Assets", "Sounds", "ugh.wav")
            if path.exists(snd_path):
                # Riproduce il suono di morte solo se il file esiste
                ugh = pygame.mixer.Sound(snd_path)
                ugh.set_volume(0.70)    # Volume al 70%: suono importante
                ugh.play(0)             # Riproduce una volta (0 = no loop)
            self.revive(surface, level)  # Avvia la sequenza di respawn

    # ── Metodi grafici ────────────────────────────────────────────────

    def display(self, surface):
        """
        BUG FIX: usa _load_char_img (con cache) invece di pygame.image.load diretto.
        Il codice originale chiamava pygame.image.load() ad ogni frame (60 volte/s),
        causando I/O disco continuo e potenziale stuttering.
        """
        if not hasattr(self, '_Character__texturePath') or self.__texturePath is None:
            return
            # Controllo di sicurezza: verifica che __texturePath esista prima di usarla.
            # '_Character__texturePath' è il nome mangled di __texturePath
            # (il prefisso doppio underscore diventa _NomeClasse__attributo in Python).
            # Previene AttributeError in edge case di inizializzazione incompleta.

        img = _load_char_img(self.__texturePath)
        # Recupera la texture dalla cache (o la carica dal disco se non ancora in cache)

        surface.blit(img, (
            self.__posX * 64 + 26,                          # X in pixel: colonna × 64px + 26 di margine
            (self.__posY * 64 + 12) - self.__blocksFallen * 64   # Y in pixel: riga × 64px + 12 di margine
                                                                   # meno l'offset di scorrimento della telecamera
        ))
        # surface.blit(img, (x, y)) disegna l'immagine alla posizione (x, y) sullo schermo

    def backDownCleanup(self, surface):
        # Pulisce la cella SOPRA il personaggio dopo uno scorrimento verso il basso.
        # Quando la telecamera scorre (blocksFallen aumenta), la riga che scompare
        # in cima può lasciare pixel "fantasma". Questo metodo la ridisegna con lo sfondo.

        bg_img = _load_char_img(self.__bg) if path.exists(self.__bg) else None
        # Cerca di caricare l'immagine di sfondo dalla cache.
        # path.exists() verifica che il file esista prima di caricarlo.
        # Se non esiste, bg_img è None.

        if bg_img:
            surface.blit(bg_img, (
                self.__posX * 64 + 26,
                (self.__posY * 64 + 12) - self.__blocksFallen * 64 - 64
                # -64 aggiuntivo rispetto a display(): ridisegna la cella UNA RIGA SOPRA
                # la posizione attuale del personaggio
            ))
        else:
            # Fallback: se lo sfondo non è disponibile, copre la cella con un rettangolo nero
            pygame.draw.rect(surface, (0, 0, 0), (
                self.__posX * 64 + 26,
                (self.__posY * 64 + 12) - self.__blocksFallen * 64 - 64,
                64, 64  # Larghezza e altezza del rettangolo (una cella intera)
            ))

        self.display(surface)
        # Ridisegna il personaggio nella sua posizione corretta (dopo aver pulito la cella sopra)