import pygame               # Libreria principale del gioco: grafica, font, eventi
from os import path, makedirs
# Importa selettivamente dal modulo os:
# - path: per costruire percorsi file cross-platform (path.join, path.exists, path.isfile)
# - makedirs: per creare cartelle (anche quelle intermedie mancanti) con una sola chiamata
import level                # Importa il modulo level per usare generateLvl() in changeLvl()


# ── GESTIONE PUNTEGGI ────────────────────────────────────────────────

def storeScore(playerScore):
    # Salva il punteggio del giocatore nel file dei record.
    # Mantiene solo i 3 punteggi più alti (top 3 leaderboard).
    # Parametro:
    #   playerScore — punteggio intero da salvare (es. 1250)

    folder_path = path.join("Assets", "Score")
    # Costruisce il percorso della cartella dove si trova il file dei punteggi.
    # path.join usa il separatore corretto per il sistema operativo.

    if not path.exists(folder_path):
        makedirs(folder_path)
        # Se la cartella "Assets/Score" non esiste ancora, la crea.
        # makedirs crea anche le cartelle intermedie mancanti (es. "Assets" se non c'è).
        # Senza questo controllo, open() più in basso lancerebbe FileNotFoundError.

    file_path = path.join(folder_path, "score.txt")
    # Percorso completo del file dei punteggi: "Assets/Score/score.txt"

    # Apre o crea il file dei punteggi
    if path.isfile(file_path):
        scoreFile = open(file_path, "r+")
        # Modalità "r+" = read+write: apre il file esistente per lettura E scrittura.
        # Il cursore parte dall'inizio del file.
        # Non cancella il contenuto esistente (a differenza di "w+").
    else:
        scoreFile = open(file_path, "w+")
        # Modalità "w+" = write+read: crea un nuovo file (o azzera quello esistente)
        # e permette sia scrittura che lettura.

    # ── Gestione file vuoto ────────────────────────────────────────────
    scoreFile.seek(0)
    # Sposta il cursore all'inizio del file.
    # Necessario dopo l'apertura in "r+" per essere sicuri di leggere dall'inizio.

    if not scoreFile.read(1):
        # Prova a leggere 1 carattere: se il file è vuoto, la stringa è "" (falsy).
        # "not scoreFile.read(1)" è True se il file è vuoto.
        for i in range(100, 400, 100):
            scoreFile.write(str(i) + "\n")
            # Inizializza il file con punteggi placeholder: 100, 200, 300.
            # range(100, 400, 100) genera [100, 200, 300].
            # str(i) + "\n" scrive ogni numero su una riga separata.
        # Questo assicura che il file abbia sempre almeno 3 righe da leggere.

    scoreFile.seek(0)
    # Riporta il cursore all'inizio dopo l'eventuale scrittura dei placeholder,
    # per poter leggere tutte le righe dall'inizio nel passo successivo.

    lines = scoreFile.readlines()
    # Legge tutte le righe del file come lista di stringhe.
    # Es. ["100\n", "200\n", "300\n"] → liste con il carattere \n alla fine.

    # ── Parsing e pulizia dei punteggi ────────────────────────────────
    scores = []
    for line in lines:
        try:
            scores.append(int(line.strip()))
            # line.strip() rimuove spazi e \n dalla stringa (es. "250\n" → "250")
            # int(...) converte la stringa pulita in un intero
        except ValueError:
            pass
            # Se la riga non è convertibile in intero (es. riga vuota, testo corrotto),
            # la salta silenziosamente. Questo rende il parsing robusto.

    scores.append(playerScore)
    # Aggiunge il nuovo punteggio alla lista di quelli letti dal file

    scores.sort(reverse=True)
    # Ordina la lista in ordine decrescente (dal punteggio più alto al più basso).
    # reverse=True inverte l'ordine ascendente di default di sort().

    scores = scores[:3]
    # Mantieni solo i primi 3 elementi (top 3).
    # Lo slicing [:3] restituisce al massimo 3 elementi, anche se la lista ha meno.

    # ── Scrittura del file aggiornato ─────────────────────────────────
    scoreFile.seek(0)       # Riporta il cursore all'inizio per sovrascrivere dall'inizio
    scoreFile.truncate()    # Cancella tutto il contenuto del file a partire dalla posizione corrente (l'inizio).
                            # FONDAMENTALE: senza truncate(), se il nuovo contenuto è più corto
                            # del vecchio, resterebbero righe residue alla fine del file.
    for s in scores:
        scoreFile.write(str(s) + "\n")
        # Scrive ogni punteggio del top 3 su una riga separata

    scoreFile.close()       # Chiude il file e libera il file descriptor


def readScore(surface):
    # Legge i punteggi dal file e li disegna sullo schermo del menu.
    # Parametro:
    #   surface — la superficie pygame su cui disegnare i punteggi

    file_path = path.join("Assets", "Score", "score.txt")

    if not path.exists(file_path):
        return
        # Se il file non esiste ancora (prima partita, mai salvato), non disegna nulla.
        # Evita FileNotFoundError e mostra semplicemente un menu senza punteggi.

    try:
        scoreFile = open(file_path, "r")
        # Apre il file in sola lettura. Se non esiste, lanciaria FileNotFoundError
        # (già gestito dal controllo path.exists sopra).

        FontUi = pygame.font.Font("Assets/Misc/police/Act_Of_Rejection.ttf", 48)
        # Carica il font personalizzato per la UI a 48 punti.
        # Se il file del font non esiste, pygame.font.Font lancia FileNotFoundError
        # (catturata dal try/except esterno).

        lines = scoreFile.readlines()
        # Legge tutte le righe del file come lista di stringhe

        lines.sort(reverse=True, key=lambda x: int(x.strip()) if x.strip().isdigit() else 0)
        # Ordina le righe in ordine decrescente di punteggio.
        # key=lambda x: ...: la funzione di ordinamento applicata a ogni elemento.
        #   x.strip().isdigit(): True se la stringa (senza spazi) è un numero puro
        #   int(x.strip()) se è un numero, 0 altrimenti (righe non numeriche vanno in fondo)
        # Questo ordinamento è ridondante se storeScore() ha già salvato i punteggi in ordine,
        # ma garantisce la correttezza anche in caso di file modificato manualmente.

        dispQyt = min(len(lines), 3)
        # Numero di punteggi da visualizzare: al massimo 3, anche se ci sono meno righe.
        # min() evita IndexError se il file ha meno di 3 righe.

        for i in range(dispQyt):
            # Itera sui (al massimo) 3 punteggi da mostrare
            val = lines[i].strip()          # Rimuove \n e spazi dalla stringa del punteggio
            ttd = f"{i + 1} : {val}"        # Formatta come "1 : 1500", "2 : 1200", "3 : 800"
            scoreDisp = FontUi.render(ttd, 1, (220, 0, 255))
            # Renderizza il testo come superficie pygame.
            # Il primo argomento è la stringa da renderizzare.
            # 1 = antialiasing abilitato (testo con bordi levigati).
            # (220, 0, 255) = colore viola/fucsia in RGB.

            surface.blit(scoreDisp, (i*240+30, 535))
            # Disegna il testo sulla superficie del menu.
            # Posizione X: i*240+30 → i punteggi sono distribuiti orizzontalmente:
            #   i=0 → x=30  (sinistra)
            #   i=1 → x=270 (centro)
            #   i=2 → x=510 (destra)
            # Posizione Y: 535 → fisso in basso nello schermo

        scoreFile.close()   # Chiude il file

    except Exception as e:
        print(f"Errore lettura score: {e}")
        # Se qualcosa va storto (font mancante, file corrotto, ecc.),
        # stampa l'errore a console ma non fa crashare il gioco.


# ── MENU GRAFICO ──────────────────────────────────────────────────────

def mainMenu(surface, optionIM):
    # Disegna lo schermo del menu principale con sfondo, bottoni e punteggi.
    # Parametri:
    #   surface  — la superficie pygame su cui disegnare
    #   optionIM — quale opzione è attualmente selezionata (1=Play, 2=Quit)
    #              Cambia quale bottone appare in stato "selected" (versione _s.png)
    #              e quale appare in stato "unselected" (versione _u.png)

    try:
        bg = pygame.image.load(path.join("Assets", "Menu", "menu.png"))
        # Carica l'immagine di sfondo del menu.
        # NOTA: non usa la cache (a differenza dei blocchi): il menu viene mostrato
        # raramente quindi l'ottimizzazione non è critica qui.

        if optionIM == 1:
            playImg = pygame.image.load(path.join("Assets", "Buttons", "play_s.png"))
            # "_s" = selected: bottone Play evidenziato (es. bordo luminoso, colore diverso)
            quitImg = pygame.image.load(path.join("Assets", "Buttons", "quit_u.png"))
            # "_u" = unselected: bottone Quit non evidenziato (aspetto normale)

        elif optionIM == 2:
            playImg = pygame.image.load(path.join("Assets", "Buttons", "play_u.png"))
            # Play non selezionato
            quitImg = pygame.image.load(path.join("Assets", "Buttons", "quit_s.png"))
            # Quit selezionato

        surface.blit(bg, (0, 0))            # Disegna lo sfondo nell'angolo in alto a sinistra
        surface.blit(playImg, (310, 300))   # Disegna il bottone Play alla posizione fissa (310, 300)
        surface.blit(quitImg, (310, 400))   # Disegna il bottone Quit 100px sotto il Play
        readScore(surface)                  # Disegna i punteggi nella parte bassa dello schermo
        pygame.display.update()             # Aggiorna lo schermo mostrando tutto ciò che è stato disegnato.
                                            # Senza questa chiamata, le modifiche alla surface
                                            # non sarebbero visibili all'utente.

    except FileNotFoundError as e:
        print(f"Errore caricamento asset menu: {e}")
        # Se un'immagine del menu non viene trovata, stampa l'errore ma non crasha.
        # Il gioco rimane in un stato indefinito (schermo vuoto o con solo ciò che è già disegnato).


def changeLvl(currentLvl, player, is_ai=False):
    # Avanza al livello successivo e genera la nuova griglia.
    # Parametri:
    #   currentLvl — indice del livello corrente (0-based: 0 significa "prima del primo livello")
    #   player     — oggetto Character del giocatore (per resettare la posizione)
    #   is_ai      — True se il gioco è in modalità AI (evita di salvare il punteggio a fine gioco)
    #
    # Restituisce:
    #   (lvl, currentLvl, game_completed)
    #   lvl            — la nuova griglia 2D di blocchi (o lista vuota se gioco finito)
    #   currentLvl     — il nuovo numero di livello corrente
    #   game_completed — True se il gioco è completato (superato livello 10)

    if currentLvl < 10:
        # ── Ci sono ancora livelli da giocare (livelli 1-10) ──────────
        currentLvl += 1     # Avanza al livello successivo
        player.resetCoord(currentLvl)
        # Riposiziona il giocatore alla sua posizione di start del livello.
        # Passa currentLvl per aggiornare anche il tema grafico dello sfondo del personaggio.

        colors = 2 if currentLvl in [2, 7] else 4
        # Sceglie quanti colori usare per i blocchi Classic di questo livello:
        # - Livello 2 e 7: solo 2 colori (livello più semplice visivamente)
        # - Tutti gli altri: 4 colori (più varietà di chain reaction)
        # Avere meno colori significa blocchi dello stesso colore più frequenti
        # → più chain reaction → livello più "esplosivo" ma forse più semplice.

        bg_name = f"{currentLvl}"
        # Nome del tema grafico: corrisponde al numero del livello.
        # Es. livello 3 → "3" → sfondo "bg_3.png", pill "pill_3.png", ecc.

        if currentLvl > 4:
            lvl = level.generateLvl(colors, 155, 7, bg_name, 5, 7, 20, 10, 10, 12)
            # Livelli avanzati (5-10): più profondi (155 righe invece di 80)
            # e con DelayedP=12 invece di 10 (leggermente più blocchi ritardati).
            # I parametri espliciti sovrascrivono i default di generateLvl():
            #   colors=colors, lines=155, width=7, background=bg_name,
            #   pillP=5, PillPL=7, PillMLE=20, SoloP=10, UnbreakableP=10, DelayedP=12
        else:
            lvl = level.generateLvl(colors, 80, 7, bg_name)
            # Livelli iniziali (1-4): più corti (80 righe) e con parametri di default
            # per pillle e blocchi speciali (generateLvl usa i default della firma della funzione).

        # ── Calcolo texture di connessione per tutti i blocchi ─────────
        for row in lvl:
            for block_obj in row:
                block_obj.updCoText(lvl)
                # Per ogni blocco Classic, calcola quale texture usare in base
                # ai vicini dello stesso colore (sistema bitmask delle connessioni).
                # Deve essere fatto DOPO che tutti i blocchi sono stati generati,
                # perché ogni blocco deve conoscere i suoi vicini.
                # Per blocchi non-Classic (Unbreakable, Pill, ecc.), updCoText fa return subito.

        return lvl, currentLvl, False
        # Restituisce: la nuova griglia, il nuovo numero di livello, game_completed=False

    else:
        # ── Tutti i livelli completati (oltre il livello 10) ──────────
        if not is_ai:
            storeScore(player.scoreAcc())
            # In modalità umana, salva il punteggio finale nella leaderboard.
            # In modalità AI (is_ai=True), non si salva per non inquinare
            # la leaderboard con i punteggi dell'agente in training.

        game_completed = True
        return [], currentLvl, game_completed
        # Restituisce: lista vuota (nessun livello), il numero di livello corrente,
        # game_completed=True → il main loop mostrerà la schermata di fine gioco.


def restart(player):
    # Riavvia la partita dal livello 1 (usato quando il giocatore fa game over
    # o quando l'AI inizia un nuovo episodio).
    # Parametro:
    #   player — oggetto Character del giocatore

    lvl, currentlvl, won = changeLvl(0, player, is_ai=True)
    # Chiama changeLvl con currentLvl=0: avanza da 0 a 1, generando il primo livello.
    # is_ai=True: non salva il punteggio (restart è usato sia dall'AI che dall'umano).
    # won sarà sempre False (livello 1 < 10, non è il livello finale).

    player.resetScore()
    # Resetta il punteggio a 0 e le vite al valore iniziale (2).
    # Chiamato DOPO changeLvl() per non perdere il punteggio prima del salvataggio
    # (anche se in restart is_ai=True e il punteggio non viene salvato comunque).

    return lvl, currentlvl, won
    # Restituisce: la griglia del livello 1, il numero 1, False (non completato)