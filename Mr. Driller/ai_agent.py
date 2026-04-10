"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ai_agent.py — Rainbow DQN per Mr. Driller                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  GLOSSARIO RAPIDO:                                                           ║
║  • DQN  = Deep Q-Network: rete neurale che impara a scegliere azioni         ║
║  • Rainbow = variante avanzata di DQN con 6 miglioramenti combinati          ║
║  • policy_net  = la rete "studente" che si aggiorna ogni step                ║
║  • target_net  = la rete "insegnante" aggiornata raramente (stabilità)       ║
║  • replay buffer = "memoria" dell'agente: salva (stato,azione,reward,...)    ║
║  • prioritized replay = le esperienze "sorprendenti" vengono viste di più    ║
║  • n-step return = accumula reward su N passi futuri (visione a lungo term)  ║
║  • dueling = separa stima del valore dello stato da stima delle azioni       ║
║  • noisy net = pesi rumorosi al posto di epsilon-greedy per l'esplorazione   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT
# ─────────────────────────────────────────────────────────────────────────────

import os           # accesso al filesystem (makedirs, path.exists, ...)
import threading    # per salvare il buffer su disco in background senza bloccare il gioco
import torch        # framework principale di deep learning (tensori + GPU)
import torch.nn as nn                  # moduli per costruire reti neurali
import torch.nn.functional as F        # funzioni "stateless": relu, smooth_l1_loss, ...
import random       # numeri casuali (ε-greedy, campionamento SumTree)
import math         # sqrt e altre funzioni matematiche
import numpy as np  # array numerici veloci (CPU)
from collections import deque, namedtuple
# deque = coda a dimensione fissa: quando è piena, butta via il vecchio
# namedtuple = tupla con campi nominati, usata per la struttura Transition

# ─────────────────────────────────────────────────────────────────────────────
# COSTANTI GLOBALI
# ─────────────────────────────────────────────────────────────────────────────

# Sceglie automaticamente GPU se disponibile, altrimenti usa la CPU.
# Tutti i tensori creati con device=device finiranno lì.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clipping della reward: nessuna reward può essere < -12 o > +12.
# Stabilizza l'addestramento evitando gradienti giganteschi da reward anomale.
REWARD_CLIP   = 12.0

# Valore massimo del timer "aria": se il player è senza ossigeno per più di
# 120 frame, muore. Usato per normalizzare il timer nel vettore di stato.
AIR_TIMER_MAX = 120.0

# Penalità base per ogni tipo di azione fallita (azione eseguita ma senza spostarsi).
# Chiave = indice azione: 1=sinistra, 2=destra, 3=drill-sx, 4=drill-dx, 5=giù
# Drill laterale (3,4) penalizzato di più perché è raro che serva davvero.
_FAIL_PENALTY_BASE = {5: 1.0, 3: 0.7, 4: 0.7, 1: 0.3, 2: 0.3}

# Valore strategico per tipo di blocco: usato nel canale 0 della finestra locale.
# "end" (ossigeno) vale di più perché il gioco si vince scendendo e respirando.
# "unbreakable" vale quasi 0 perché è irrilevante: non si può rompere.
_STRATEGIC_VALUES = {
    "end":         1.0,   # blocco-fine livello: massima priorità
    "pill":        0.9,   # pillola: alta priorità
    "classic":     0.7,   # blocco normale
    "delayed":     0.6,   # blocco a scomparsa
    "solo":        0.5,   # blocco singolo
    "unbreakable": 0.1,   # muro indistruttibile: quasi irrilevante
}


# =============================================================================
# SEZIONE 1 — ESTRAZIONE DELLE FEATURE
# =============================================================================

def get_local_window_tensor(level, pX, pY, window_h=11, window_w=9):
    """
    Costruisce un tensore 6×11×9 che rappresenta la "visuale" del player.

    Ogni piano (canale) descrive un aspetto diverso della mappa:
      Canale 0 → valore strategico di ogni blocco (0.0 - 1.0)
      Canale 1 → combo di colore: quanti vicini stesso colore ha un blocco
      Canale 2 → durezza / HP / timer del blocco (resistenza fisica)
      Canale 3 → urgenza fisica: pericolo di essere schiacciato (0.0 - 1.0)
      Canale 4 → peso di prossimità: celle vicine al player pesano di più
      Canale 5 → maschera "aria": 1.0 se la cella è vuota (nessun blocco)

    Args:
        level     : griglia 2D di oggetti Block
        pX, pY    : posizione corrente del player (colonna, riga)
        window_h  : altezza della finestra visuale (default 11 righe)
        window_w  : larghezza della finestra visuale (default 9 colonne)

    Returns:
        torch.Tensor di shape [6, 11, 9] e dtype float32
    """

    # dy = quante righe sopra/sotto il player occupa la finestra.
    # Con window_h=11 → dy=5: 5 righe sopra, la riga del player, 5 sotto.
    # dx = quante colonne a sx/dx: con window_w=9 → dx=4.
    dy, dx     = window_h // 2, window_w // 2

    # Dimensioni totali della mappa di gioco (numero righe e colonne).
    rows, cols = len(level), len(level[0])

    # Assicura che le coordinate siano interi (potrebbero arrivare come float
    # se la posizione del player è interpolata graficamente).
    pX, pY     = int(pX), int(pY)

    # Crea il tensore di output: 6 canali × 11 righe × 9 colonne, tutto a zero.
    # np.float32 è richiesto da PyTorch; inizializzare a zero è importante
    # perché le celle "fuori mappa" restano 0 in tutti i canali tranne 0 e 2.
    grid = np.zeros((6, window_h, window_w), dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # CANALE 4: Peso di prossimità
    # Scopo: dire alla rete "queste celle sono vicine al player, guardaci bene"
    # ─────────────────────────────────────────────────────────────────────────

    # Array 1D [0, 1, 2, ..., 10]: indici di riga della finestra
    i_idx = np.arange(window_h, dtype=np.float32)
    # Array 1D [0, 1, 2, ..., 8]: indici di colonna della finestra
    j_idx = np.arange(window_w, dtype=np.float32)

    # rel_y[i] = distanza verticale della riga i dal centro (può essere negativa
    # se la riga è sopra il player). reshape(-1,1) crea una colonna per il broadcast.
    rel_y = (i_idx - dy).reshape(-1, 1)

    # rel_x[j] = distanza orizzontale ASSOLUTA della colonna j dal centro.
    # np.abs perché la distanza orizzontale è simmetrica.
    # reshape(1,-1) crea una riga per il broadcast con rel_y.
    rel_x = np.abs(j_idx - dx).reshape(1, -1)

    # Formula del gradiente di prossimità: 1 / (1 + distanza_euclidea_pesata)
    # L'asse X pesa 2.5×, quello Y solo 0.8×: la rete "vede" meglio in orizzontale
    # perché nel gioco spostarsi lateralmente è spesso l'azione chiave.
    # Il risultato varia da ~1.0 (centro) a ~0.05 (angoli estremi).
    grid[4] = 1.0 / (1.0 + np.sqrt((rel_y * 0.8) ** 2 + (rel_x * 2.5) ** 2))

    # ─────────────────────────────────────────────────────────────────────────
    # PRE-CALCOLO: coordinate assolute e maschere "fuori mappa"
    # ─────────────────────────────────────────────────────────────────────────

    # abs_y_arr[i] = riga REALE nel level corrispondente alla riga i della finestra.
    # Esempio: player a Y=3, dy=5 → abs_y_arr[0]=3-5=-2 (fuori mappa, sopra).
    abs_y_arr = (pY - dy + np.arange(window_h)).astype(np.int32)

    # Stesso ragionamento per le colonne.
    abs_x_arr = (pX - dx + np.arange(window_w)).astype(np.int32)

    # out_y[i] = True se la riga assoluta i è fuori dai limiti del livello.
    # Questo succede vicino ai bordi della mappa (in cima o in fondo).
    out_y = (abs_y_arr < 0) | (abs_y_arr >= rows)

    # out_x[j] = True se la colonna assoluta j è fuori dai limiti laterali.
    out_x = (abs_x_arr < 0) | (abs_x_arr >= cols)

    # ─────────────────────────────────────────────────────────────────────────
    # LOOP PRINCIPALE: riempie i canali 0, 1, 2, 3, 5 per ogni cella
    # ─────────────────────────────────────────────────────────────────────────

    for i in range(window_h):          # itera su ogni riga della finestra (0..10)
        ay = abs_y_arr[i]              # riga reale nel livello corrispondente
        ry = i - dy                    # distanza relativa: <0 = sopra player, 0 = stessa riga, >0 = sotto

        if out_y[i]:
            # La riga è fuori dalla mappa: trattala come un muro esterno.
            grid[0, i, :] = 0.1        # basso valore strategico (non è interessante)
            grid[2, i, :] = 1.0        # appare come muro duro (hp=1)
            continue                   # salta l'analisi colonna per colonna: è tutto uguale

        level_row = level[ay]          # estrae la riga del livello (ottimizzazione: una volta sola)

        for j in range(window_w):      # itera su ogni colonna della finestra (0..8)
            ax = abs_x_arr[j]          # colonna reale nel livello corrispondente
            rx = abs(j - dx)           # distanza orizzontale assoluta dal player (0 = stessa col)

            if out_x[j]:
                # La colonna è fuori mappa: bordo laterale → muro esterno
                grid[0, i, j] = 0.1
                grid[2, i, j] = 1.0
                continue

            # Recupera l'oggetto blocco dalla mappa reale
            block  = level_row[ax]
            hp     = block.hpAccess()     # punti vita del blocco (0 = aria/vuoto)
            b_type = block.typeAccess()   # stringa tipo: "classic", "pill", "unbreakable", ...

            if hp <= 0:
                # Blocco con 0 HP = spazio vuoto (aria).
                # Canale 5 = 1 significa "qui non c'è nessun blocco".
                # Tutti gli altri canali restano a 0 (inizializzati sopra).
                grid[5, i, j] = 1.0
                continue               # non serve più nessun calcolo per questa cella

            # ── Canale 0: Valore Strategico ──────────────────────────────────
            # Recupera il punteggio del tipo di blocco dalla tabella globale.
            # Se il tipo non è in tabella, usa 0.5 come default neutro.
            grid[0, i, j] = _STRATEGIC_VALUES.get(b_type, 0.5)

            # ── Canale 1: Combo di Colori ─────────────────────────────────────
            if b_type == "classic":
                # Ottiene l'ID numerico del colore (es. 0=rosso, 1=blu, ecc.)
                my_color      = block.ColorAccess()
                # Normalizza tra 0 e 1 dividendo per 5 (si assume ~5 colori distinti)
                grid[1, i, j] = my_color / 5.0

                combo = 0.0  # accumulatore bonus combo
                # Controlla i 4 vicini diretti (su, giù, sinistra, destra)
                for ny, nx in ((ay-1,ax),(ay+1,ax),(ay,ax-1),(ay,ax+1)):
                    if 0<=ny<rows and 0<=nx<cols:          # sicurezza bordi mappa
                        nb = level[ny][nx]                 # blocco vicino
                        # Conta il vicino come combo se:
                        # 1) è un blocco classico, 2) è ancora vivo (hp>0), 3) stesso colore
                        if nb.typeAccess()=="classic" and nb.hpAccess()>0 and nb.ColorAccess()==my_color:
                            combo += 0.25  # ogni vicino dello stesso colore aggiunge 0.25
                # Somma il bonus combo al valore base del colore
                # Es: colore=2, 2 vicini uguali → 2/5 + 0.5 = 0.9
                grid[1, i, j] += combo

            # ── Canale 2: Durezza / HP ────────────────────────────────────────
            if b_type == "delayed":
                # Blocco "ritardato": scompare dopo un timer, non subito
                # getattr con default lambda: fallback sicuro se il metodo non esiste
                is_active = getattr(block, 'idAcc', lambda: False)()  # il timer è partito?
                if is_active:
                    # Legge il valore del timer e lo normalizza su 2 (max timer = ~2 secondi)
                    timer_val     = getattr(block, 'getTimer', lambda: 2)()
                    grid[2, i, j] = float(timer_val) / 2.0  # → 0.0 (per scomparire) a 1.0 (appena attivato)
                else:
                    grid[2, i, j] = 1.0  # timer non ancora partito: appare solido come un blocco normale
            elif b_type == "unbreakable":
                # Blocco indistruttibile: ha HP molto alti (es. 5).
                # Si divide per 5 per riportarlo nel range [0..1], altrimenti
                # il valore 5 farebbe "esplodere" gli altri canali per confronto.
                grid[2, i, j] = float(hp) / 5.0
            else:
                # Blocchi normali (classic, pill, ecc.): HP di solito = 1.
                # Si usa direttamente senza normalizzazione.
                grid[2, i, j] = float(hp)

            # ── Canale 3: Urgenza Fisica ──────────────────────────────────────
            # Risponde alla domanda: "Questo blocco mi schiaccia se non mi muovo?"
            is_falling = getattr(block, 'isFalling', lambda: False)()  # sta cadendo ora?
            is_shaking = getattr(block, 'isShaking', lambda: False)()  # sta per cadere (vibrazione)?

            # col_weight: se il blocco è nella colonna del player → peso 1.0 (max pericolo).
            # Se è a distanza rx > 0 (colonna adiacente), il pericolo scende.
            # Formula: 1/(1 + rx*0.7) → rx=0→1.0, rx=1→0.59, rx=2→0.41
            col_weight = 1.0 / (1.0 + rx * 0.7)

            urgency = 0.0  # default: nessun pericolo

            if is_falling:
                # Blocco in caduta libera = pericolo IMMEDIATO.
                # Ma solo se è sopra o alla stessa riga del player (ry <= 0).
                # Se è sotto (ry > 0), non può cadere addosso al player → nessuna urgenza.
                if ry <= 0:
                    urgency = 0.95 * col_weight  # 0.95 = quasi certezza di pericolo

            elif is_shaking:
                # Blocco che vibra = sta per cadere, ma non è ancora in caduta.
                # Pericolo solo se è STRETTAMENTE sopra (ry < 0), non sulla stessa riga.
                if ry < 0:
                    urgency = 0.50 * col_weight  # 0.50 = pericolo medio

            grid[3, i, j] = urgency  # salva nel canale 3

    # ─────────────────────────────────────────────────────────────────────────
    # POST-PROCESSING: propagazione della traiettoria di caduta (canale 3)
    # ─────────────────────────────────────────────────────────────────────────
    # Problema: un blocco in caduta si sposterà VERSO IL BASSO nelle celle
    # di aria sottostanti. Ma il loop sopra ha messo l'urgenza solo sulla
    # cella del blocco stesso, non su dove atterrerà.
    # Questo secondo loop "dipinge" il pericolo nelle celle d'aria sotto
    # ogni blocco pericoloso, così la rete vede dove il blocco arriverà.

    for j in range(window_w):               # per ogni colonna della finestra
        rx_col = abs(j - dx)               # distanza colonna dal player
        if rx_col > 2:
            continue  # ottimizzazione: la scia interessa solo le 2 colonne vicine al player

        for i in range(window_h - 1):       # per ogni riga (tranne l'ultima)
            cell_urgency = grid[3, i, j]   # urgenza della cella corrente
            if cell_urgency < 0.80:
                continue   # non abbastanza pericoloso da proiettare la scia
            if grid[5, i, j] == 1.0:
                continue   # la cella corrente è aria: l'aria non "cade"

            # Ok: questa cella ha un blocco che cade con alta urgenza.
            # Ora propaghiamo il pericolo verso il basso.
            for di in range(1, window_h - i):  # di = distanza in righe verso il basso
                ti = i + di                    # riga target (sotto il blocco pericoloso)
                if ti >= window_h:
                    break  # usciti dalla finestra visiva

                if grid[5, ti, j] == 1.0:
                    # La cella sotto è aria: il blocco cadente arriverà qui.
                    # L'urgenza decresce del 15% per ogni riga (0.85^di):
                    # la cella immediatamente sotto ha urgenza alta, quelle
                    # più lontane progressivamente meno.
                    traj_u = cell_urgency * (0.85 ** di)
                    if grid[3, ti, j] < traj_u:
                        grid[3, ti, j] = traj_u  # aggiorna solo se più urgente di prima
                else:
                    break  # la scia si ferma al primo blocco solido incontrato

    # Converte l'array numpy in tensore PyTorch float32 (formato richiesto dalla rete neurale)
    return torch.from_numpy(grid).float()


def get_internal_state_vector(player, total_level_rows, total_level_cols,
                               level_data, last_action_idx, n_actions):
    """
    Costruisce un vettore 1D di 32 valori float che descrivono lo stato
    "scalare" del giocatore (non visuale): ossigeno, vite, punteggio, ecc.

    Questo vettore viene passato al branch scalare della rete neurale
    (fc_vars) separatamente dalla finestra visuale, poi i due rami vengono
    concatenati prima delle layer finali.

    Returns:
        torch.Tensor di shape [32] e dtype float32, su device globale
    """
    state = []  # lista che costruiamo incrementalmente

    # Recupera posizione player e ossigeno attuale
    pY, pX = player.posAcc()   # nota: posAcc() restituisce (Y, X), non (X, Y)
    oxy    = player.oxyAcc()   # ossigeno: 0..100

    # ── Slot 1-6: Feature base ────────────────────────────────────────────────
    state += [
        pX / max(total_level_cols, 1),   # [0] posizione X normalizzata (0=sx, 1=dx)
        pY / max(total_level_rows, 1),   # [1] profondità normalizzata (0=top, 1=bottom)
        oxy / 100.0,                     # [2] ossigeno normalizzato (0=morto, 1=pieno)
        max(0, player.livesAcc()) / 3.0, # [3] vite normalizzate (max 3 vite → 0,0.33,0.67,1)
        min(player.scoreAcc() / 10000.0, 1.0),  # [4] punteggio (capped a 10000)
        min(pY / 100.0, 1.0),            # [5] profondità assoluta (capped a 100)
    ]

    # ── Slot 7-8: Soglie ossigeno (segnali di allerta) ────────────────────────
    state += [
        1.0 if oxy < 20 else 0.0,  # [6] flag "ossigeno critico" (< 20%)
        1.0 if oxy < 50 else 0.0,  # [7] flag "ossigeno basso" (< 50%)
    ]

    # ── Slot 9-12: Direzione animazione (one-hot) ─────────────────────────────
    # imgIndAcc() restituisce un indice 0..3 che indica la direzione corrente
    # del player: 0=sù, 1=dx, 2=giù, 3=sx (dipende dall'implementazione).
    # Si codifica come one-hot: solo uno dei 4 slot è 1, gli altri 0.
    d  = [0.0] * 4          # vettore di 4 zeri
    di = player.imgIndAcc() # indice direzione
    if 0 <= di < 4:
        d[di] = 1.0         # accende il bit corrispondente alla direzione
    state += d

    # ── Slot 13-18: Ultima azione (one-hot) ──────────────────────────────────
    # n_actions = numero totale di azioni possibili (es. 6: idle/sx/dx/drill-sx/drill-dx/giù)
    # Rappresentare l'ultima azione aiuta la rete a evitare comportamenti ripetitivi.
    a = [0.0] * n_actions
    if 0 <= last_action_idx < n_actions:
        a[last_action_idx] = 1.0  # accende il bit dell'ultima azione compiuta
    state += a

    # ── Slot 19-22: HP delle celle adiacenti (4 direzioni) ───────────────────
    # Dice alla rete quanto "duro" è il terreno immediatamente intorno al player.
    # Utile per capire se può muoversi o è bloccato.
    sur = [0.0] * 4  # [sx, dx, giù, su]
    if isinstance(level_data, list) and level_data:
        try:
            if pX > 0:                  sur[0] = min(level_data[pY][pX-1].hpAccess(), 1.0)  # sinistra
            if pX < total_level_cols-1: sur[1] = min(level_data[pY][pX+1].hpAccess(), 1.0)  # destra
            if pY < total_level_rows-1: sur[2] = min(level_data[pY+1][pX].hpAccess(), 1.0)  # sotto
            if pY > 0:                  sur[3] = min(level_data[pY-1][pX].hpAccess(), 1.0)  # sopra
        except Exception:
            pass  # se fallisce (bordi mappa o bug), lascia i valori a 0 silenziosamente
    state += sur

    # ── Slot 23-26: Pillole adiacenti (4 direzioni) ───────────────────────────
    # Le pillole sono l'ossigeno del gioco: la rete deve imparare ad avvicinarsi.
    # Ogni slot vale 1.0 se c'è una pillola in quella direzione, 0.0 altrimenti.
    pill = [0.0] * 4  # [sx, dx, giù, su]
    if isinstance(level_data, list) and level_data:
        try:
            if pX > 0 and level_data[pY][pX-1].typeAccess() == "pill":                  pill[0] = 1.0
            if pX < total_level_cols-1 and level_data[pY][pX+1].typeAccess() == "pill": pill[1] = 1.0
            if pY < total_level_rows-1 and level_data[pY+1][pX].typeAccess() == "pill": pill[2] = 1.0
            if pY > 0 and level_data[pY-1][pX].typeAccess() == "pill":                  pill[3] = 1.0
        except Exception:
            pass
    state += pill

    # ── Slot 27-28: Stato caduta e timer aria ─────────────────────────────────
    state += [
        1.0 if player.fallAcc() else 0.0,             # [26] il player sta cadendo?
        min(player.airTimerAcc() / AIR_TIMER_MAX, 1.0) # [27] quanto tempo ha ancora aria (0=pieno, 1=morto)
    ]

    # ── Slot 29: Instabilità sotto il player [PHYS-F] ────────────────────────
    # Se il blocco su cui stai PER APPOGGIARTI è instabile, scappa!
    # Controlla anche 2 blocchi sotto (caso indiretto).
    below_unstable = 0.0
    if isinstance(level_data, list) and level_data:
        try:
            if pY + 1 < total_level_rows:
                b1 = level_data[pY + 1][pX]  # blocco immediatamente sotto
                if (getattr(b1, 'isFalling', lambda: False)() or
                        getattr(b1, 'isShaking', lambda: False)()):
                    below_unstable = 1.0      # pericolo immediato: il blocco sotto cade
                elif b1.hpAccess() > 0 and pY + 2 < total_level_rows:
                    b2 = level_data[pY + 2][pX]  # 2 blocchi sotto
                    if (getattr(b2, 'isFalling', lambda: False)() or
                            getattr(b2, 'isShaking', lambda: False)()):
                        below_unstable = 0.5  # pericolo indiretto (meno urgente)
        except Exception:
            pass
    state.append(below_unstable)

    # ── Slot 30: Blocco in caduta sopra (con distanza) [PHYS-E] ──────────────
    # Cerca blocchi pericolosi nelle 5 righe sopra il player.
    # Più è vicino, più è urgente: 1/k (distanza 1→1.0, 2→0.5, 3→0.33, ecc.)
    above_falling = 0.0
    if isinstance(level_data, list) and level_data:
        try:
            for k in range(1, 6):    # k = numero di righe sopra il player
                ty = pY - k          # riga assoluta da controllare
                if ty < 0:
                    break            # uscito dalla mappa (siamo già in cima)
                blk = level_data[ty][pX]
                if blk.hpAccess() <= 0:
                    continue         # cella vuota: il blocco sopra potrebbe essere ancora più su
                if getattr(blk, 'isFalling', lambda: False)():
                    above_falling = 1.0 / k   # caduta in atto: urgenza = 1/distanza
                    break
                elif getattr(blk, 'isShaking', lambda: False)():
                    above_falling = 0.5 / k   # vibrazione: urgenza dimezzata
                    break
                elif blk.hpAccess() > 1:
                    break            # blocco solido e duro: fa da scudo, stop ricerca
        except Exception:
            pass
    state.append(above_falling)

    # ── Slot 31: Contatore blocchi pericolosi in colonna sopra [PHYS-H] ───────
    # Quanti blocchi CONSECUTIVI che cadono/vibrano ci sono sopra il player?
    # Più sono, più la situazione è grave.
    column_danger_count = 0
    if isinstance(level_data, list) and level_data:
        try:
            for k in range(1, 6):
                ty = pY - k
                if ty < 0:
                    break
                blk = level_data[ty][pX]
                if blk.hpAccess() <= 0:
                    continue  # aria: salta, continua a cercare più su
                if (getattr(blk, 'isFalling', lambda: False)() or
                        getattr(blk, 'isShaking', lambda: False)()):
                    column_danger_count += 1  # conta questo blocco pericoloso
                else:
                    break     # blocco stabile: la catena di pericolo si interrompe
        except Exception:
            pass
    # Normalizza su 5 (max blocchi contati) per tenerlo nel range [0..1]
    state.append(min(column_danger_count / 5.0, 1.0))

    # ── Slot 32: Pericolo nelle colonne adiacenti [PHYS-H] ───────────────────
    # Non solo la tua colonna: anche a sinistra (-1) e a destra (+1).
    # Un blocco in caduta nella colonna vicina può colpirti se ti muovi.
    adjacent_col_danger = 0.0
    if isinstance(level_data, list) and level_data:
        try:
            for ddx in (-1, 0, 1):    # -1=sinistra, 0=stessa colonna, 1=destra
                tx = pX + ddx
                if not (0 <= tx < total_level_cols):
                    continue          # colonna fuori mappa: salta
                for k in range(1, 5):
                    ty = pY - k
                    if ty < 0:
                        break
                    blk = level_data[ty][tx]
                    if blk.hpAccess() <= 0:
                        continue
                    if getattr(blk, 'isFalling', lambda: False)():
                        # contrib diminuisce con: distanza verticale k E distanza laterale abs(ddx)
                        # (1 + abs(ddx)): stessa col→1, col vicina→2 → pericolo dimezzato
                        contrib = 1.0 / (k * (1 + abs(ddx)))
                        adjacent_col_danger = max(adjacent_col_danger, contrib)
                        break
                    elif getattr(blk, 'isShaking', lambda: False)():
                        contrib = 0.4 / (k * (1 + abs(ddx)))
                        adjacent_col_danger = max(adjacent_col_danger, contrib)
                        break
                    else:
                        break  # blocco stabile → interrompe la ricerca su questa colonna
        except Exception:
            pass
    state.append(min(adjacent_col_danger, 1.0))

    # Padding di sicurezza: se per qualche motivo il vettore è più corto di 32,
    # aggiunge zeri fino a 32. Se è più lungo, tronca a 32.
    while len(state) < 32:
        state.append(0.0)

    # Crea il tensore finale su device (CPU o GPU) con dtype float32
    return torch.tensor(state[:32], dtype=torch.float32, device=device)


# =============================================================================
# SEZIONE 2 — NOISY LINEAR LAYER
# =============================================================================

class NoisyLinear(nn.Module):
    """
    Layer lineare con rumore "apprendibile" nei pesi (NoisyNets, Fortunato et al. 2017).

    Perché? L'esplorazione classica usa ε-greedy: con probabilità ε fa un'azione
    casuale. Il problema è che ε è globale: esplora allo stesso modo in ogni stato.

    NoisyLinear invece aggiunge rumore DIRETTAMENTE ai pesi della rete.
    Ogni forward pass produce un Q-value leggermente diverso, forzando
    l'agente ad esplorare naturalmente, di più nei stati incerti
    e di meno in quelli dove ha già imparato bene.

    Come funziona:
      peso_effettivo = peso_mu + peso_sigma * epsilon
      dove epsilon è un vettore di rumore campionato ad ogni step.
      mu e sigma sono parametri APPRENDIBILI: la rete decide quanta incertezza avere.
    """

    def __init__(self, in_features, out_features, std_init=0.5):
        """
        Args:
            in_features  : dimensione input
            out_features : dimensione output
            std_init     : deviazione standard iniziale del rumore (default 0.5)
        """
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init

        # Parametri apprendibili del peso: media (mu) e ampiezza del rumore (sigma)
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        # Il rumore epsilon NON è un parametro (non si aggiorna con backprop),
        # ma è registrato come buffer per essere spostato automaticamente su GPU.
        self.register_buffer('weight_epsilon', torch.zeros(out_features, in_features))

        # Stesso schema per il bias
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))

        # Inizializzazione dei parametri:
        # mu_r = 1/sqrt(in_features): range di inizializzazione uniforme standard
        mu_r = 1.0 / math.sqrt(in_features)
        self.weight_mu.data.uniform_(-mu_r, mu_r)         # pesi iniziali casuali uniforme
        self.weight_sigma.data.fill_(std_init / math.sqrt(in_features))  # sigma piccolo
        self.bias_mu.data.uniform_(-mu_r, mu_r)
        self.bias_sigma.data.fill_(std_init / math.sqrt(out_features))

        self.sample_noise()  # campiona il primo batch di rumore

    @staticmethod
    def _f(x):
        """
        Trasformazione fattorizzata del rumore: sign(x) * sqrt(|x|).
        Usata nella "factorized noise" di NoisyNets per ridurre la complessità
        computazionale: invece di n*m numeri casuali, bastano n+m.
        Questa funzione mantiene il segno ma "schiaccia" i valori estremi.
        """
        return x.sign() * x.abs().sqrt()

    def sample_noise(self):
        """
        Ricampiona i vettori di rumore ε_i (input) e ε_j (output).
        Il rumore del peso[j][i] è dato dal prodotto esterno: ε_j ⊗ ε_i.
        Questo approccio "fattorizzato" richiede solo n+m campionamenti
        anziché n*m, riducendo il costo computazionale.
        """
        dev = self.weight_mu.device              # assicura che il rumore sia sulla stessa device
        ei  = self._f(torch.randn(self.in_features,  device=dev))   # rumore input: vettore n
        ej  = self._f(torch.randn(self.out_features, device=dev))   # rumore output: vettore m
        self.weight_epsilon.set_(ej.outer(ei))   # prodotto esterno m×n: il rumore del peso
        self.bias_epsilon.set_(ej)               # il rumore del bias usa solo ε_j

    def forward(self, x):
        """
        Forward pass con rumore (training) o senza rumore (eval/inference).
        
        In training: usa peso = mu + sigma * epsilon per l'esplorazione.
        In eval:     usa solo peso = mu (comportamento deterministico pulito).
        """
        if self.training:
            # Peso effettivo = media appresa + rumore apprendibile * epsilon campionato
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu   + self.bias_sigma   * self.bias_epsilon
            )
        # Modalità valutazione: usa solo i pesi "puliti" senza rumore
        return F.linear(x, self.weight_mu, self.bias_mu)


# =============================================================================
# SEZIONE 3 — ARCHITETTURA DELLA RETE NEURALE
# =============================================================================

class ResBlock(nn.Module):
    """
    Blocco residuale: un modulo fondamentale introdotto in ResNet (He et al. 2015).

    Struttura: output = ReLU(F(x) + x)
    dove F(x) = due Conv2d con BatchNorm tra loro.

    Perché? Senza il collegamento "skip" (+x), reti profonde hanno il problema
    del "vanishing gradient": i gradienti diventano minuscoli e la rete smette
    di imparare. Il collegamento residuale crea un "autoroute" che permette
    ai gradienti di scorrere direttamente all'indietro senza degradarsi.

    Risultato pratico: si possono impilare molti blocchi senza perdere performance.
    """

    def __init__(self, ch):
        """
        Args:
            ch : numero di canali (stessa dimensione in input e output per il residual)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),  # conv 3×3, padding=1 mantiene la dimensione
            nn.BatchNorm2d(ch),                            # normalizza le attivazioni per canale
            nn.ReLU(True),                                 # inplace=True risparmiava memoria (deprecato ma funziona)
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),  # seconda conv: raffina le feature
            nn.BatchNorm2d(ch)                             # seconda normalizzazione
        )
        self.act = nn.ReLU(True)  # attivazione finale (applicata DOPO la somma residuale)

    def forward(self, x):
        # Applica la trasformazione F(x), poi SOMMA l'input originale x (residuo).
        # Infine applica ReLU al risultato della somma.
        return self.act(self.net(x) + x)


class SpatialAttention(nn.Module):
    """
    Meccanismo di attenzione spaziale: la rete impara a "focalizzarsi"
    sulle zone della mappa più rilevanti per la decisione.

    Come funziona:
      1) Applica una convoluzione 1×1 per ridurre a 1 canale (mappa scalare)
      2) Applica sigmoid: valori tra 0 e 1 (quanto "attention" per ogni posizione)
      3) Moltiplica la feature map originale per questa maschera di attenzione

    Risultato: le zone "attenzionate" vengono amplificate, le altre attenuate.
    È un modo economico per dare alla rete la capacità di ignorare zone irrilevanti.
    """

    def __init__(self, ch):
        super().__init__()
        # Convolution 1×1: un "peso" per canale, trasforma ch canali in 1 canale scalare
        self.conv = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        # self.conv(x): shape [B, 1, H, W] → maschera di attenzione non normalizzata
        # torch.sigmoid: porta i valori in [0, 1]
        # x * ...: broadcasting: moltiplica ogni canale per la stessa maschera spaziale
        return x * torch.sigmoid(self.conv(x))


class DrillerDuelingDQN(nn.Module):
    """
    Rete neurale principale: Dueling DQN con CNN, ResNet, Attenzione e NoisyNets.

    Architettura:
      ┌─────────────────────────┐   ┌──────────────────┐
      │ grid [B, 6, 11, 9]      │   │ vars [B, 32]     │
      │   → Conv2d(6→64, 3×3)  │   │   → Linear(128)  │
      │   → ResBlock(64)        │   │   → Linear(256)  │
      │   → ResBlock(64)        │   └────────┬─────────┘
      │   → SpatialAttention    │            │
      │   → Flatten → [B, 5376] │            │
      └──────────────┬──────────┘            │
                     └──────────┬────────────┘
                          concat [B, 5376+256=5632]
                                │
                    ┌───────────┴───────────┐
                    │                       │
              VALUE branch             ADVANTAGE branch
              NoisyLinear(512)         NoisyLinear(512)
              NoisyLinear(1)           NoisyLinear(n_actions)
                    │                       │
                    └───────────┬───────────┘
                          Q = V + (A - mean(A))
                          [B, n_actions]

    Perché Dueling?
      Separare il valore dello STATO (V) dal vantaggio di ogni AZIONE (A)
      permette alla rete di imparare meglio: spesso il valore dello stato
      è identico per tutte le azioni, quindi è più efficiente apprenderlo
      separatamente piuttosto che ridondantemente in ogni ramo azione.
    """

    def __init__(self, input_shape, n_actions, state_dim):
        """
        Args:
            input_shape : tuple (canali, altezza, larghezza) della finestra visuale → (6, 11, 9)
            n_actions   : numero di azioni possibili (es. 6)
            state_dim   : dimensione del vettore scalare di stato → 32
        """
        super().__init__()
        c, h, w = input_shape  # unpack: c=6, h=11, w=9

        # ── Branch CNN (visuale) ──────────────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 64, 3, padding=1, bias=False),  # 6 canali → 64 feature maps, kernel 3×3
            nn.BatchNorm2d(64),                           # normalizzazione batch
            nn.ReLU(True)                                 # attivazione non lineare
        )
        self.res1 = ResBlock(64)           # primo blocco residuale: 64→64 canali
        self.res2 = ResBlock(64)           # secondo blocco residuale: 64→64 canali
        self.attn = SpatialAttention(64)  # attenzione spaziale: "dove guardare"

        # Dimensione dell'output piatto dopo CNN: 64 canali × 11 righe × 9 colonne
        flat = 64 * h * w  # = 64 × 11 × 9 = 6336... aggiornato: h=11, w=9

        # ── Branch scalare (state_vars) ───────────────────────────────────────
        self.fc_vars = nn.Sequential(
            nn.Linear(state_dim, 128),  # 32 → 128: espande le feature scalari
            nn.ReLU(True),
            nn.Linear(128, 256),        # 128 → 256: ultima rappresentazione scalare
            nn.ReLU(True)
        )

        # Dimensione totale dopo concatenazione: flat + 256
        combined = flat + 256

        # ── Dueling: ramo Value V(s) ──────────────────────────────────────────
        # Stima un singolo valore scalare: "quanto è buono essere in questo stato?"
        self.val = nn.Sequential(
            NoisyLinear(combined, 512),  # compressione con rumore
            nn.ReLU(True),
            NoisyLinear(512, 1)          # output: un solo numero = V(s)
        )

        # ── Dueling: ramo Advantage A(s,a) ────────────────────────────────────
        # Stima il vantaggio di ogni azione rispetto alla media.
        self.adv = nn.Sequential(
            NoisyLinear(combined, 512),
            nn.ReLU(True),
            NoisyLinear(512, n_actions)  # output: n_actions numeri = A(s,a) per ogni azione
        )

        # Inizializzazione dei pesi con Kaiming Normal (ottimale per ReLU):
        # evita la saturazione iniziale delle attivazioni.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # bias inizializzati a 0

    def forward(self, grid, vars):
        """
        Forward pass completo.

        Args:
            grid : tensor [B, 6, 11, 9] — la finestra visuale
            vars : tensor [B, 32]       — il vettore scalare di stato

        Returns:
            q_values : tensor [B, n_actions] — Q-value per ogni azione
        """
        # 1. Passa la griglia visuale attraverso CNN + ResBlocks + Attenzione
        #    poi "appiattisce" il tensore 3D in un vettore 1D per ogni batch.
        x = self.attn(self.res2(self.res1(self.cnn(grid)))).flatten(1)
        # x.shape = [B, flat] = [B, 6336]

        # 2. Concatena la rappresentazione visuale con le feature scalari elaborat
        c = torch.cat([x, self.fc_vars(vars)], 1)
        # c.shape = [B, flat+256]

        # 3. I due rami calcolano V(s) e A(s,a) separatamente
        val, adv = self.val(c), self.adv(c)
        # val.shape = [B, 1]       — valore dello stato
        # adv.shape = [B, n_actions] — vantaggio per ogni azione

        # 4. Combina con la formula dueling:
        #    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        #    Sottrarre la media rende l'aggiornamento dei pesi più stabile:
        #    garantisce che il contributo di V e A siano identificabili.
        return val + (adv - adv.mean(1, keepdim=True))
        # output.shape = [B, n_actions]

    def sample_noise(self):
        """Ricampiona il rumore in tutti i NoisyLinear della rete."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample_noise()


# =============================================================================
# SEZIONE 4 — PRIORITIZED REPLAY MEMORY + PERSISTENZA BUFFER
# =============================================================================

class SumTree:
    """
    Struttura dati ad albero binario per il Prioritized Experience Replay.

    Il problema: nel replay buffer vogliamo estrarre le esperienze con
    probabilità proporzionale alla loro priorità (TD-error), ma farlo
    naïvamente richiederebbe O(n) per ogni campionamento.

    SumTree lo fa in O(log n):
      - I nodi foglia contengono le priorità delle singole esperienze.
      - Ogni nodo interno contiene la SOMMA delle priorità dei suoi figli.
      - Il nodo radice contiene la SOMMA TOTALE di tutte le priorità.
      - Per campionare proporzionalmente: genera un numero casuale s < total,
        poi scendi nell'albero finché non trovi il segmento corretto.

    Layout dell'array:
      tree[0] = nodo radice (somma totale)
      tree[1], tree[2] = figli della radice
      tree[cap-1 .. 2*cap-2] = nodi foglia (le priorità delle esperienze)
      data[0 .. cap-1] = le esperienze (Transition) corrispondenti alle foglie
    """

    def __init__(self, cap):
        """
        Args:
            cap : capacità massima del buffer (numero di esperienze)
        """
        self.cap  = cap
        # L'albero ha 2*cap-1 nodi totali: cap foglie + cap-1 nodi interni
        self.tree = np.zeros(2 * cap - 1, dtype=np.float64)
        # Array delle esperienze: dtype=object perché contengono tensori PyTorch
        self.data = np.empty(cap, dtype=object)
        self.wr   = 0  # write pointer: dove scrivere la prossima esperienza (circolare)
        self.n    = 0  # numero di esperienze attualmente nel buffer

    def _prop(self, idx, delta):
        """
        Propaga una variazione delta verso la radice aggiornando tutti i nodi padre.
        Indispensabile dopo ogni modifica di una foglia per mantenere le somme coerenti.

        Args:
            idx   : indice del nodo foglia appena modificato
            delta : variazione da propagare (p_nuova - p_vecchia)
        """
        while idx > 0:
            p = (idx - 1) >> 1  # indice del padre: (idx-1) // 2 (bitshift = divisione per 2)
            self.tree[p] += delta  # aggiorna il padre con la variazione
            idx = p                # sali al padre e continua

    def _get(self, idx, s):
        """
        Scende nell'albero per trovare il segmento che contiene il valore s.
        È il cuore del campionamento proporzionale alla priorità.

        Algoritmo:
          - A ogni nodo interno, controlla il figlio sinistro.
          - Se s <= figlio_sx: scendi a sinistra.
          - Altrimenti: sottrai figlio_sx da s e scendi a destra.
          - Quando arrivi a una foglia, hai trovato l'esperienza cercata.

        Args:
            idx : nodo di partenza (di solito 0 = radice)
            s   : valore casuale da cercare (0 <= s < total)

        Returns:
            indice della foglia trovata
        """
        while True:
            l = 2 * idx + 1  # figlio sinistro
            if l >= len(self.tree):
                return idx   # siamo a una foglia: trovato!
            if s <= self.tree[l]:
                idx = l      # scendi a sinistra
            else:
                s -= self.tree[l]  # sottrai il sottoalbero sx
                idx = l + 1        # scendi a destra

    @property
    def total(self):
        """Somma totale di tutte le priorità (= valore del nodo radice)."""
        return float(self.tree[0])

    def add(self, p, data):
        """
        Aggiunge una nuova esperienza con priorità p.
        Il buffer è circolare: quando è pieno, sovrascrive le esperienze più vecchie.

        Args:
            p    : priorità dell'esperienza (solitamente max_p per i nuovi inserimenti)
            data : l'esperienza da salvare (un Transition namedtuple)
        """
        # L'indice nell'albero corrispondente alla posizione wr nel data array
        i = self.wr + self.cap - 1  # le foglie iniziano da indice cap-1
        self.data[self.wr] = data   # salva l'esperienza nell'array dati
        self.update(i, p)           # aggiorna la priorità e propaga verso la radice
        self.wr = (self.wr + 1) % self.cap  # avanza il puntatore di scrittura (circolare)
        self.n  = min(self.n + 1, self.cap) # aggiorna il contatore (satura a cap)

    def update(self, idx, p):
        """
        Aggiorna la priorità di un nodo e propaga la variazione verso la radice.

        Args:
            idx : indice nell'albero (nodo foglia)
            p   : nuova priorità
        """
        d = p - self.tree[idx]  # calcola la variazione rispetto alla priorità precedente
        self.tree[idx] = p      # aggiorna la foglia
        self._prop(idx, d)      # propaga la variazione verso la radice

    def get(self, s):
        """
        Campiona un'esperienza con probabilità proporzionale alla sua priorità.

        Args:
            s : numero casuale nell'intervallo [0, total)

        Returns:
            (idx, priorità, experience): indice albero, priorità, esperienza campionata
        """
        # Clamp s per sicurezza numerica (evita problemi floating point ai bordi)
        s   = max(0.0, min(float(s), self.total - 1e-8))
        idx = self._get(0, s)  # scende nell'albero per trovare la foglia giusta
        # idx - (cap-1) converte indice albero → indice dati
        return idx, self.tree[idx], self.data[idx - self.cap + 1]


# Namedtuple per una singola transizione (s, a, s', r, done).
# Usare un namedtuple anziché una lista aumenta la leggibilità e l'efficienza.
Transition = namedtuple('Transition',
    ('state_grid',       # finestra visuale dello stato corrente
     'state_vars',       # vettore scalare dello stato corrente
     'action',           # azione intrapresa
     'next_state_grid',  # finestra visuale dello stato successivo
     'next_state_vars',  # vettore scalare dello stato successivo
     'reward',           # reward ricevuta
     'done'))            # flag terminale (True se episodio finito)

# Chiavi dei file numpy che compongono il buffer salvato su disco
_BUFFER_KEYS = ('grids', 'vars', 'actions', 'next_grids', 'next_vars',
                'rewards', 'dones', 'priorities', 'meta')


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay (PER) — Schaul et al. 2015.

    In un replay buffer uniforme, tutte le esperienze vengono campionate
    con uguale probabilità. PER invece assegna priorità più alte alle
    esperienze "sorprendenti" (alto TD-error), così la rete le rivede
    più spesso e impara più velocemente dagli errori grandi.

    Corregge il bias introdotto (le esperienze più prioritarie vengono
    viste troppo) usando i Importance Sampling Weights (is_w):
    p(i) = (priorità_i / somma_totale)^alpha
    w(i) = (1 / N*p(i))^beta

    man mano che il training avanza, beta → 1.0 (correzione completa del bias).
    """

    def __init__(self, cap, alpha=0.4, beta_start=0.4, beta_frames=200_000, eps=1e-5):
        """
        Args:
            cap         : dimensione massima del buffer
            alpha       : esponente di priorità (0=uniforme, 1=massima priorità)
            beta_start  : valore iniziale di beta per la correzione IS
            beta_frames : numero di frame per portare beta da beta_start a 1.0
            eps         : epsilon aggiunto alla priorità per evitare priorità zero
        """
        self.alpha       = alpha
        self.eps         = eps
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.frame       = 1     # contatore globale di frame (per lo scheduling di beta)
        self.tree        = SumTree(cap)  # struttura dati per il campionamento efficiente
        self._max_p      = 1.0   # priorità massima vista finora (nuovi inserimenti usano questa)

    def _beta(self):
        """
        Calcola il beta corrente: cresce linearmente da beta_start a 1.0.
        min(1.0, ...) evita che superi 1.0 dopo beta_frames frame.
        """
        return min(1.0, self.beta_start + (1 - self.beta_start) * self.frame / self.beta_frames)

    def push(self, *args):
        """
        Aggiunge una nuova transizione al buffer con priorità massima.
        Usare la max priorità garantisce che ogni esperienza venga vista almeno una volta.
        """
        # _max_p**alpha: la priorità viene elevata ad alpha prima di inserirla nel SumTree
        self.tree.add(self._max_p ** self.alpha, Transition(*args))
        self.frame += 1  # incrementa il contatore globale

    def sample(self, bs):
        """
        Campiona bs transizioni con priorità proporzionale.

        Divide l'intervallo [0, total] in bs segmenti e campiona uno per segmento
        (stratified sampling): garantisce diversità nel batch evitando che
        poche esperienze ad alta priorità dominino completamente.

        Returns:
            trans   : lista di Transition
            idxs    : lista di indici SumTree (per aggiornare le priorità dopo)
            is_w    : tensor [bs] dei pesi importance sampling
        """
        if self.tree.total <= 0:
            raise RuntimeError("SumTree vuota.")
        idxs, prios, trans = [], [], []
        seg = self.tree.total / bs  # larghezza di ogni segmento

        for i in range(bs):
            # Campiona uniformemente in questo segmento
            s = random.uniform(seg * i, seg * (i + 1))
            idx, p, t = self.tree.get(s)
            if t is None:  # fallback se la cella è vuota (non dovrebbe succedere)
                idx, p, t = self.tree.get(random.uniform(0, self.tree.total))
            idxs.append(idx)
            prios.append(p)
            trans.append(t)

        # Calcola i pesi IS:
        # probs[i] = p_i / total (probabilità normalizzata)
        probs   = np.array(prios, dtype=np.float64) / (self.tree.total + 1e-8)
        # weights[i] = (N * probs[i])^(-beta): esperienze ad alta probabilità hanno peso minore
        weights = (self.tree.n * probs) ** (-self._beta())
        # Normalizza dividendo per il peso massimo: stabilizza i gradienti
        weights /= weights.max()

        return trans, idxs, torch.tensor(weights, dtype=torch.float32, device=device)

    def update_priorities(self, idxs, errs):
        """
        Aggiorna le priorità delle esperienze campionate con i nuovi TD-error.
        Viene chiamato dopo optimize_model() con i TD-error calcolati.

        Args:
            idxs : indici SumTree da aggiornare
            errs : array di TD-error (|Q_target - Q_pred|) per ogni esperienza
        """
        for i, e in zip(idxs, errs):
            e_clamped = min(abs(float(e)), 10.0)    # clamp a 10 per stabilità
            p = (e_clamped + self.eps) ** self.alpha  # priorità = (|td| + eps)^alpha
            self._max_p = max(self._max_p, p)         # aggiorna la max priorità globale
            self.tree.update(i, p)                     # aggiorna il SumTree

    def __len__(self):
        """Restituisce il numero di esperienze attualmente nel buffer."""
        return self.tree.n

    def _snapshot_to_numpy(self):
        """
        Serializza tutte le transizioni del buffer in array numpy per il salvataggio.
        
        Crea un dizionario di array compatibili con np.save() per la persistenza.
        'meta' salva i metadati: puntatore di scrittura, dimensione, frame count, max_p.
        """
        n   = self.tree.n
        cap = self.tree.cap
        if n == 0:
            return None  # buffer vuoto: niente da salvare

        # Ricostruisce gli indici validi nell'ordine corretto (dal più vecchio al più recente)
        valid_idx = [(self.tree.wr - n + i) % cap for i in range(n)]

        # Liste per accumulare i dati da ogni Transition
        grids, vars_, actions           = [], [], []
        next_grids, next_vars           = [], []
        rewards, dones, prios           = [], [], []

        for idx in valid_idx:
            t = self.tree.data[idx]
            if t is None:
                continue  # slot non inizializzato: salta
            # .cpu().numpy(): porta il tensore dalla GPU alla CPU e converte in numpy
            grids.append(t.state_grid.cpu().numpy())
            vars_.append(t.state_vars.cpu().numpy())
            actions.append(t.action.cpu().numpy())
            next_grids.append(t.next_state_grid.cpu().numpy())
            next_vars.append(t.next_state_vars.cpu().numpy())
            rewards.append(t.reward.cpu().numpy())
            dones.append(t.done.cpu().numpy())
            # La priorità è nel nodo foglia dell'albero: indice = idx + cap - 1
            prios.append(float(self.tree.tree[idx + cap - 1]))

        if not grids:
            return None

        return dict(
            grids      = np.array(grids,      dtype=np.float32),
            vars       = np.array(vars_,      dtype=np.float32),
            actions    = np.array(actions,    dtype=np.int64),
            next_grids = np.array(next_grids, dtype=np.float32),
            next_vars  = np.array(next_vars,  dtype=np.float32),
            rewards    = np.array(rewards,    dtype=np.float32),
            dones      = np.array(dones,      dtype=bool),
            priorities = np.array(prios,      dtype=np.float64),
            # meta: 4 scalari per ripristinare lo stato del buffer
            meta       = np.array([self.tree.wr, n, self.frame, self._max_p]),
        )

    def save_async(self, buffer_dir):
        """
        Salva il buffer su disco in un thread separato (asincrono).
        
        Perché asincrono? Il buffer può essere gigabyte di dati.
        Salvarlo nel thread principale bloccherebbe il gioco per secondi.
        Con un daemon thread, il salvataggio avviene in background.
        'daemon=True' significa che il thread viene ucciso automaticamente
        quando il programma principale termina.
        """
        arrays = self._snapshot_to_numpy()  # serializza PRIMA di passare al thread
        if arrays is None:
            print("[PER] Buffer vuoto — skip salvataggio.")
            return
        n = len(arrays['grids'])

        def _write():
            try:
                os.makedirs(buffer_dir, exist_ok=True)  # crea la cartella se non esiste
                for key, arr in arrays.items():
                    # Salva ogni array come file .npy separato
                    np.save(os.path.join(buffer_dir, f"{key}.npy"), arr)
                # Calcola la dimensione totale in MB per il log
                total_mb = sum(
                    os.path.getsize(os.path.join(buffer_dir, f"{k}.npy"))
                    for k in arrays
                ) / 1e6
                print(f"[PER] Buffer salvato: {n} transizioni → {buffer_dir}/ ({total_mb:.0f} MB)")
            except Exception as e:
                print(f"[PER] Errore salvataggio buffer: {e}")

        threading.Thread(target=_write, daemon=True).start()

    def load(self, buffer_dir, max_load=None):
        """
        Carica un buffer precedentemente salvato da disco.
        
        Supporta due formati:
          - Nuovo (cartella di .npy): più veloce, supporta memory-mapping
          - Legacy (file .npz singolo): compatibilità con versioni vecchie

        Memory mapping (mmap_mode='r'): i dati non vengono caricati tutti in RAM
        subito, ma letti dal disco man mano che servono. Utile per buffer enormi.

        Args:
            buffer_dir : percorso alla cartella (nuovo) o al file .npz (legacy)
            max_load   : se specificato, carica solo le ultime max_load transizioni
        """
        is_new_format = os.path.isdir(buffer_dir)
        is_old_format = (not is_new_format) and os.path.exists(buffer_dir)
        if not is_new_format and not is_old_format:
            print(f"[PER] Nessun buffer in '{buffer_dir}' — parto vuoto.")
            return
        try:
            if is_new_format:
                print(f"[PER] Caricamento buffer (npy+mmap) da '{buffer_dir}/'...")
                arrays = {}
                for key in _BUFFER_KEYS:
                    fpath = os.path.join(buffer_dir, f"{key}.npy")
                    if not os.path.exists(fpath):
                        raise FileNotFoundError(f"File mancante: {fpath}")
                    # mmap_mode='r': solo lettura, non carica tutto in RAM subito
                    # meta non usa mmap perché è piccolo e serve subito
                    arrays[key] = np.load(fpath,
                                          mmap_mode=(None if key == 'meta' else 'r'),
                                          allow_pickle=False)
            else:
                print(f"[PER] Caricamento buffer (npz LEGACY) da '{buffer_dir}'...")
                arrays = np.load(buffer_dir, allow_pickle=False)

            n = len(arrays['grids'])
            if max_load is not None and n > max_load:
                start     = n - max_load  # prendi le ultime max_load transizioni
                n_to_load = max_load
                print(f"[PER] Buffer ha {n} transizioni, carico le ultime {max_load}.")
            else:
                start     = 0
                n_to_load = n

            print(f"[PER] Conversione di {n_to_load} transizioni in tensori PyTorch...")
            for i in range(start, start + n_to_load):
                # Log di avanzamento ogni 10.000 transizioni
                if (i - start) > 0 and (i - start) % 10_000 == 0:
                    pct = (i - start) / n_to_load * 100
                    print(f"[PER]   {i - start:>6}/{n_to_load}  ({pct:.0f}%)")
                # Ricostruisce la Transition: .copy() è necessario con mmap per
                # ottenere un array scrivibile (mmap è read-only)
                t = Transition(
                    torch.from_numpy(arrays['grids'][i].copy()).float().to(device),
                    torch.from_numpy(arrays['vars'][i].copy()).float().to(device),
                    torch.from_numpy(arrays['actions'][i].copy()).long().to(device),
                    torch.from_numpy(arrays['next_grids'][i].copy()).float().to(device),
                    torch.from_numpy(arrays['next_vars'][i].copy()).float().to(device),
                    torch.from_numpy(arrays['rewards'][i].copy()).float().to(device),
                    torch.from_numpy(arrays['dones'][i].copy()).to(device),
                )
                self.tree.add(float(arrays['priorities'][i]), t)

            # Ripristina i metadati per continuare il training esattamente da dove era
            self.frame  = int(arrays['meta'][2])    # numero di frame già visti
            self._max_p = float(arrays['meta'][3])  # max priorità storica
            fmt = "npy+mmap" if is_new_format else "npz-legacy"
            print(f"[PER] ✓ {n_to_load} transizioni caricate [{fmt}] | "
                  f"frame={self.frame} | max_p={self._max_p:.4f} | beta={self._beta():.4f}")
        except Exception as e:
            print(f"[PER] Errore caricamento: {e} — parto vuoto.")


# Alias: ReplayMemory è usato nel codice principale, ma punta a PrioritizedReplayMemory
ReplayMemory = PrioritizedReplayMemory


# =============================================================================
# SEZIONE 5 — N-STEP BUFFER
# =============================================================================

class NStepBuffer:
    """
    Accumula N transizioni consecutive per calcolare il "N-step return".

    Nel DQN standard: Q(s,a) ← r + γ * Q(s')
    Con N-step return: Q(s,a) ← r₀ + γ*r₁ + γ²*r₂ + ... + γᴺ⁻¹*rₙ₋₁ + γᴺ * Q(sₙ)

    Perché? Propaga le reward più lontano nel tempo: se tocchi una pillola
    di ossigeno al passo 3, il N-step return lo "sente" già al passo 0.
    Riduce il bias (sopravvalutazione) e accelera la propagazione del valore.

    Svantaggio: introduce varianza (l'ambiente potrebbe essere stocastico).
    Il compromesso n=5 è empiricamente buono per giochi 2D.
    """

    def __init__(self, n_step=5, gamma=0.99):
        """
        Args:
            n_step : numero di passi da accumulare (default 5)
            gamma  : fattore di sconto (default 0.99 — molto a lungo termine)
        """
        self.n_step = n_step
        self.gamma  = gamma
        self.buf    = deque(maxlen=n_step)  # coda circolare: scarta automaticamente le vecchie

    def push(self, sg, sv, a, sg_next, sv_next, r, done):
        """Aggiunge una transizione alla coda."""
        self.buf.append((sg, sv, a, sg_next, sv_next, r, done))

    def is_ready(self):
        """True solo quando il buffer ha esattamente n_step transizioni."""
        return len(self.buf) == self.n_step

    def _calc_return(self, sub):
        """
        Calcola il return cumulativo scontato per una sottosequenza di transizioni.
        Si ferma al primo step terminale (done=True).

        Returns:
            R           : return accumulato
            terminal_idx: indice dell'ultimo step considerato (dove l'episodio finisce)
        """
        R            = 0.0
        terminal_idx = len(sub) - 1  # default: ultimo step della sequenza

        for k, t in enumerate(sub):
            # Estrae la reward (gestisce sia tensor che float puri)
            r    = t[5].item() if isinstance(t[5], torch.Tensor) else float(t[5])
            # Estrae il flag done
            done = bool(t[6].item() if isinstance(t[6], torch.Tensor) else t[6])
            R   += self.gamma ** k * r  # aggiunge la reward scontata al return
            if done:
                terminal_idx = k  # episodio terminato qui: non guardare oltre
                break

        return R, terminal_idx

    def get(self):
        """
        Estrae la prossima transizione N-step dal buffer (se pronto).
        La transizione risultante ha:
          - stato iniziale: buf[0]
          - azione: buf[0]
          - reward: somma scontata degli N passi (o fino a done)
          - stato successivo: buf[terminal_idx] (ultimo stato prima di done)

        Returns:
            tupla (s_grid, s_vars, action, ns_grid, ns_vars, reward_nstep, done)
            oppure None se il buffer non è ancora pieno.
        """
        if not self.is_ready():
            return None
        buf_list = list(self.buf)
        R, ti    = self._calc_return(buf_list)
        result = (
            buf_list[0][0],  # stato iniziale: griglia
            buf_list[0][1],  # stato iniziale: vars
            buf_list[0][2],  # azione al primo step
            buf_list[ti][3], # stato successivo: griglia (dopo ti step)
            buf_list[ti][4], # stato successivo: vars
            torch.tensor([R], device=device),  # return N-step calcolato
            buf_list[ti][6]  # done flag al passo terminale
        )
        self.buf.popleft()  # rimuove il primo elemento per fare spazio al prossimo push
        return result

    def drain(self):
        """
        Svuota il buffer calcolando le transizioni N-step per tutti i passi rimanenti.
        Chiamato alla FINE di un episodio per non perdere le ultime transizioni
        (che potrebbero essere meno di n_step).

        Per ogni possibile "inizio" residuo, calcola il return massimo possibile
        con i passi che restano (es. 4-step, 3-step, 2-step, 1-step return).
        """
        if not self.buf:
            return []   # [FIX-DRAIN-GUARD] guard esplicito su buf vuoto
        buf_list = list(self.buf)
        results  = []
        for start in range(len(buf_list)):
            sub      = buf_list[start:]  # sottolista da questo punto in poi
            R, ti    = self._calc_return(sub)
            results.append((
                sub[0][0], sub[0][1], sub[0][2],
                sub[ti][3], sub[ti][4],
                torch.tensor([R], device=device),
                sub[ti][6]
            ))
        self.buf.clear()  # svuota il buffer dopo aver estratto tutto
        return results

    def flush(self):
        """Svuota il buffer senza estrarre nulla (es. reset di emergenza)."""
        self.buf.clear()


# =============================================================================
# SEZIONE 6 — REWARD SHAPER
# =============================================================================

class RewardShaper:
    """
    Potential-Based Reward Shaping: aggiunge un bonus di "progresso"
    senza alterare la policy ottimale appresa.

    Teoria (Ng et al. 1999): se aggiungi F(s,s') = γ*Φ(s') - Φ(s)
    come bonus reward, la policy ottimale rimane invariata.
    Qui Φ(s) = (profondità / righe_totali) * 2: una funzione potenziale
    che cresce mano a mano che il player scende (obiettivo = scendere in fondo).

    Il bonus sarà positivo quando il player scende (γ*Φ_nuovo > Φ_vecchio)
    e negativo quando risale, senza introdurre bias permanente.
    """

    def __init__(self):
        self.prev_phi = 0.0  # valore del potenziale allo step precedente

    def _phi(self, pY, oxy, rows):
        """
        Funzione potenziale: misura quanto è "buona" la posizione corrente.
        Cresce linearmente con la profondità. L'ossigeno non è usato
        (era nella versione precedente, ma è stato rimosso per semplicità).
        """
        return (pY / max(rows, 1)) * 2.0

    def reset(self, pY, oxy, rows):
        """
        Inizializza il potenziale a inizio episodio.
        Chiamare questo ogni volta che inizia un nuovo livello/vita.
        """
        self.prev_phi = self._phi(pY, oxy, rows)

    def step(self, pY, oxy, rows, gamma=0.99):
        """
        Calcola il bonus di shaping per questo step.

        Returns:
            b : float, il bonus di shaping (può essere negativo se risale)
        """
        phi         = self._phi(pY, oxy, rows)
        b           = gamma * phi - self.prev_phi  # formula potential-based
        self.prev_phi = phi  # salva per il prossimo step
        return b


# =============================================================================
# SEZIONE 6b — DRILL TRACKER
# =============================================================================

class DrillTracker:
    """
    Tiene traccia del comportamento del player per penalizzare loop e stasi.

    Problemi che risolve:
      1) Oscillazione: player che va avanti-indietro senza progredire
         → penalità oscillation_penalty()
      2) Column lock: player bloccato nella stessa colonna senza scendere
         → penalità column_lock_penalty()
      3) Idle streak: player che non fa niente per troppi step
         → tracciato da idle_streak (usato in calculate_reward)
    """

    def __init__(self, osc_window=8, col_window=12):
        """
        Args:
            osc_window : ultime N posizioni per rilevare oscillazioni (default 8)
            col_window : ultimi M step per rilevare column lock (default 12)
        """
        self.recent_pos     = deque(maxlen=osc_window)  # posizioni recenti (per oscillazione)
        self.recent_history = deque(maxlen=col_window)  # storia più lunga (per column lock)
        self.consecutive_fails = 0  # quante azioni consecutive sono fallite (non si è mosso)
        self.idle_streak       = 0  # quanti step consecutivi l'agente ha scelto "idle" (azione 0)

    def reset(self):
        """Resetta tutto a inizio di un nuovo episodio o dopo una morte."""
        self.recent_pos.clear()
        self.recent_history.clear()
        self.consecutive_fails = 0
        self.idle_streak       = 0

    def record_action_result(self, moved: bool, action_idx: int) -> None:
        """
        Aggiorna i contatori in base all'esito dell'ultima azione.

        Args:
            moved      : True se il player si è effettivamente spostato
            action_idx : indice dell'azione appena eseguita
        """
        if action_idx == 0:
            # Azione "idle" (stare fermi): aumenta streak idle, azzera fail
            self.idle_streak       += 1
            self.consecutive_fails  = 0
        elif moved:
            # Azione riuscita (ci si è mossi): azzera entrambi i contatori
            self.consecutive_fails = 0
            self.idle_streak       = 0
        else:
            # Azione fallita (si è tentato qualcosa ma non ci si è mossi): aumenta fail
            self.consecutive_fails += 1
            self.idle_streak       = 0

    def update(self, x, y):
        """
        Registra la nuova posizione del player nelle code di storico.
        DEVE essere chiamata DOPO oscillation_penalty() (vedi commento sotto).
        """
        self.recent_pos.append((x, y))
        self.recent_history.append((x, y))

    def oscillation_penalty(self, x, y):
        """
        Calcola la penalità di oscillazione: quanto spesso il player
        è già stato in questa posizione recentemente.

        IMPORTANTE [FIX-OSC-ORDER]: deve essere chiamata PRIMA di update(x,y).
        Se chiamata dopo, (x,y) sarebbe già in recent_pos → hits >= 1 sempre
        → penalità garantita ad ogni step (bug nella v6.8).

        Eccezioni (nessuna penalità):
          - Progresso verticale ≥ 2 righe nell'ultima finestra (si sta scendendo)
          - Navigazione laterale legittima (lateral_range ≥ 1, cioè ≥ 2 colonne distinte)

        Returns:
            penalità tra 0.0 e 6.0
        """
        # Conta quante volte (x,y) appare nelle posizioni recenti
        hits = sum(1 for p in self.recent_pos if p == (x, y))
        if hits == 0:
            return 0.0  # posizione nuova: nessuna penalità

        if len(self.recent_history) > 1:
            # net_progress: differenza di profondità (Y) dall'inizio della finestra ad ora
            net_progress  = y - self.recent_history[0][1]
            # lateral_range: quante colonne diverse ha toccato nella finestra
            xs            = [p[0] for p in self.recent_history]
            lateral_range = max(xs) - min(xs)

            if net_progress >= 2:
                return 0.0  # sta scendendo abbastanza: non è un loop

            # [FIX-LATERAL] >= 1: se ha usato almeno 2 colonne distinte
            # (range=1 significa toccare es. colonne 3 e 4), considera navigazione legittima
            if net_progress >= 0 and lateral_range >= 1:
                return 0.0

        # [FIX-OSC-CAP] Cap a 6.0: senza cap con 8 posizioni max,
        # hits potrebbe arrivare a 7 → penalità = 7*1.5 = 10.5 (troppo)
        return min(hits * 1.5, 6.0)

    def column_lock_penalty(self):
        """
        Penalizza il player se è rimasto bloccato nella stessa colonna senza scendere.

        Rileva due pattern:
          1) TUTTA la finestra: stessa colonna E profondità non aumentata → penalità 2.0
          2) ULTIMI 6 step: stessa colonna E profondità non aumentata → penalità 0.5

        Se la profondità aumenta anche stando in colonna (corridoio verticale),
        non penalizza: stare in colonna è OK se si sta scendendo.

        Returns:
            0.0, 0.5 o 2.0
        """
        if len(self.recent_history) < 12:
            return 0.0  # non abbastanza storia per valutare

        history = list(self.recent_history)
        cols    = [p[0] for p in history]   # lista delle X (colonne)
        depths  = [p[1] for p in history]   # lista delle Y (profondità)

        if len(set(cols)) == 1:
            # Tutti i 12 step nella stessa colonna
            return 2.0 if depths[-1] <= depths[0] else 0.0  # penalizza solo se non scende

        # Controlla gli ultimi 6 step
        rc = cols[-6:]
        rd = depths[-6:]
        if len(set(rc)) == 1 and rd[-1] <= rd[0]:
            return 0.5  # bloccato negli ultimi 6 step senza scendere

        return 0.0


# =============================================================================
# SEZIONE 7 — SELECT ACTION
# =============================================================================

def select_action(policy_net, state_grid, state_vars, epsilon, n_actions):
    """
    Sceglie l'azione da eseguire usando la policy ε-greedy.

    ε-greedy: con probabilità ε sceglie un'azione casuale (esplorazione),
    altrimenti sceglie l'azione con Q-value massimo (sfruttamento).

    In Rainbow DQN, l'esplorazione è gestita principalmente dai NoisyLayers,
    ma ε-greedy viene ancora usato come backup di sicurezza (soprattutto
    nelle prime fasi di training quando ε è alto).

    Args:
        policy_net  : la rete neurale corrente
        state_grid  : tensore [6, 11, 9] dello stato visuale corrente
        state_vars  : tensore [32] dello stato scalare corrente
        epsilon     : probabilità di azione casuale (0.0 = solo greedy)
        n_actions   : numero totale di azioni

    Returns:
        (action, q_values): tensore [[azione_scelta]], array numpy dei Q-values (o None)
    """
    policy_net.eval()  # disattiva dropout e BatchNorm in modalità training

    if random.random() < epsilon:
        # Esplorazione: sceglie un'azione completamente a caso
        return (torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), None)

    with torch.no_grad():  # no_grad: non calcola i gradienti (risparmia memoria e tempo)
        # unsqueeze(0): aggiunge la dimensione batch → [1, 6, 11, 9]
        q = policy_net(state_grid.unsqueeze(0), state_vars.unsqueeze(0))

    # Controllo di sicurezza: NaN/Inf nei Q-values indicano training instabile
    if torch.isnan(q).any() or torch.isinf(q).any():
        print("[WARNING] Q-values NaN/Inf in select_action — fallback random")
        return (torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), None)

    # q.max(1): trova il massimo lungo la dimensione azioni
    # [1]: prende l'indice (argmax), non il valore
    # .view(1,1): reshape a [[n]] per compatibilità con il resto del codice
    return q.max(1)[1].view(1, 1), q.squeeze().cpu().numpy()


# =============================================================================
# SEZIONE 8 — REWARD FUNCTION
# =============================================================================

def calculate_reward(prev_y, prev_x, new_y, new_x,
                     post_action_y, post_action_x,
                     prev_oxy, new_oxy,
                     prev_score, new_score, prev_lives, new_lives,
                     action_idx, is_hard_block, is_delayed_block,
                     drill_tracker, total_rows, shaping_bonus=0.0,
                     is_level_complete=False, blocks_around=None,
                     is_game_over=False):
    """
    Calcola la reward per uno step di gioco.
    Versione v6.9.

    La reward è la somma/differenza di molti componenti:
      + Vittoria livello / - Morte
      + Profondità guadagnata
      + Ossigeno raccolto (continuo, non più a tier)
      + Schivata blocco cadente
      - Blocco duro tentato (indistruttibile)
      - Azione fallita (non ci si è mossi)
      - Oscillazione / column lock
      - Deficit ossigeno (penalità passiva costante)
      ± Shaping bonus (da RewardShaper)

    La reward finale viene clampata in [-REWARD_CLIP, +REWARD_CLIP] = [-12, +12].

    Args:
        prev_y/x, new_y/x  : posizione prima e dopo lo step
        post_action_y/x    : posizione IMMEDIATAMENTE dopo l'azione (prima della gravità)
        prev/new_oxy       : ossigeno prima e dopo
        prev/new_score     : punteggio prima e dopo
        prev/new_lives     : vite prima e dopo
        action_idx         : azione scelta (0=idle, 1=sx, 2=dx, 3=drill-sx, 4=drill-dx, 5=giù)
        is_hard_block      : True se ha tentato di rompere un muro indistruttibile
        is_delayed_block   : True se ha tentato un blocco ritardato
        drill_tracker      : istanza DrillTracker per penalità anti-loop
        total_rows         : altezza totale del livello (per normalizzazioni)
        shaping_bonus      : bonus da RewardShaper (potential-based)
        is_level_complete  : True se ha completato il livello
        blocks_around      : dict con info sui blocchi sopra il player
        is_game_over       : True se ha perso tutte le vite
    """

    # ── CASO TERMINALE: Vittoria ──────────────────────────────────────────────
    if is_level_complete:
        # Bonus ossigeno: arrivare con più aria = reward più alta (max +20)
        oxy_bonus = (new_oxy / 100.0) * 20.0
        # Base +80, +15 per ogni vita rimasta, +oxy_bonus
        return torch.tensor([80.0 + new_lives * 15.0 + oxy_bonus], device=device)

    # ── CASO TERMINALE: Perdita di vita ──────────────────────────────────────
    if new_lives < prev_lives:
        drill_tracker.reset()  # resetta i tracker: nuovo episodio
        # Più è in profondità quando muore, più grave è la perdita (stava per finire)
        rows_remaining = max(total_rows - 1 - prev_y, 0)
        # [VAR-DEATH] Base -15, fino a -40 per morte in fondo, -50 se game over
        penalty = -15.0 - (rows_remaining / max(total_rows - 1, 1)) * 25.0
        if is_game_over:
            penalty -= 10.0
        return torch.tensor([penalty], device=device)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP NON-TERMINALE: calcola la reward componente per componente
    # ─────────────────────────────────────────────────────────────────────────

    reward  = 0.0               # accumulatore reward
    d_score = new_score - prev_score  # differenza punteggio
    dy      = new_y - prev_y         # differenza profondità (>0 = sceso)
    d_oxy   = new_oxy - prev_oxy     # differenza ossigeno (>0 = raccolto)
    earned_dodge = False              # flag: ha già guadagnato una dodge reward?

    # [FIX1] Post-action snapshot: capisce se si è mosso escludendo la gravità.
    # Se usassimo new_y/new_x, la caduta fisica conterebbe come "mi sono mosso"
    # anche se l'azione era idle. Con post_action_y/x confrontiamo SOLO l'effetto
    # dell'azione, non la fisica successiva.
    actually_moved = (post_action_y != prev_y) or (post_action_x != prev_x)

    # [FIX2] Salva consecutive_fails PRIMA di aggiornarlo: la penalità di azione
    # fallita deve usare il valore DI PRIMA di questo step, non quello aggiornato.
    prev_consecutive_fails = drill_tracker.consecutive_fails
    drill_tracker.record_action_result(actually_moved, action_idx)

    # ── Scansione minacce sopra il player ────────────────────────────────────
    # blocks_around è un dict con chiavi "up", "above_2", "above_3" (1,2,3 celle sopra)
    # ogni valore è un dict con "falling": bool, "shaking": bool
    above_falling_now   = False   # c'è un blocco IN CADUTA LIBERA sopra di me?
    column_threat_above = False   # c'è una minaccia (anche solo vibrazione) sopra?
    if blocks_around:
        for key in ("up", "above_2", "above_3"):
            bk = blocks_around.get(key)
            if bk:
                if bk.get("falling", False):
                    above_falling_now   = True
                    column_threat_above = True
                    break  # basta trovarne uno in caduta: è il caso peggiore
                if bk.get("shaking", False):
                    column_threat_above = True  # minaccia meno immediata, continua a cercare

    # ── Penalità IDLE in situazioni pericolose ────────────────────────────────
    # Se c'è pericolo sopra, stare fermi (idle) o fare drill laterale è sbagliato.
    _idle_col_penalized = False  # flag per mutua esclusività con PHYS-REWARD sotto

    if above_falling_now:
        # Un blocco sta CADENDO ADESSO sulla testa del player.
        if action_idx == 0:
            reward -= 2.0    # idle: non stai scappando! Penalità moderata.
        elif action_idx in [3, 4]:
            reward -= 3.0    # drill laterale: stai sprecando tempo su muri mentre cadi
                             # È PEGGIO dell'idle: almeno idle non fa danni
        elif action_idx in [1, 2]:
            pass             # movimento laterale: CORRETTO, nessuna penalità
    elif column_threat_above:
        # Minaccia sopra ma non in caduta immediata (vibrazione)
        if action_idx == 0:
            reward -= 1.0    # idle: potresti prepararti
        elif action_idx in [3, 4]:
            reward -= 1.5    # drill inutile quando c'è pericolo sopra

    # ── Penalità azione fallita ───────────────────────────────────────────────
    # Se ha tentato un'azione non-idle ma non si è mosso, piccola penalità.
    # Scala decrescente: più fallimenti consecutivi → penalità ridotta (per evitare
    # che l'agente venga punito all'infinito per muri indistruttibili).
    if action_idx != 0 and not actually_moved:
        if not is_hard_block and not is_delayed_block:
            base_pen   = _FAIL_PENALTY_BASE.get(action_idx, 0.3)
            # fail_scale: diminuisce con i fallimenti consecutivi → penalità ridotta
            # fail_scale: 1/(1 + n*0.4) → 1 fail=0.71, 2=0.55, 3=0.45
            fail_scale = 1.0 / (1.0 + prev_consecutive_fails * 0.4)
            reward -= base_pen * fail_scale

    # ── Penalità blocchi speciali ─────────────────────────────────────────────
    if is_hard_block:
        reward -= 8.0    # muro indistruttibile: forte penalità per disincentivare questi tentativi
    elif is_delayed_block:
        # [FIX-DELAYED-OK] In v6.8 penalizzava SEMPRE il delayed, anche se lo stava
        # rompendo correttamente (richiedono più colpi). Ora penalizza solo i tentativi FALLITI.
        if not actually_moved:
            reward -= 3.0

    # ── Reward schivata blocco cadente ────────────────────────────────────────
    # Se si muove mentre c'è un blocco in caduta sopra di lui → ha schivato!
    # Viene dato un bonus significativo per rinforzare il comportamento di fuga.
    if blocks_around and actually_moved and action_idx in [1, 2, 3, 4]:
        for key in ("up", "above_2", "above_3"):
            bk = blocks_around.get(key)
            if bk and bk.get("falling", False):
                reward += 8.0            # bonus schivata: grande perché è la risposta corretta
                earned_dodge = True      # segna che la dodge è già stata premiata
                break                    # conta solo una volta anche se ci sono più blocchi

    # ── PHYS-REWARD: pressione per minaccia in colonna ────────────────────────
    # [FIX-IDLE-DOUBLE] Se abbiamo già penalizzato per idle sopra, non penalizziamo
    # di nuovo qui. La penalità deve essere applicata UNA SOLA VOLTA.
    if column_threat_above and not earned_dodge and not above_falling_now \
            and not _idle_col_penalized:
        if action_idx == 5:    # drill verso il basso mentre c'è minaccia in colonna
            reward -= 0.5      # leggera penalità: potrebbe funzionare ma è rischioso

    # ── Urgenza ossigeno bassa ────────────────────────────────────────────────
    # Se ha poco ossigeno E fa qualcosa (non idle), piccolo bonus per incoraggiare azione
    if prev_oxy < 20 and action_idx != 0:
        reward += 0.5

    # ── Reward profondità ─────────────────────────────────────────────────────
    if dy > 0:
        # È sceso: reward proporzionale alla distanza scesa
        # depth_progress: bonus aggiuntivo per chi è già sceso in profondità
        depth_progress = min(new_y / max(total_rows, 1), 1.0)
        reward += dy * 2.5 + depth_progress * 0.5  # +2.5 per riga + bonus profondità
        if action_idx == 5:
            reward += 0.6  # bonus extra per drill diretto verso il basso (incoraggiato)
    elif dy < 0:
        # È risalito: penalità proporzionale (ma meno grave della reward di scesa)
        reward += dy * 1.0  # dy è negativo → reward negativa

    # ── Reward ossigeno raccolta ──────────────────────────────────────────────
    # [VAR-OXY] Formula continua: più sei a corto d'ossigeno, più vale raccoglierlo.
    # Evita "cliff edges" (salti brutti nel valore) della versione precedente a tier.
    if d_oxy > 0:
        oxy_factor = max(0.0, 1.0 - prev_oxy / 100.0)  # 0.0 se ossigeno pieno, 1.0 se vuoto
        # Es: prev_oxy=0  → oxy_factor=1.0 → reward = 3 + 17 = +20
        # Es: prev_oxy=50 → oxy_factor=0.5 → reward = 3 + 8.5 = +11.5
        # Es: prev_oxy=100→ oxy_factor=0.0 → reward = 3 + 0   = +3
        reward += 3.0 + oxy_factor * 17.0

    # ── Penalità movimento laterale inefficace ────────────────────────────────
    # Se va a sinistra/destra senza raccogliere ossigeno né scendere, piccola penalità.
    # Non penalizza se ha guadagnato una dodge (quello vale di più).
    if action_idx in [1, 2] and not earned_dodge:
        if d_oxy <= 0 and dy == 0:
            reward -= 0.5  # movimento laterale che non porta a niente

    # ── Reward/penalità drill laterale ───────────────────────────────────────
    if action_idx in [3, 4]:
        # Il drill laterale è rischioso: se non porta punteggio né blocchi speciali, penalizza
        if not d_score > 0 and not is_hard_block and not is_delayed_block:
            reward -= 0.5

    # ── Reward punteggio generico ─────────────────────────────────────────────
    if d_score > 0:
        # Normalizza il punteggio: 100 punti = +1.0, capped a +2.0
        base_score_reward = min(d_score / 100.0, 2.0)

        # ANTI-CHAIN-ABUSE: se ha distrutto molti blocchi con drill laterale (chain > 3),
        # riduce la reward perché grandi chain creano instabilità fisica pericolosa.
        # danger_factor cresce linearmente oltre i 30 punti (≈3 blocchi).
        if d_score > 30 and action_idx in [3, 4]:
            danger_factor = min((d_score - 30) / 100.0, 1.0)  # 0→1 tra 30 e 130 punti
            base_score_reward -= danger_factor * 1.5           # max -1.5 per chain enorme

        reward += base_score_reward

    # ── Penalità anti-loop ────────────────────────────────────────────────────
    # [FIX-OSC-ORDER] PRIMA si calcola la penalità, POI si aggiorna il tracker.
    # Vedi commento dettagliato in DrillTracker.oscillation_penalty().
    reward -= drill_tracker.oscillation_penalty(new_x, new_y)  # penalità oscillazione
    drill_tracker.update(new_x, new_y)                          # aggiorna DOPO
    reward -= drill_tracker.column_lock_penalty() * 1.0         # penalità column lock

    # ── Penalità ossigeno mancante (costo continuo) ───────────────────────────
    # Piccola penalità passiva proporzionale al deficit di ossigeno.
    # Se ossigeno = 100 → -0.0; se ossigeno = 0 → -0.15 per step.
    # Crea una leggera pressione costante a cercare ossigeno.
    oxy_deficit = max(0.0, 100.0 - new_oxy) / 100.0  # 0.0 = pieno, 1.0 = vuoto
    reward -= oxy_deficit * 0.15

    # ── Shaping bonus ─────────────────────────────────────────────────────────
    reward += shaping_bonus  # aggiunge il bonus potential-based da RewardShaper

    # ── Clipping finale ───────────────────────────────────────────────────────
    # Garantisce che la reward sia in [-12, +12] qualunque cosa sia successa.
    # Importante per la stabilità del training (evita gradienti enormi).
    return torch.tensor([max(-REWARD_CLIP, min(REWARD_CLIP, reward))], device=device)


# =============================================================================
# SEZIONE 9 — OPTIMIZE MODEL
# =============================================================================

def optimize_model(policy_net, target_net, memory, optimizer, batch_size=64, gamma=0.99):
    """
    Esegue UN passo di ottimizzazione della rete neurale.
    
    Algoritmo: Double DQN con Prioritized Replay e Importance Sampling.

    Double DQN (van Hasselt et al.):
      - policy_net sceglie l'azione migliore per lo stato successivo
      - target_net valuta quella azione
      Separa selezione e valutazione per ridurre la sopravvalutazione dei Q-values.

    Passi:
      1. Campiona un batch dal buffer prioritizzato
      2. Calcola Q(s,a) con policy_net (il valore stimato attualmente)
      3. Calcola il target: r + γ * Q_target(s', argmax_policy(s'))
      4. Calcola il TD-error e aggiorna le priorità nel buffer
      5. Calcola la loss pesata (IS weights) e backpropagation
      6. Controlla NaN/Inf e ripristina i pesi se necessario

    Args:
        policy_net  : la rete che si aggiorna ad ogni step
        target_net  : la rete stabile (aggiornata raramente per stabilità)
        memory      : il buffer di replay prioritizzato
        optimizer   : ottimizzatore (di solito Adam o RMSprop)
        batch_size  : numero di esperienze per batch (default 64)
        gamma       : fattore di sconto (default 0.99)

    Returns:
        (loss, td_errors) se il training ha successo, None altrimenti
    """
    if len(memory) < batch_size:
        return None  # non abbastanza esperienze: aspetta

    # 1. Campionamento dal buffer prioritizzato
    # trans = lista di Transition
    # tree_idx = indici SumTree (per aggiornare le priorità dopo)
    # is_w = pesi importance sampling per correggere il bias del PER
    trans, tree_idx, is_w = memory.sample(batch_size)

    # Unzip: da lista di Transition a Transition di liste
    batch = Transition(*zip(*trans))

    # Stack: da lista di tensori a tensore [batch_size, ...]
    sg = torch.stack(batch.state_grid)   # [bs, 6, 11, 9]
    sv = torch.stack(batch.state_vars)   # [bs, 32]
    ac = torch.cat(batch.action)         # [bs, 1] — azioni scelte
    rw = torch.cat(batch.reward)         # [bs] — rewards ricevute

    # done_batch: True per le transizioni terminali (episodio finito)
    done_batch        = torch.cat(batch.done).bool()
    # non_terminal_mask: True per le transizioni NON terminali (si può fare bootstrapping)
    non_terminal_mask = ~done_batch

    # Ricampiona il rumore PRIMA del forward pass (NoisyNets)
    policy_net.sample_noise()
    policy_net.train()

    # 2. Q(s,a) corrente: la stima attuale della rete per l'azione presa
    # .gather(1, ac): per ogni esperienza, prende solo il Q-value dell'azione effettuata
    sav = policy_net(sg, sv).gather(1, ac)  # shape [bs, 1]

    # 3. Calcolo del target: inizializza a 0 (per gli stati terminali, V(s')=0)
    nsv = torch.zeros(batch_size, device=device)  # [bs]

    if non_terminal_mask.any():
        # Considera solo le transizioni non terminali per il bootstrapping
        indices = non_terminal_mask.nonzero(as_tuple=True)[0]  # indici delle transizioni non-terminali
        nfg_t   = torch.stack([batch.next_state_grid[i] for i in indices.tolist()])  # griglie s'
        nfv_t   = torch.stack([batch.next_state_vars[i] for i in indices.tolist()])  # vars s'

        with torch.no_grad():  # nessun gradiente per il calcolo del target
            # DOUBLE DQN: policy_net sceglie l'azione migliore per s'
            policy_net.eval()
            best_a = policy_net(nfg_t, nfv_t).argmax(1, keepdim=True)  # argmax secondo policy
            # target_net valuta quella azione: più stabile, riduce overestimation
            nsv[non_terminal_mask] = target_net(nfg_t, nfv_t).gather(1, best_a).squeeze(1)

    policy_net.train()  # torna in modalità training

    # Target finale: r + γ * V(s') secondo Double DQN
    # .detach(): il target non deve avere gradienti (è un "obiettivo fisso" per questo passo)
    exp_q = (rw + gamma * nsv).unsqueeze(1).detach()  # [bs, 1]

    # 4. Calcola TD-errors per aggiornare le priorità nel buffer
    with torch.no_grad():
        td_err = (exp_q - sav).abs().squeeze(1).cpu().numpy()  # |target - pred|
    memory.update_priorities(tree_idx, td_err)  # aggiorna il SumTree

    # 5. Calcola la loss pesata (Huber loss per robustezza agli outlier)
    # smooth_l1_loss = Huber loss: L2 per errori piccoli, L1 per errori grandi
    # * is_w: moltiplicazione elementwise per i pesi IS (corregge il bias PER)
    # .mean(): media sul batch
    loss = (F.smooth_l1_loss(sav, exp_q, reduction='none').squeeze(1) * is_w).mean()

    # Controllo di stabilità: se la loss è NaN/Inf, salta l'aggiornamento
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"[WARNING] Loss NaN/Inf ({loss.item():.4f}) — skip update")
        memory.update_priorities(tree_idx, [1e-5] * len(tree_idx))  # resetta le priorità
        return None

    # 6. Salva snapshot dei pesi PRIMA del backprop (recovery da NaN)
    param_backup = {k: v.clone() for k, v in policy_net.state_dict().items()}

    optimizer.zero_grad()              # azzera i gradienti accumulati nel passo precedente
    loss.backward()                    # calcola i gradienti con backpropagation
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)  # gradient clipping: cap a 1.0
    optimizer.step()                   # aggiorna i pesi con i gradienti calcolati

    # Controllo post-aggiornamento: se i pesi sono diventati NaN, ripristina
    has_nan = any(torch.isnan(p).any() or torch.isinf(p).any()
                  for p in policy_net.parameters())
    if has_nan:
        print("[CRITICAL] NaN nei pesi — ripristino snapshot")
        policy_net.load_state_dict(param_backup)  # ripristina lo snapshot pre-update
        return None

    # Restituisce la loss e i TD-errors per il logging
    return float(loss.item()), td_err