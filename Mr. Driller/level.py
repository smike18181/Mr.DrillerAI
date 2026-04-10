from random import randint  
import block              
import numpy as np         
import torch               
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seleziona il dispositivo di calcolo per PyTorch:
# - "cuda" se è disponibile una GPU NVIDIA con CUDA → training molto più veloce
# - "cpu" altrimenti → training più lento ma sempre funzionante
# torch.cuda.is_available() restituisce True se la GPU è presente e configurata.


def generateLvl(colors, lines, width, background, pillP=5, PillPL=7, PillMLE=20, SoloP=10, UnbreakableP=5, DelayedP=10):
    # Genera e restituisce una nuova griglia 2D di blocchi (il livello di gioco).
    #
    # Parametri:
    #   colors       — numero di colori disponibili per i blocchi Classic (es. 4 → colori 1,2,3,4)
    #   lines        — numero di righe di gioco effettive del livello (escluse le righe vuote in cima)
    #   width        — numero di colonne della griglia (larghezza del livello)
    #   background   — indice del tema grafico di sfondo (es. "1", "2", "3")
    #   pillP        — probabilità% di generare una Pill in una cella (default 5%)
    #   PillPL       — numero massimo di Pill generabili nell'intero livello (default 7)
    #   PillMLE      — numero minimo di righe tra due Pill consecutive (default 20)
    #   SoloP        — probabilità% di generare un blocco Solo (default 10%)
    #   UnbreakableP — probabilità% di generare un blocco Unbreakable X (default 5%)
    #   DelayedP     — probabilità% di generare un blocco Delayed (default 10%)

    level = []
    # Lista 2D che rappresenta il livello. Struttura: level[riga][colonna] = oggetto Block.
    # Sarà popolata nell'iterazione seguente e restituita alla fine.

    LineRemainBeforePill = 0
    # Contatore di "righe da attendere" prima di poter generare un'altra Pill.
    
    for i in range(lines + 5):
        # Itera su ogni riga del livello.
        # Il livello ha 'lines' righe di gioco PIÙ 5 righe speciali iniziali PLUS le righe End finali.
        # Totale righe = lines + 5 (5 righe vuote in cima) + n righe End in fondo.

        line = []
        # Lista che raccoglierà i blocchi di questa riga (lunghezza = width)

        if i in range(5):
            # ── Righe 0-4: ZONA VUOTA IN CIMA ─────────────────────────
            # Le prime 5 righe sono sempre vuote (Classic con HP=0 = aria).
            # Servono come "buffer" sopra al punto di spawn del giocatore:
            # il giocatore appare in (x=3, y=4) e le righe 0-3 sopra di lui sono sempre libere.
            for j in range(width):
                # Itera su ogni colonna della riga
                newBlock = block.Classic(j, i, 1, 0)
                # Crea un blocco Classic con:
                #   posX=j, posY=i  → posizione nella griglia
                #   colors=1        → colore 1 (non importa, è un blocco vuoto)
                #   forceHP=0       → HP=0: è "aria", non è visibile né colpibile
                newBlock.changeBG(background)   # Imposta lo sfondo corretto per il tema del livello
                line.append(newBlock)           # Aggiunge il blocco alla riga corrente

        elif i in range(lines):
            # ── Righe 5 .. lines-1: ZONA DI GIOCO EFFETTIVA ───────────
            # Per ogni riga di gioco, genera ogni cella in modo casuale
            # scegliendo il tipo di blocco in base alle probabilità configurate.

            LineRemainBeforePill -= 1
            # Decrementa il contatore di distanza dalla Pill precedente.
            # Quando scende sotto 0, sarà possibile generare una nuova Pill.

            for j in range(width):
                # Itera su ogni colonna della riga corrente

                # Genera un numero casuale per ogni tipo di blocco speciale.
                # Ogni numero è tra 0 e 100 e viene confrontato con la soglia di probabilità.
                # Es. se PillRn=3 e pillP=5 → 3 < 5 → viene generata una Pill.
                PillRn        = randint(0, 100)   # Numero casuale per la probabilità Pill
                UnbreakableRn = randint(0, 100)   # Numero casuale per la probabilità Unbreakable
                DelayedRn     = randint(0, 100)   # Numero casuale per la probabilità Delayed
                SoloRn        = randint(0, 100)   # Numero casuale per la probabilità Solo

                if PillRn < pillP and PillPL != 0 and LineRemainBeforePill < 0:
                    # ── Genera una PILL ───────────────────────────────
                    # Tre condizioni devono essere tutte vere:
                    # 1. PillRn < pillP: il numero casuale è sotto la soglia di probabilità
                    # 2. PillPL != 0: il limite massimo di Pill del livello non è esaurito
                    # 3. LineRemainBeforePill < 0: sono passate abbastanza righe dalla Pill precedente

                    PillPL -= 1
                    # Consuma un "slot" Pill disponibile (scende verso 0)

                    newBlock = block.Pill(j, i)
                    newBlock.changeBG(background)
                    line.append(newBlock)

                    LineRemainBeforePill = PillMLE
                    # Reimposta il contatore di distanza minima tra Pill.
                    # Le prossime PillMLE righe non potranno contenere un'altra Pill.

                elif SoloRn < SoloP:
                    # ── Genera un blocco SOLO ─────────────────────────
                    # Blocco senza chain reaction, si distrugge con 1 colpo.
                    # Verificato come secondo (dopo Pill) per dare priorità alle Pill.
                    newBlock = block.Solo(j, i)
                    newBlock.changeBG(background)
                    line.append(newBlock)

                elif UnbreakableRn < UnbreakableP:
                    # ── Genera un blocco UNBREAKABLE (X) ─────────────
                    # Blocco quasi indistruttibile (5 HP). Raro (5% di default).
                    newBlock = block.Unbreakable(j, i)
                    newBlock.changeBG(background)
                    line.append(newBlock)

                elif DelayedRn < DelayedP:
                    # ── Genera un blocco DELAYED ──────────────────────
                    # Blocco cristallo che scompare dopo un countdown di 2 secondi.
                    newBlock = block.Delayed(j, i)
                    newBlock.changeBG(background)
                    line.append(newBlock)

                else:
                    # ── Genera un blocco CLASSIC (default) ───────────
                    # Se nessun tipo speciale è stato selezionato, genera un blocco
                    # colorato standard con chain reaction.
                    newBlock = block.Classic(j, i, randint(1, colors), 1)
                    # randint(1, colors) sceglie un colore casuale tra 1 e colors (inclusi).
                    # Es. con colors=4 → colore random tra 1, 2, 3, 4.
                    # Il secondo argomento '1' abilita la chain reaction.
                    newBlock.changeBG(background)
                    line.append(newBlock)

        else:
            # ── Righe lines e oltre: ZONA DI FINE LIVELLO ─────────────
            # Queste righe contengono i blocchi End (uscita del livello).
            # Quando il giocatore raggiunge questa zona, il livello è completato.
            for j in range(width):
                newBlock = block.End(j, i)
                newBlock.changeBG(background)
                line.append(newBlock)

        level.append(line)
        # Aggiunge la riga completata alla griglia del livello.
        # Dopo il loop, level è una lista di liste: level[i][j] = blocco in riga i, colonna j.

    return level    # Restituisce la griglia 2D completa del livello


def render(surface, level, currOffset):
    # Disegna sullo schermo le righe del livello attualmente visibili.
    # Non si ridisegna l'intero livello (può essere lungo centinaia di righe),
    # ma solo la "finestra" di 9 righe visibili nella telecamera corrente.
    #
    # Parametri:
    #   surface    — la superficie pygame su cui disegnare
    #   level      — la griglia 2D di blocchi
    #   currOffset — indice della riga in cima allo schermo (quante righe sono "scomparse" in alto)

    start = currOffset
    # La prima riga da disegnare è quella in cima alla finestra visibile.
    # Corrisponde a quante righe il giocatore ha già superato (sono uscite dallo schermo in alto).

    end = min(currOffset + 9, len(level))
    # L'ultima riga da disegnare: 9 righe dopo l'inizio (lo schermo mostra 9 righe).
    # min() evita di uscire dai limiti della griglia se il livello è più corto di 9 righe
    # dal currOffset in poi (protezione contro IndexError).

    for i in range(start, end):
        # Itera sulle righe visibili (da start a end escluso)
        for element in level[i]:
            # Itera su ogni blocco della riga corrente
            element.display(surface, currOffset)