"""
main.py — Mr. Driller AI Training Loop (Rainbow DQN) — v6.9

DESCRIZIONE GENERALE DEL FILE:
    Questo è il file principale (entry point) dell'intero progetto.
    Coordina e collega tra loro tutti i sistemi:
      - Il motore di gioco Pygame (grafica, input, fisica dei blocchi)
      - L'agente di intelligenza artificiale Rainbow DQN
      - Il loop di training (addestramento della rete neurale)
      - Il sistema di reward (punteggio pedagogico per l'AI)
      - Il curriculum learning (progressione graduale dei livelli)
    
    FLUSSO ESECUTIVO:
      1. Vengono definiti tutti i parametri globali (iperparametri, costanti)
      2. Vengono create le classi di supporto (CurriculumManager, SmartFrameStacker)
      3. Viene avviata la funzione game() che contiene il loop principale
      4. Il loop gira 60 volte al secondo, gestendo input, fisica, AI e rendering
"""

# =============================================================================
# IMPORT LIBRERIE STANDARD PYTHON
# =============================================================================

import os
# 'os' è il modulo della libreria standard Python per interagire col sistema operativo.
# Ci permette di: leggere/scrivere variabili d'ambiente (os.environ),
# verificare se un file esiste (os.path.exists), creare cartelle, ecc.

import random
# 'random' fornisce funzioni per generare numeri casuali.
# Nel contesto RL, è usato per la strategia epsilon-greedy:
# con probabilità epsilon, l'agente sceglie un'azione casuale invece di usare la rete.

import sys
# 'sys' fornisce accesso a parametri e funzioni del runtime Python.
# Lo usiamo per: sys.exit() (chiudere il programma), sys.setrecursionlimit() (limite ricorsione).

import threading
# 'threading' permette l'esecuzione di codice in parallelo su thread separati.
# Lo usiamo per salvare i pesi della rete su disco in background,
# così il salvataggio non interrompe o rallenta il gioco.

from os import path
# Importa direttamente il sottomodulo 'path' da 'os'.
# Ci permette di scrivere path.join("cartella", "file.txt") invece di os.path.join(...).
# path.join costruisce percorsi di file in modo compatibile con qualsiasi sistema operativo.

from collections import deque
# 'deque' (double-ended queue) è una lista ottimizzata per aggiungere/rimuovere
# elementi sia in testa che in coda in tempo O(1) (velocissimo).
# Lo usiamo per: lo stack dei frame visivi (SmartFrameStacker),
# la storia delle vittorie nel curriculum, e il buffer N-step.
# Il parametro maxlen=N fa sì che quando è piena, elimina automaticamente l'elemento più vecchio.

# =============================================================================
# OTTIMIZZAZIONI AMBIENTE E CPU
# =============================================================================
# Queste righe impostano variabili d'ambiente PRIMA di importare le librerie pesanti.

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# TensorFlow (se installato nel sistema) usa oneDNN per ottimizzazioni CPU.
# Disabilitarlo evita messaggi di log fastidiosi e conflitti con PyTorch.
# "0" = disabilitato, "1" = abilitato.

os.environ["KMP_DUPLICATE_LIB_OK"]  = "True"
# Su macOS e Windows, alcune librerie matematiche (come libiomp) vengono caricate due volte
# da dipendenze diverse, causando un crash con errore "OMP: Error #15".
# Questa variabile dice a OpenMP di ignorare il problema e procedere comunque.

os.environ["OMP_NUM_THREADS"]       = "1"
# OpenMP è una libreria per il calcolo parallelo su CPU.
# Impostare a "1" significa usare un solo thread per le operazioni matematiche.
# Questo evita che NumPy/PyTorch creino troppi thread in background, che su sistemi
# multi-processo (come il training RL) crea overhead e rallentamenti.

os.environ["MKL_NUM_THREADS"]       = "1"
# Stesso concetto di OMP_NUM_THREADS, ma per la libreria MKL di Intel
# (Math Kernel Library), usata da NumPy per operazioni lineari vettorizzate.
# "1" thread evita conflitti quando si gestiscono già più processi di training.

# =============================================================================
# FLAG PRINCIPALE: PLAY vs TRAIN
# =============================================================================

PLAY_MODE = True
# Variabile booleana che controlla la modalità del programma:
#   True  → Modalità PLAY: mostra la finestra, l'AI gioca visivamente usando i pesi salvati
#   False → Modalità TRAIN: addestra la rete, preferibilmente headless (senza finestra)
# Cambiare questo flag è il modo principale per passare dall'addestramento alla demo.

HEADLESS_TRAINING = not PLAY_MODE
# Derivato automaticamente da PLAY_MODE.
# Se PLAY_MODE è False → HEADLESS_TRAINING è True.
# In modalità headless, Pygame gira senza finestra visibile, il che permette
# di eseguire migliaia di frame al secondo invece dei 60 fps standard.

if HEADLESS_TRAINING:
    # Questo blocco viene eseguito SOLO se siamo in modalità training senza interfaccia.
    
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    # SDL è la libreria C su cui si basa Pygame per gestire la grafica.
    # Il driver "dummy" fa finta di avere una scheda video, ma non disegna nulla.
    # Questo permette a Pygame di funzionare su server senza schermo (come cloud AWS/Colab).
    
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    # Come sopra, ma per l'audio. Previene crash su sistemi senza scheda audio.
    
    print("[INIT] Modalita HEADLESS (no finestra) — TRAINING MODE")
    # Stampa un messaggio di conferma nella console all'avvio.
else:
    print("[INIT] Modalita DISPLAY (finestra visibile) — PLAY MODE")
    # Modalità normale con finestra visibile.

# =============================================================================
# IMPORT LIBRERIE SCIENTIFICHE E DI GIOCO
# =============================================================================

import numpy as np
# NumPy è la libreria fondamentale per il calcolo numerico in Python.
# Fornisce array N-dimensionali molto efficienti (np.array) e operazioni matematiche vettorizzate.
# Lo usiamo per: creare array di stati, manipolare immagini come matrici, calcolare statistiche.

try:
    import torch
    # PyTorch è il framework di deep learning principale del progetto.
    # Gestisce: tensori (simili agli array NumPy ma su GPU), reti neurali, backpropagation automatica.
    # torch.Tensor è la struttura dati fondamentale: può vivere su CPU o GPU (CUDA/MPS).
    
    import torch.optim as optim
    # Sottomodulo di PyTorch con gli ottimizzatori.
    # Gli ottimizzatori aggiornano i pesi della rete dopo ogni backward pass.
    # Esempi: Adam, AdamW, SGD, RMSprop.
    # Useremo AdamW: come Adam ma con weight decay (penalizza pesi troppo grandi, riduce overfitting).
    
except ImportError:
    pass
# Se PyTorch non è installato, il blocco try fallisce e 'pass' ignora l'errore silenziosamente.
# Questo permette al gioco di avviarsi in modalità umana anche senza PyTorch installato.

import pygame
# Pygame è la libreria che gestisce tutto il motore di gioco:
#   - Finestra e rendering grafico
#   - Gestione degli eventi (tastiera, mouse, chiusura finestra)
#   - Sistema audio (musica e effetti sonori)
#   - Timer personalizzati (USEREVENT) usati per sincronizzare fisica e animazioni
#   - Clock per limitare i frame per secondo

from pygame.locals import *
# Importa TUTTE le costanti di Pygame nello spazio dei nomi locale.
# Questo ci permette di scrivere K_UP invece di pygame.K_UP, QUIT invece di pygame.QUIT.
# Le costanti più usate nel codice:
#   K_UP, K_DOWN, K_LEFT, K_RIGHT, K_w, K_a, K_s, K_d → tasti direzionali e WASD
#   K_RETURN → tasto Invio
#   K_ESCAPE → tasto Escape
#   K_r      → tasto R (restart)
#   KEYDOWN  → evento "tasto premuto"
#   QUIT     → evento "chiudi finestra"
#   SRCALPHA → flag per superfici con canale alpha (trasparenza)

# =============================================================================
# IMPORT MODULI PERSONALIZZATI DEL GIOCO
# =============================================================================

from character import Character
# Importa la classe Character dal file character.py.
# Character gestisce tutto il personaggio giocante (Mr. Driller):
#   - Posizione nella griglia (posAcc() restituisce y, x)
#   - Vite rimaste (livesAcc())
#   - Ossigeno (oxyAcc(), updateOxygen())
#   - Stato animazione (Anim(), IdlingAcc() → True se fermo)
#   - Movimento e caduta (fall(), NeedToIdle())
#   - Morte e respawn (revive())
#   - Punteggio (scoreAcc())

from block import Classic, Pill, Solo, Delayed, clear_cache
# Importa le classi dei vari tipi di blocchi dal file block.py.
# Ogni classe rappresenta un tipo di blocco con comportamento diverso:
#   Classic    → blocco normale colorato; si attacca ai vicini dello stesso colore (gluing)
#   Pill       → speciale, dà ossigeno al giocatore quando distrutto
#   Solo       → blocco singolo, non si attacca agli altri, cade sempre
#   Delayed    → blocco a tempo, scompare dopo un delay quando il suo timer scade
#   clear_cache → funzione che svuota la cache delle texture grafiche dei blocchi
# Ogni blocco ha metodi comuni:
#   hpAccess() → punti vita (0 = vuoto/aria)
#   typeAccess() → stringa del tipo ("classic", "solo", "delayed", "unbreakable")
#   isFalling() / isShaking() → stato fisico del blocco
#   startShaking() / stopFalling() → cambio di stato fisico
#   updatePos(x, y) → aggiorna coordinate interne
#   updCoText(level) → aggiorna connessioni visive con blocchi vicini
#   updOffset(offset) → aggiorna lo spostamento visivo per la telecamera scorrevole

from menu import changeLvl, restart, mainMenu
# Importa le funzioni di navigazione dal file menu.py:
#   mainMenu(surface, option) → disegna la schermata del menu principale
#   restart(player)           → crea una nuova mappa di gioco dal livello 1, resetta il player
#                               ritorna (level, levelID, won)
#   changeLvl(levelID, player, is_ai) → passa al livello successivo
#                               ritorna (level, levelID, game_completed)

# =============================================================================
# IMPORT COMPONENTI AI
# =============================================================================

try:
    from ai_agent import (
        DrillerDuelingDQN,
        # La rete neurale principale.
        # Architettura Dueling DQN:
        #   - Uno stream "Value" → quanto è buono stare in questo stato?
        #   - Uno stream "Advantage" → quanto vale questa azione rispetto alle altre?
        # L'input è: un tensore visivo (finestra intorno al player) + un vettore di variabili interne
        # L'output è: un Q-value per ognuna delle 6 azioni possibili
        
        PrioritizedReplayMemory as ReplayMemory,
        # Buffer di Replay con Priorità (PER = Prioritized Experience Replay).
        # Invece di campionare transizioni in modo casuale, campiona più spesso
        # quelle con errore TD (temporal difference) alto = le più "sorprendenti" o difficili.
        # Questo velocizza l'apprendimento focalizzandosi sulle esperienze più informative.
        # Salvata con l'alias ReplayMemory per uniformità nel codice.
        
        NStepBuffer,
        # Buffer che accumula N transizioni consecutive prima di inviarle alla memoria.
        # Invece di stimare la reward su 1 passo avanti (standard DQN),
        # calcola la reward accumulata sui prossimi N passi (N=8 qui).
        # La formula è: R = r0 + γ*r1 + γ²*r2 + ... + γ^(N-1)*r(N-1) + γ^N * Q(s_N)
        # Questo riduce la varianza della stima e accelera la propagazione della reward.
        
        RewardShaper,
        # Sistema di Reward Shaping (PBRS = Potential-Based Reward Shaping).
        # Aggiunge un bonus/malus artificiale per guidare l'agente verso l'obiettivo.
        # In particolare: premia il giocatore quando scende in profondità (si avvicina al fondo),
        # e lo penalizza se risale. Questo accelera enormemente l'apprendimento iniziale.
        
        DrillTracker,
        # Sistema anti-loop per punire comportamenti ripetitivi inutili.
        # Traccia: quante volte consecutive l'agente ha provato a fare azioni senza effetto,
        # se sta oscillando tra 2 celle (su-giù o sinistra-destra senza mai trivellare),
        # se è rimasto fermo troppo a lungo.
        # Genera penalità nella reward per scoraggiare questi pattern.
        
        select_action,
        # Funzione che implementa la policy epsilon-greedy:
        #   Con probabilità epsilon → sceglie azione casuale (esplorazione)
        #   Con probabilità (1-epsilon) → usa la rete neurale (sfruttamento)
        # Ritorna: (tensore_azione, array_q_values)
        # L'azione è un intero 0-5 che corrisponde a una delle 6 mosse.
        
        optimize_model,
        # Il cuore dell'apprendimento. Questa funzione:
        #   1. Campiona un batch di transizioni dalla memoria PER
        #   2. Calcola i Q-values predetti dalla Policy Net
        #   3. Calcola i Q-values target dalla Target Net (con formula Bellman)
        #   4. Calcola la loss (Huber Loss / Smooth L1)
        #   5. Esegue la backpropagation (calcola gradienti)
        #   6. Applica il gradient clipping (evita esplosione dei gradienti)
        #   7. Aggiorna i pesi con l'ottimizzatore
        #   8. Aggiorna le priorità nel buffer PER
        # Ritorna: (loss_value, mean_q_value) o None se il batch è troppo piccolo.
        
        calculate_reward,
        # Funzione molto complessa che calcola lo "stipendio" dell'agente ad ogni mossa.
        # Considera decine di fattori:
        #   - Ha perso ossigeno? Penalità progressiva.
        #   - Si è mosso più in basso? Bonus.
        #   - Ha aumentato il punteggio? Bonus.
        #   - Ha perso una vita? Penalità grande.
        #   - Ha completato il livello? Bonus enorme.
        #   - Ha trivellato un blocco difficile? Bonus speciale.
        #   - Sta oscillando senza progredire? Penalità crescente.
        #   - Ha schivato un blocco in caduta? Bonus.
        
        device,
        # Stringa o oggetto torch.device che indica dove eseguire i calcoli:
        #   "cpu"  → su processore (sempre disponibile, lento)
        #   "cuda" → su GPU NVIDIA (molto veloce)
        #   "mps"  → su GPU Apple Silicon (Mac M1/M2/M3)
        # Determinato automaticamente da ai_agent.py in base all'hardware disponibile.
        
        get_local_window_tensor,
        # Genera il tensore visivo che rappresenta ciò che "vede" l'agente.
        # Crea una finestra 11 righe x 9 colonne centrata sul player.
        # Ogni cella è codificata su 6 canali (come i "colori" di un'immagine):
        #   Canale 0: tipo del blocco (classic, solo, delayed, ecc.)
        #   Canale 1: HP del blocco (salute)
        #   Canale 2: colore del blocco
        #   Canale 3: flag "sta cadendo?"
        #   Canale 4: flag "sta tremando?"
        #   Canale 5: posizione relativa del player (sempre al centro)
        # Output: tensore PyTorch di shape (6, 11, 9)
        
        get_internal_state_vector
        # Genera il vettore delle variabili interne dello stato dell'agente.
        # Contiene informazioni che non si vedono visivamente:
        #   - Ossigeno attuale (normalizzato 0-1)
        #   - Vite rimaste
        #   - Profondità attuale nella mappa
        #   - Azione precedente (one-hot encoding)
        #   - Punteggio normalizzato
        #   - ecc.
        # Output: tensore PyTorch di dimensione STATE_DIM (32).
    )
    AI_AVAILABLE = True
    # Flag: se tutti gli import hanno avuto successo, l'AI è utilizzabile.
    print("[INIT] AI caricata correttamente")
    
except ImportError as e:
    # Se uno qualsiasi degli import sopra fallisce (file mancante, errore di sintassi,
    # dipendenza mancante), AI_AVAILABLE diventa False e il gioco gira senza AI.
    print(f"[INIT] AI non disponibile: {e}")
    AI_AVAILABLE = False

# =============================================================================
# IPERPARAMETRI v6.9
# Questi sono i "parametri di configurazione" del training.
# Cambiarli modifica il comportamento dell'addestramento.
# =============================================================================

BATCH_SIZE    = 128
# Numero di transizioni campionate dalla memoria in ogni step di training.
# Un batch più grande → gradiente più stabile ma più lento.
# Un batch più piccolo → più rumoroso ma aggiorna più spesso.
# 128 è un buon compromesso per questo tipo di gioco.

GAMMA         = 0.99
# Fattore di sconto (discount factor). Range: 0.0 - 1.0
# Indica quanto l'agente "valorizza" le reward future rispetto a quelle immediate.
# 0.99 → le reward future (anche fino a 100 passi avanti) hanno ancora molto peso.
# 0.5  → l'agente è "miope", guarda quasi solo all'immediato.
# Formula: Q(s,a) = r + γ * max_a' Q(s', a')

LEARNING_RATE = 2e-5
# Tasso di apprendimento dell'ottimizzatore AdamW.
# Controlla quanto "in grande" spostiamo i pesi dopo ogni backpropagation.
# 2e-5 = 0.00002 → piccoli aggiustamenti, più stabili ma convergenza più lenta.
# Troppo alto → oscilla, non converge. Troppo basso → impara troppo lentamente.

EPS_START     = 0.05
# Epsilon iniziale per la strategia epsilon-greedy.
# 0.05 = 5% di probabilità di scegliere un'azione casuale.
# Qui è già basso perché si ipotizza di riprendere un training avanzato da checkpoint.
# Se partissimo da zero, useremmo tipicamente EPS_START = 1.0.

EPS_END       = 0.02
# Epsilon minimo assoluto verso cui tende il decadimento.
# Non scenderà mai sotto il 2% di casualità.
# Un po' di esplorazione residua previene che l'agente si "fossilizzi" su una policy sub-ottima.

EPS_FLOOR     = 0.05
# Epsilon minimo durante il training standard (più alto di EPS_END).
# Se l'epsilon decadrebbe sotto questo valore, viene tenuto qui.
# Garantisce un minimo di esplorazione continua durante tutto l'addestramento.

TARGET_UPDATE = 1000
# Ogni quanti step copiare i pesi della Policy Net nella Target Net.
# La Target Net è una copia "stabile" usata per calcolare i Q-values target nella loss.
# Aggiornarla troppo spesso → instabilità (target che si muove mentre si impara).
# Aggiornarla troppo di rado → i target diventano obsoleti.
# 1000 step è un valore classico e robusto.

MEMORY_SIZE   = 50_000
# Capienza massima del Replay Buffer (quante transizioni memorizzare).
# Quando è pieno, le transizioni più vecchie vengono sovrascritte (FIFO con priorità).
# 50.000 transizioni bilanciano diversità delle esperienze e uso della RAM.

N_STEP        = 8
# Numero di passi futuri per il calcolo della reward N-step.
# R_8step = r0 + 0.99*r1 + 0.99²*r2 + ... + 0.99^7*r7 + 0.99^8 * Q(s8)
# Con N=8, l'agente impara guardando le conseguenze a medio termine delle sue azioni.
# Riduce il bias della stima ma aumenta la varianza (trade-off standard in RL).

MODEL_PATH    = "RenfLearningAgent/driller_ai_v31.pth"
# Percorso del file dove vengono salvati/caricati i pesi del modello (checkpoint).
# Il formato .pth è il formato nativo di PyTorch per serializzare modelli e ottimizzatori.
# Contiene: pesi rete, stato ottimizzatore, numero di step, epsilon, stato curriculum.

MAX_LEVEL     = 10
# Numero totale di livelli nel gioco.
# Usato in PLAY_MODE per determinare quando il gioco è completato.

N_CHANNELS    = 6
# Numero di canali per frame visivo dell'AI (come i canali RGB di un'immagine, ma 6).
# Ogni canale codifica una caratteristica diversa della griglia di gioco.

N_FRAMES      = 4
# Numero di frame impilati per dare all'AI la percezione del movimento nel tempo.
# L'AI riceve in input gli ultimi 4 frame contemporaneamente.
# Così può capire: "questo blocco si stava muovendo verso di me negli ultimi 4 tick?"
# Tecnica classica in DQN (inventata da DeepMind per Atari).

INPUT_SHAPE   = (N_CHANNELS * N_FRAMES, 11, 9)
# Shape del tensore di input alla rete neurale: (24, 11, 9)
# 24 canali = 6 canali × 4 frame
# 11 righe × 9 colonne = la finestra visiva centrata sul player

STATE_DIM     = 32
# Dimensione del vettore di stato interno (variabili numeriche non visive).
# Questo vettore viene concatenato all'output visivo nella rete neurale.

TRAIN_EVERY   = 3
# Ogni quante azioni di gioco eseguire una backpropagation.
# 3 → per ogni 3 mosse dell'AI, viene eseguito 1 update della rete.
# Più basso → più aggiornamenti ma più overhead computazionale.

WARMUP_STEPS  = 2_000
# Numero di step iniziali in cui NON viene eseguita la backpropagation.
# Durante il warmup, l'AI gioca (casualmente o con epsilon alto) per riempire la memoria.
# Questo evita di imparare da un buffer quasi vuoto, il che causerebbe overfitting su poche esperienze.
# Dopo 2000 step, il buffer è abbastanza vario da iniziare ad imparare.

BUFFER_DIR        = "RenfLearningAgent/driller_buffer_v31"
# Cartella su disco dove viene salvato il Replay Buffer in modo persistente.
# Il buffer viene serializzato in file .npz (formato NumPy compresso).
# Questo permette di riprendere il training esattamente da dove ci si era fermati.

BUFFER_SAVE_EVERY = 5_000
# Ogni quanti step salvare il buffer su disco.
# Il salvataggio è asincrono (su un thread separato) per non bloccare il gioco.

BUFFER_MAX_LOAD   = 20_000
# Limite di transizioni da ricaricare dal disco all'avvio.
# Evita di caricare l'intero buffer (che potrebbe essere da GB) consumando tutta la RAM.
# Carica solo le 20.000 transizioni più recenti.

MAX_CONSEC_FAILS = 4
# Numero massimo di fallimenti consecutivi tollerati prima di forzare la fase di Learning.
# Se l'AI prova 4 volte di fila ad eseguire mosse senza effetto (es. colpire l'aria),
# viene forzata a "riflettere" (fase LEARNING) anche se la gravità non ha ancora completato il ciclo.
# Previene stalli infiniti nel training.

EPS_CHECK_INTERVAL   = 2_000
# Ogni quanti step controllare le performance dell'agente per ricalibrare epsilon.
# Questo è il meccanismo di "Adaptive Epsilon" che reagisce ai periodi di scarso apprendimento.

EPS_BOOST_THRESHOLD  = 0.15
# Soglia del win-rate sotto la quale scatta il boost di epsilon.
# Se nelle ultime 20 partite l'AI ha vinto meno del 15% delle volte...

EPS_BOOST_VALUE      = 0.15
# ...epsilon viene alzato al 15% per forzare più esplorazione.
# L'idea è: se stai perdendo sempre, smetti di sfruttare la policy attuale (che è scarsa)
# e inizia ad esplorare soluzioni diverse.

EPS_BOOST_MIN_EP     = 20
# Il boost epsilon si attiva solo dopo aver giocato almeno 20 episodi.
# Con meno episodi, il win_rate è statisticamente inaffidabile.

OXY_DEATH_THRESHOLD = 2
# Soglia di ossigeno per classificare una morte come "oxy" (mancanza d'aria).
# Se l'ossigeno era ≤ 2 quando il player ha perso una vita, la causa è mancanza di ossigeno.
# Il valore 2 (invece di 0) tiene conto del fatto che l'ossigeno potrebbe scendere
# da 1 a 0 e innescare la morte nell'istante tra il calo e la lettura del valore.

sys.setrecursionlimit(5000)
# Python per default limita la profondità di ricorsione a 1000 livelli.
# Alcune funzioni di Pygame o del motore grafico potrebbero superare questo limite
# in casi di mappe complesse o callback annidati.
# 5000 è un valore sicuro senza esagerare (troppo alto esaurisce lo stack di sistema).

# =============================================================================
# DIMENSIONI SCHERMO
# =============================================================================

GAME_W, UI_W       = 600, 200
# GAME_W: larghezza in pixel dell'area di gioco (la griglia dei blocchi).
# UI_W: larghezza in pixel del pannello UI laterale (punteggio, ossigeno, vite).

SCREEN_W, SCREEN_H = GAME_W + UI_W, 600
# SCREEN_W = 600 + 200 = 800 pixel di larghezza totale della finestra.
# SCREEN_H = 600 pixel di altezza.
# La risoluzione finale è quindi 800x600.

GAME_SPEED = 1.0
# Moltiplicatore di velocità generale del gioco in PLAY_MODE.
# 1.0 = velocità normale, 2.0 = doppia velocità.
# Viene usato per scalare i timer sottostanti.
# Usato principalmente per demo o debug, non influisce sul training headless.

# Timer degli eventi Pygame, scalati da GAME_SPEED:
_T_GRAVITY = 400.0 / GAME_SPEED
# Intervallo in ms tra i tick di gravità (400ms → 2.5 volte al secondo).
# Ogni tick, i blocchi sospesi nel vuoto avanzano di una fase (fermo → tremante → cadente).

_T_ANIM    = 80.0  / GAME_SPEED
# Intervallo in ms per l'aggiornamento delle animazioni sprite (80ms → ~12.5 fps animazione).

_T_SEC     = 1050.0 / GAME_SPEED
# "Secondo di gioco": ogni 1050ms viene sottratto 1 punto di ossigeno al player.
# 1050ms invece di 1000ms dà un piccolo margine di tolleranza rispetto al secondo reale.

_T_PLY     = 120.0  / GAME_SPEED
# Intervallo per l'update dello stato "idling" del player (120ms → ~8 volte al secondo).
# Dopo un'azione, il player impiega ~120ms per tornare allo stato "idle" (pronto ad agire).


# =============================================================================
# CURRICULUM MANAGER
# =============================================================================

class CurriculumManager:
    """
    Gestisce il Curriculum Learning.
    
    FILOSOFIA:
        Imparare a trivellare 10 livelli complessi tutti insieme è difficilissimo.
        Il Curriculum Learning risolve questo problema dividendo il compito in tappe:
        l'AI prima padroneggia il livello 1 (semplice), poi affronta il livello 3,
        e così via fino al livello 10.
        
    FUNZIONAMENTO:
        - L'AI gioca sempre a partire dal livello 1.
        - Il "livello massimo corrente" determina fino a quale livello può avanzare prima di resettare.
        - Per "sbloccare" il livello successivo, deve raggiungere un win-rate minimo su N partite.
        - Il win-rate viene calcolato sulle ultime min_episodes partite (finestra scorrevole).
    
    STATI POSSIBILI:
        current_max_level = 1  → può giocare solo il livello 1
        current_max_level = 3  → può giocare i livelli 1, 2, 3
        current_max_level = 10 → può giocare tutti i livelli
        training_complete = True → ha vinto con win_rate ≥ 90% su 30 partite al livello 10
    """
    
    _PROGRESSIONS = [
        (1,  3,   0.80),
        # Tupla: (livello_corrente, livello_da_sbloccare, win_rate_richiesto)
        # Per sbloccare il livello 3, il win-rate al livello 1 deve superare l'80%.
        
        (3,  5,   0.80),
        # Per sbloccare il livello 5, win-rate al livello 3 ≥ 80%.
        
        (5,  7,   0.82),
        # Per sbloccare il livello 7, win-rate al livello 5 ≥ 82%.
        # La soglia sale leggermente perché i livelli diventano più difficili.
        
        (7,  10,  0.85),
        # Per sbloccare il livello 10, win-rate al livello 7 ≥ 85%.
        
        (10, None, 0.90),
        # L'ultimo passo: None indica "nessun livello successivo".
        # Con win-rate ≥ 90% al livello 10, il training è completato.
    ]

    def __init__(self, min_episodes: int = 30):
        """
        Costruttore della classe.
        
        Argomenti:
            min_episodes: numero minimo di partite da giocare prima che il win-rate
                          sia considerato statisticamente affidabile per la valutazione.
        """
        self.min_episodes      = min_episodes
        # Finestra minima di episodi per il calcolo del win-rate.
        # Con meno di 30 partite, il win-rate potrebbe essere 100% per pura fortuna.
        
        self.current_max_level = 1
        # Livello massimo inizialmente sbloccato: solo il livello 1.
        
        self.training_complete = False
        # Flag che diventa True quando tutto il curriculum è completato con successo.
        
        self._history: dict    = {}
        # Dizionario che mappa livello → deque di risultati (1=vinto, 0=perso).
        # Esempio: {1: deque([1, 0, 1, 1, 0, ...]), 3: deque([1, 1, 0, ...])}

    def record_episode(self, completed_max: bool) -> None:
        """
        Registra il risultato dell'episodio appena terminato.
        
        Argomenti:
            completed_max: True se l'episodio è stato vinto raggiungendo il livello massimo,
                           False se l'episodio è terminato con Game Over.
        """
        key = self.current_max_level
        # La chiave del dizionario è il livello massimo corrente.
        
        if key not in self._history:
            self._history[key] = deque(maxlen=500)
        # Crea la deque per questo livello se non esiste ancora.
        # maxlen=500: conserva solo gli ultimi 500 risultati, i più vecchi vengono scartati.
        
        self._history[key].append(1 if completed_max else 0)
        # Aggiunge 1 per vittoria, 0 per sconfitta alla storia del livello corrente.

    def check_and_advance(self) -> bool:
        """
        Verifica se il win-rate ha superato la soglia e, in caso affermativo,
        sblocca il livello successivo.
        
        Ritorna:
            True se il training è completato (curriculum finito), False altrimenti.
        """
        if self.training_complete:
            return True
        # Se il training è già completato, non c'è nulla da verificare.

        for cur, nxt, threshold in self._PROGRESSIONS:
            # Scorre le tappe di progressione definite sopra.
            
            if self.current_max_level != cur:
                continue
            # Salta le tappe che non corrispondono al livello corrente.
            # Così trova la tappa giusta in O(n) dove n è il numero di tappe (piccolo).

            hist = list(self._history.get(cur, deque()))
            # Recupera la storia degli episodi per questo livello come lista Python.
            # Se non esiste ancora, usa una deque vuota come default.
            
            if len(hist) < self.min_episodes:
                break
            # Non abbastanza episodi per una valutazione affidabile: esce dal loop.

            recent   = hist[-self.min_episodes:]
            # Prende solo gli ultimi min_episodes risultati (finestra scorrevole).
            # Esempio: se min_episodes=30 e ho 500 risultati, considera solo gli ultimi 30.
            
            win_rate = sum(recent) / self.min_episodes
            # Calcola il win-rate: somma delle vittorie (1) diviso il numero di episodi.
            # Esempio: [1,0,1,1,1,0,1,1,1,1,...] su 30 episodi → win_rate = 0.80

            if win_rate >= threshold:
            # Soglia superata! Avanza di livello.
                
                if nxt is None:
                # Nessun livello successivo → training completato!
                    self.training_complete = True
                    print(f"\n[CURRICULUM] ★★★  TRAINING COMPLETO!  ★★★\n")
                    return True
                
                old = self.current_max_level
                # Salva il livello vecchio per il messaggio di log.
                
                self.current_max_level = nxt
                # Aggiorna il livello massimo sbloccato al nuovo.
                
                self._history.setdefault(nxt, deque(maxlen=500))
                # Prepara la deque per il nuovo livello (vuota, pronta a ricevere episodi).
                # setdefault: crea la chiave solo se non esiste già.
                
                print(f"[CURRICULUM] Livello {old} → {win_rate:.1%} ≥ {threshold:.0%}"
                      f" — ✓ Sblocco livelli fino a {nxt}!")
                # Esempio output: "[CURRICULUM] Livello 1 → 82.0% ≥ 80% — ✓ Sblocco livelli fino a 3!"
                
            else:
                n_ep = len(hist)
                # Numero totale di episodi giocati (può essere > min_episodes).
                
                print(f"[CURRICULUM] Livello {cur} | "
                      f"ep={n_ep} | win={win_rate:.1%} "
                      f"(serve {threshold:.0%} su {self.min_episodes} ep)")
                # Stampa lo stato corrente del curriculum: quanti episodi, quale win-rate,
                # quale soglia deve superare. Utile per monitorare i progressi.
            break
        # Importante: il break alla fine del for evita di processare più tappe nella stessa chiamata.
        # Vogliamo avanzare di un livello alla volta per chiarezza.

        return False
        # Training non ancora completato.

    def get_win_rate(self) -> float:
        """
        Calcola e restituisce il win-rate al livello corrente.
        Usa al massimo gli ultimi min_episodes episodi.
        
        Ritorna:
            float tra 0.0 (perso sempre) e 1.0 (vinto sempre).
        """
        hist = list(self._history.get(self.current_max_level, deque()))
        if not hist:
            return 0.0
        # Se non ci sono ancora episodi, win-rate = 0.
        
        recent = hist[-self.min_episodes:] if len(hist) >= self.min_episodes else hist
        # Se ho abbastanza episodi, usa gli ultimi min_episodes.
        # Altrimenti usa tutti quelli disponibili.
        
        return sum(recent) / len(recent)
        # Win-rate come rapporto vittorie / totale.

    def get_short_win_rate(self, n: int = 20) -> float:
        """
        Calcola il win-rate sulle ultimissime n partite (finestra molto breve).
        Usato per il meccanismo di boost epsilon: reagisce velocemente a un calo di performance.
        
        Argomenti:
            n: numero di episodi recenti da considerare (default 20).
        
        Ritorna:
            float tra 0.0 e 1.0.
        """
        hist = list(self._history.get(self.current_max_level, deque()))
        if not hist:
            return 0.0
        recent = hist[-n:]
        # Prende solo gli ultimi n episodi, anche se ho una storia più lunga.
        return sum(recent) / len(recent)

    def n_episodes(self) -> int:
        """
        Ritorna il numero totale di episodi giocati al livello corrente.
        Usato per log e visualizzazione nell'HUD di debug.
        """
        return len(self._history.get(self.current_max_level, deque()))

    def state_dict(self) -> dict:
        """
        Serializza lo stato del CurriculumManager in un dizionario salvabile.
        Questo dizionario viene incluso nel checkpoint .pth del modello.
        
        Ritorna:
            dict con tutti i campi necessari per ripristinare lo stato completo.
        """
        return {
            "current_max_level": self.current_max_level,
            # Il livello massimo corrente → sappiamo da dove riprendere.
            
            "training_complete": self.training_complete,
            # Flag di completamento → se True, non serve riaddestrare.
            
            "history":           {str(k): list(v) for k, v in self._history.items()},
            # Converte le deque in liste e le chiavi int in stringhe per compatibilità JSON.
            # torch.save può salvare dizionari Python, ma non oggetti deque direttamente.
            
            "min_episodes":      self.min_episodes,
            # Salva anche questo valore per ricaricarlo correttamente.
        }

    def load_state_dict(self, d: dict) -> None:
        """
        Ripristina lo stato del CurriculumManager dal dizionario caricato dal checkpoint.
        
        Argomenti:
            d: dizionario precedentemente salvato con state_dict().
        """
        self.current_max_level = int(d.get("current_max_level", 1))
        # Ripristina il livello massimo. Se la chiave non esiste nel dizionario
        # (checkpoint vecchio), usa il default 1.
        
        self.training_complete = bool(d.get("training_complete", False))
        # Ripristina il flag di completamento.
        
        self.min_episodes      = int(d.get("min_episodes", self.min_episodes))
        # Ripristina il numero minimo di episodi.
        
        for k, v in d.get("history", {}).items():
            self._history[int(k)] = deque(v, maxlen=500)
        # Riconverte le chiavi stringa in int e le liste in deque con maxlen=500.
        # Se "history" non esiste nel dizionario, usa {} come default (nessuna storia).
        
        print(f"[CURRICULUM] Caricato → max_level={self.current_max_level}, "
              f"ep={self.n_episodes()}, win={self.get_win_rate():.1%}")
        # Conferma visiva del caricamento avvenuto con i valori principali.


# =============================================================================
# FUNZIONI DI SUPPORTO (Helper Functions)
# =============================================================================

def save_checkpoint(model, optimizer, steps, epsilon, scheduler=None, curriculum=None):
    """
    Salva un checkpoint completo del training su disco.
    
    Un checkpoint contiene tutto il necessario per riprendere il training esattamente
    da dove era stato interrotto: pesi della rete, stato ottimizzatore, step corrente,
    epsilon corrente, stato dello scheduler, stato del curriculum.
    
    Il salvataggio avviene in un thread separato per non bloccare il gioco.
    
    Argomenti:
        model:      la Policy Net (DrillerDuelingDQN) di cui salvare i pesi
        optimizer:  l'ottimizzatore AdamW di cui salvare lo stato
        steps:      numero di step completati fino ad ora
        epsilon:    valore corrente dell'epsilon greedy
        scheduler:  (opzionale) lo scheduler del learning rate
        curriculum: (opzionale) il CurriculumManager con la storia degli episodi
    """
    ckpt = {
        "model_state_dict":     model.state_dict(),
        # state_dict() restituisce un dizionario {nome_layer: tensore_pesi}.
        # Contiene tutti i pesi e bias della rete neurale.
        
        "optimizer_state_dict": optimizer.state_dict(),
        # Stato interno dell'ottimizzatore: learning rate corrente, momentum,
        # medie mobili dei gradienti (per Adam), ecc.
        
        "steps":                steps,
        # Numero totale di azioni/step completati. Usato per la schedulazione di
        # epsilon, target update, salvataggi periodici.
        
        "epsilon":              epsilon
        # Valore corrente dell'epsilon greedy. Riprendere con l'epsilon corretto
        # evita di ricominciare l'esplorazione da capo.
    }
    
    if scheduler:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
        # Salva lo stato dello scheduler: a che punto è nella curva di coseno.
        # Così il learning rate riparte dal punto giusto invece che dall'inizio.
    
    if curriculum is not None:
        ckpt["curriculum"] = curriculum.state_dict()
        # Incorpora lo stato del curriculum nel checkpoint.

    def _write():
        """Funzione interna eseguita nel thread di salvataggio."""
        torch.save(ckpt, MODEL_PATH)
        # torch.save serializza il dizionario Python in formato binario .pth.
        # È equivalente a pickle ma ottimizzato per i tensori PyTorch.
        
        curr_info = ""
        if curriculum is not None:
            curr_info = (f" | curriculum max_lvl={curriculum.current_max_level}"
                         f" win={curriculum.get_win_rate():.1%}")
        # Prepara una stringa informativa sul curriculum per il log.
        
        print(f"[AI] Checkpoint salvato — step {steps}, e {epsilon:.4f}{curr_info}")
        # Log di conferma del salvataggio.

    threading.Thread(target=_write, daemon=True).start()
    # Avvia _write() in un thread daemon separato.
    # daemon=True → il thread si chiude automaticamente se il programma principale termina,
    # senza bloccare la chiusura con salvataggi in corso.


def load_checkpoint(model, optimizer, scheduler=None):
    """
    Carica un checkpoint esistente e ripristina lo stato del training.
    
    Se il file non esiste o è incompatibile (architettura cambiata),
    ritorna i valori di default per ricominciare da zero.
    
    Argomenti:
        model:     la Policy Net in cui caricare i pesi
        optimizer: l'ottimizzatore in cui caricare lo stato
        scheduler: (opzionale) lo scheduler da ripristinare
    
    Ritorna:
        (start_steps, start_eps): numero di step da cui ripartire ed epsilon iniziale
    """
    start_steps, start_eps = 0, EPS_START
    # Valori di default: ricomincia da zero con epsilon iniziale.

    if os.path.exists(MODEL_PATH):
    # Controlla se il file .pth esiste prima di tentare il caricamento.
        
        print(f"[AI] Caricamento da {MODEL_PATH}...")
        try:
            ckpt = torch.load(MODEL_PATH, map_location=device)
            # torch.load deserializza il file .pth.
            # map_location=device sposta i tensori sul dispositivo corretto
            # (utile se salvato su GPU ma si vuole caricare su CPU).
            
            try:
                model.load_state_dict(ckpt["model_state_dict"], strict=True)
                # strict=True → carica i pesi SOLO se l'architettura corrisponde esattamente.
                # Se un layer è stato aggiunto/rimosso/ridimensionato, lancia RuntimeError.
            except RuntimeError:
                print("[AI] Architettura cambiata, riparto da zero")
                return 0, EPS_START
            # Se l'architettura non corrisponde (versione del modello cambiata),
            # ignoriamo il checkpoint e ripartiamo con pesi casuali.

            if optimizer:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    # Ripristina lo stato dell'ottimizzatore.
                except Exception:
                    pass
                # Se fallisce (es. numero di parametri diverso), continua senza crash.

            if scheduler and "scheduler_state_dict" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    # Ripristina la posizione dello scheduler nella curva di coseno.
                except Exception:
                    pass

            start_steps = ckpt.get("steps", 0)
            # Ripristina il numero di step. .get con default 0 per compatibilità con checkpoint vecchi.
            
            start_eps   = max(EPS_FLOOR, EPS_END)
            # Non far scendere epsilon sotto il floor configurato.
            # Anche se nel checkpoint epsilon era 0.01, ripartiamo da EPS_FLOOR (0.05).
            
            print(f"[AI] Caricato! Step: {start_steps}, e: {start_eps:.4f}")
        except Exception as e:
            print(f"[AI] Errore: {e}, riparto da zero")
            # Qualsiasi altro errore (file corrotto, versione incompatibile di PyTorch):
            # stampa l'errore e riparte da zero.
    else:
        print("[AI] Nessun salvataggio, pesi casuali")
        # Il file non esiste: prima esecuzione, pesi inizializzati casualmente.

    return start_steps, start_eps


def _load_curriculum_from_checkpoint(curriculum: CurriculumManager) -> None:
    """
    Carica separatamente solo la parte del curriculum dal checkpoint.
    
    Questa funzione esiste separatamente da load_checkpoint perché il curriculum
    deve essere caricato DOPO che il modello è stato creato e caricato,
    ma prima di iniziare a giocare.
    
    Argomenti:
        curriculum: l'istanza del CurriculumManager da popolare con i dati salvati
    """
    if not os.path.exists(MODEL_PATH):
        return
    # Nessun file = nessun curriculum da caricare. Parte dal livello 1.
    
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device)
        # Carica l'intero checkpoint in memoria.
        
        if "curriculum" in ckpt:
            curriculum.load_state_dict(ckpt["curriculum"])
            # Se il checkpoint contiene dati curriculum, li carica.
        else:
            print("[CURRICULUM] Nessun dato curriculum nel checkpoint — parto da livello 1.")
            # Checkpoint vecchio senza curriculum: parti dal livello 1 (default).
    except Exception as e:
        print(f"[CURRICULUM] Errore caricamento stato: {e}. Curriculum azzerato.")
        # In caso di errore, il curriculum rimane nei valori di default (livello 1, storia vuota).


def load_oxy_frames():
    """
    Precarica in memoria RAM tutti i 101 frame dell'animazione della barra ossigeno.
    
    L'animazione va da 0 a 100 (corrispondente al % di ossigeno rimanente).
    Precaricando all'avvio, evitiamo di accedere al disco ad ogni frame durante il gioco.
    
    Ritorna:
        dict: mappa {int_0_a_100 → Surface pygame o None se file mancante}
    """
    frames, base_path = {}, path.join("Assets", "Misc", "oxyAnim")
    # Inizializza il dizionario vuoto e costruisce il percorso della cartella delle immagini.
    
    for i in range(101):
    # Itera da 0 a 100 incluso (101 valori).
        
        p = path.join(base_path, f"{i}.png")
        # Costruisce il percorso completo: es. "Assets/Misc/oxyAnim/75.png"
        
        frames[i] = pygame.image.load(p).convert_alpha() if path.exists(p) else None
        # Se il file esiste: carica l'immagine e la converte nel formato interno di Pygame
        # con canale alpha (trasparenza). convert_alpha() ottimizza la blittatura.
        # Se non esiste: salva None (il rendering controllerà questo caso).
    
    return frames


def draw_ai_debug(surface, debug_font, epsilon, steps_done, last_action_idx, ai_state,
                  curriculum=None, in_warmup=False, consec_fails=0):
    """
    Disegna il pannello HUD di debug dell'AI nell'angolo in alto a sinistra dello schermo.
    Visibile solo in PLAY_MODE o con interfaccia attiva.
    
    Argomenti:
        surface:         la Surface Pygame su cui disegnare
        debug_font:      il font da usare per il testo
        epsilon:         valore corrente dell'epsilon (esplorazione)
        steps_done:      numero di step completati
        last_action_idx: indice dell'ultima azione eseguita (0-5)
        ai_state:        stato corrente del ciclo AI ("ACTION", "WAITING", "LEARNING")
        curriculum:      il CurriculumManager (opzionale, mostra progresso curriculum)
        in_warmup:       True se siamo ancora nel periodo di warmup
        consec_fails:    numero di fallimenti consecutivi correnti
    """
    lines_count = 8 if curriculum is not None else 7
    # Il box ha 8 righe se mostriamo il curriculum, 7 altrimenti.
    
    overlay = pygame.Surface((320, 20 + lines_count * 25), pygame.SRCALPHA)
    # Crea una surface trasparente grande abbastanza per contenere tutto il testo.
    # SRCALPHA → abilita il canale alpha per la trasparenza.
    # Dimensioni: 320px di larghezza, altezza dinamica in base al numero di righe.
    
    overlay.fill((0, 0, 0, 180))
    # Riempie con nero semitrasparente (RGBA: rosso=0, verde=0, blu=0, alpha=180/255 ≈ 70%).
    # Questo crea l'effetto di box scuro che si sovrappone al gioco restando leggibile.
    
    surface.blit(overlay, (5, 5))
    # Copia l'overlay sulla surface principale, posizionato a 5px dal bordo in alto a sinistra.
    # blit() = "block image transfer": operazione fondamentale di rendering in Pygame.

    action_names = ["IDLE", "LEFT", "RIGHT", "DRILL_L", "DRILL_R", "DRILL_D"]
    # Nomi leggibili per le 6 azioni (usati per il display, non per la logica).
    
    act_label = action_names[last_action_idx] if last_action_idx < len(action_names) else str(last_action_idx)
    # Converte l'indice numerico in stringa leggibile. Se per qualche bug l'indice è fuori range,
    # mostra il numero direttamente invece di crashare con IndexError.

    warmup_label = f"WARMUP ({steps_done}/{WARMUP_STEPS})" if in_warmup else "TRAINING"
    # Label dinamico che mostra il progresso del warmup se siamo in quella fase.

    infos = [
        (f"EPS:   {epsilon:.4f}", (0, 255, 0)),
        # Epsilon con 4 decimali, colore verde. Verde = ok, stiamo imparando.
        
        (f"STEP:  {steps_done}",   (0, 255, 255)),
        # Numero totale di step, colore ciano.
        
        (f"ACT:   {act_label}",    (255, 255, 0)),
        # Ultima azione eseguita, colore giallo.
        
        (f"MODE:  {'PLAY' if PLAY_MODE else warmup_label}", (100, 200, 255) if not in_warmup else (255, 165, 0)),
        # Modalità attuale (PLAY/TRAINING/WARMUP), colore azzurro in training, arancione in warmup.
        
        (f"PHASE: {ai_state}", (255, 100, 100) if ai_state != "ACTION" else (100, 255, 100)),
        # Fase corrente del ciclo AI: verde se ACTION (attivo), rosso se WAITING o LEARNING.
        
        (f"FAILS: {consec_fails}", (255, 80, 80) if consec_fails >= MAX_CONSEC_FAILS else (180, 180, 180)),
        # Fallimenti consecutivi: rosso brillante se ha raggiunto il limite, grigio altrimenti.
    ]
    
    if curriculum is not None:
        infos.append((
            f"CURR:  Lv{curriculum.current_max_level} "
            f"win={curriculum.get_win_rate():.0%} ep={curriculum.n_episodes()}",
            (255, 200, 80)
        ))
        # Aggiunge la riga del curriculum: livello corrente, win-rate, episodi giocati.
        # Colore arancio dorato per distinguerla dalle altre.

    for i, (txt, col) in enumerate(infos):
        surface.blit(debug_font.render(txt, True, col), (10, 10 + i * 25))
    # Itera su tutte le righe di informazione e le disegna una sotto l'altra.
    # debug_font.render(testo, antialias, colore) → crea una Surface con il testo renderizzato.
    # True = antialias attivato (testo più morbido).
    # Posizione y: 10 + indice*25 → ogni riga è 25px più in basso della precedente.


def draw_gameover(surface, font, won_game=False):
    """
    Disegna la schermata di fine partita sovrapposta al gioco.
    
    Mostra un overlay scuro semi-opaco con:
    - Messaggio principale ("GAME OVER" o "MISSION COMPLETE!")
    - Istruzioni per restartare o uscire
    
    Argomenti:
        surface:  la Surface Pygame principale su cui disegnare
        font:     il font da usare per il testo (grande)
        won_game: True se ha vinto, False se ha perso
    """
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    # Crea un overlay grande quanto l'intera finestra.
    
    overlay.fill((0, 0, 0, 200))
    # Quasi completamente opaco (200/255 ≈ 78%). Oscura bene il gioco sottostante.
    
    surface.blit(overlay, (0, 0))
    # Disegna l'overlay su tutta la finestra a partire dall'angolo in alto a sinistra.

    msg1  = "MISSION COMPLETE!" if won_game else "GAME OVER"
    # Messaggio principale: celebrativo se ha vinto, sobrio se ha perso.
    
    color = (255, 215, 0) if won_game else (255, 50, 50)
    # Oro per la vittoria, rosso per il game over.
    
    s1 = font.render(msg1, True, color)
    # Renderizza il messaggio principale con il font grande e il colore appropriato.
    
    s2 = font.render("Press R to Restart", True, (200, 200, 200))
    # Istruzione secondaria in grigio chiaro.
    
    s3 = font.render("Press ESC to Quit",  True, (150, 150, 150))
    # Istruzione terziaria in grigio più scuro.

    cx, cy = SCREEN_W // 2, SCREEN_H // 2
    # Centro della schermata: SCREEN_W//2 = 400, SCREEN_H//2 = 300.
    # L'operatore // è la divisione intera (arrotonda verso il basso).
    
    surface.blit(s1, s1.get_rect(center=(cx, cy - 50)))
    # Posiziona il messaggio principale 50px sopra il centro.
    # get_rect(center=...) calcola automaticamente la posizione top-left per centrare il testo.
    
    surface.blit(s2, s2.get_rect(center=(cx, cy + 20)))
    # "Press R to Restart" 20px sotto il centro.
    
    surface.blit(s3, s3.get_rect(center=(cx, cy + 60)))
    # "Press ESC to Quit" 60px sotto il centro.


def applyGravity(level, player, surf):
    """
    Il motore fisico del gioco: gestisce la caduta dei blocchi.
    
    Questa è una delle funzioni più complesse del motore di gioco.
    Viene chiamata ogni _T_GRAVITY millisecondi dall'evento EVTIC_GRAVITY.
    
    MECCANICHE IMPLEMENTATE:
        1. Caduta dei blocchi (3 fasi: fermo → tremante → cadente)
        2. Adesione dei blocchi dello stesso colore (gluing)
        3. Collisione con il giocatore (lo schiaccia)
        4. Aggiornamento connessioni visive dopo ogni movimento
    
    Argomenti:
        level:  matrice 2D di oggetti blocco che rappresenta la mappa
        player: l'istanza del Character (giocatore)
        surf:   la Surface Pygame (passata a player.revive() per aggiornamenti visivi)
    
    Ritorna:
        bool: True se almeno un blocco si è mosso o cambiato stato, False se tutto fermo.
    """
    change_happened  = False
    # Flag inizialmente False; diventa True se qualsiasi blocco cambia stato.
    # Usato per ottimizzare: aggiorna le connessioni visive SOLO se necessario.
    
    rows, cols       = len(level), len(level[0])
    # Dimensioni della matrice di gioco.
    # len(level) = numero di righe, len(level[0]) = numero di colonne.
    
    playerY, playerX = player.posAcc()
    # Posizione corrente del giocatore nella griglia (riga, colonna).
    # Necessario per il controllo collisioni blocco-player.

    # =========================================================================
    # SCANSIONE DAL BASSO VERSO L'ALTO (rows - 2 fino a 0)
    # =========================================================================
    # PERCHÉ DAL BASSO?
    # Immagina una colonna con blocchi A (in alto) e B (in mezzo) e vuoto (in basso):
    #   Riga 0: A
    #   Riga 1: B
    #   Riga 2: vuoto
    # Se scansionassimo dall'alto (riga 0 prima):
    #   - A vuole cadere? No, ha B sotto.
    #   - B vuole cadere? Sì, ha vuoto sotto. Spostiamo B alla riga 2.
    #   - Ora A ha vuoto sotto → la stessa scansione lo sposterebbe alla riga 1.
    #   → In un solo tick, A sarebbe sceso di 1 e B di 1. CORRETTO una volta.
    #   Ma se continuassimo la scansione, A potrebbe cadere di nuovo!
    # Scansionando dal basso:
    #   - Riga 1 (B): vuole cadere? Sì. Scende alla riga 2.
    #   - Riga 0 (A): vuole cadere? No, ha la riga 1 che ora è vuota... aspetta!
    #     Ma abbiamo già processato la riga 1 PRIMA di A. A vedrà vuoto.
    # La soluzione dal basso assicura che ogni blocco scenda di UN passo per tick.
    
    for y in range(rows - 2, -1, -1):
    # range(rows-2, -1, -1): parte dalla penultima riga e va verso l'alto.
    # L'ultima riga (rows-1) non può cadere (è il pavimento del gioco).
        
        for x in range(cols):
        # Itera su tutte le colonne da sinistra a destra.
            
            blk      = level[y][x]
            # Il blocco corrente che stiamo analizzando.
            
            blkBelow = level[y + 1][x]
            # Il blocco immediatamente sotto al blocco corrente.

            falling_types = ["classic", "solo", "delayed"]
            # Solo questi tipi di blocchi sono soggetti alla gravità.
            # "unbreakable" (muri marroni/neri) rimane sempre fermo.
            # "pill" (medicina) non cade da solo (è ancorata alla struttura).
            
            if blk.typeAccess() not in falling_types or blk.hpAccess() <= 0:
                continue
            # Se il blocco non è un tipo che cade, o è già distrutto (HP 0 = aria vuota),
            # non c'è fisica da applicare. Saltiamo con continue.

            # =========================================================================
            # CONDIZIONE 1: C'È SPAZIO VUOTO SOTTO IL BLOCCO
            # =========================================================================
            if blkBelow.hpAccess() == 0:
            # Un blocco con HP = 0 è "aria" (cella vuota). Il blocco sopra può cadere.

                is_glued = False
                # Flag: questo blocco è incollato a un vicino? Default: no.

                # --- MECCANICA DI ADESIONE (GLUING) ---
                # In Mr. Driller originale, i blocchi dello stesso colore si "incollano".
                # Un blocco NON cade se è adiacente a un blocco vivo dello stesso colore.
                # Questo crea strutture stabili che richiedono strategia per demolire.
                
                if blk.typeAccess() == "classic":
                # Solo i "classic" hanno la meccanica di incollamento per colore.
                    
                    myColor = blk.ColorAccess()
                    # Recupera il colore di questo blocco (es. "red", "blue", "green").
                    
                    if (x > 0 and level[y][x-1].typeAccess() == "classic" and
                            level[y][x-1].ColorAccess() == myColor and level[y][x-1].hpAccess() > 0):
                        is_glued = True
                    # Controlla blocco a SINISTRA:
                    #   x > 0: non uscire dalla griglia a sinistra
                    #   typeAccess() == "classic": deve essere dello stesso tipo
                    #   ColorAccess() == myColor: deve essere dello stesso colore
                    #   hpAccess() > 0: deve essere ancora vivo (non già distrutto)
                    # Se tutte le condizioni sono vere → incollato a sinistra.
                        
                    if (x < cols-1 and level[y][x+1].typeAccess() == "classic" and
                            level[y][x+1].ColorAccess() == myColor and level[y][x+1].hpAccess() > 0):
                        is_glued = True
                    # Controlla blocco a DESTRA.
                    # x < cols-1: non uscire dalla griglia a destra.
                        
                    if (y > 0 and level[y-1][x].typeAccess() == "classic" and
                            level[y-1][x].ColorAccess() == myColor and level[y-1][x].hpAccess() > 0):
                        is_glued = True
                    # Controlla blocco SOPRA.
                    # y > 0: non uscire dalla griglia verso l'alto.
                    # Nota: il blocco SOTTO non viene controllato perché quello è il vuoto
                    # (abbiamo già verificato blkBelow.hpAccess() == 0).

                if is_glued:
                    blk.stopFalling()
                    # Se incollato, assicura che non sia in stato "falling" o "shaking".
                    # (Potrebbe essere in quello stato da un tick precedente in cui
                    #  il supporto era ancora presente.)
                    continue
                # Salta al prossimo blocco senza applicare la gravità.

                # --- LE TRE FASI DELLA CADUTA ---
                # La caduta non è immediata: passa per 3 stati per rendere il gioco
                # più leggibile e dare al giocatore il tempo di reagire.
                
                if blk.isFalling():
                # FASE C: il blocco è nello stato "sta cadendo" → si sposta di una riga.
                    
                    target_y = y + 1
                    # La destinazione è la riga immediatamente sotto.
                    
                    if target_y == playerY and x == playerX:
                    # Collisione! La destinazione è la cella esatta del giocatore.
                        
                        player.revive(surf, level)
                        # Il giocatore viene schiacciato: perde una vita e respawna.
                        # revive() gestisce la perdita di vita, l'animazione di morte
                        # e il respawn nella posizione iniziale.
                        
                        level[y+1][x], level[y][x] = blk, blkBelow
                        # SCAMBIO di celle: il blocco va alla riga y+1, l'aria viene a riga y.
                        # Python permette questa swap in un'unica istruzione senza variabile temporanea!
                        
                        blk.updatePos(x, y + 1)
                        # Aggiorna le coordinate interne del blocco alla nuova posizione.
                        # Necessario per il rendering e le future verifiche di collisione.
                        
                        blkBelow.updatePos(x, y)
                        # Aggiorna le coordinate dell'aria che ora si trova sopra.
                        
                        change_happened = True
                        # Segnala che qualcosa è cambiato nella mappa.
                    else:
                        # Nessun player sotto: caduta normale.
                        level[y+1][x], level[y][x] = blk, blkBelow
                        blk.updatePos(x, y + 1)
                        blkBelow.updatePos(x, y)
                        change_happened = True

                elif blk.isShaking():
                # FASE B: il blocco sta tremando (ha il vuoto sotto ma non ha ancora "deciso" di cadere).
                    
                    if y == playerY and x == playerX:
                    # Un blocco tremante nella stessa cella del player uccide il player!
                    # Questo è intenzionale: se stai fermo sotto un blocco che trema, muori.
                    # (Nel gioco originale, il tremore è il "warning" per fuggire.)
                        player.revive(surf, level)
                        
                    blk.tickShake()
                    # Avanza il timer interno del tremore.
                    # Dopo un certo numero di tick, lo stato diventa "falling" automaticamente.
                    
                    change_happened = True
                    
                else:
                # FASE A: il blocco ha il vuoto sotto ma era completamente fermo.
                # Inizia la sequenza di avviso con il tremore.
                    blk.startShaking()
                    # Cambia lo stato interno del blocco a "shaking".
                    # Visivamente, il blocco inizierà ad oscillare.
                    change_happened = True
                    
            else:
            # =========================================================================
            # CONDIZIONE 2: C'È UN OSTACOLO SOTTO IL BLOCCO
            # =========================================================================
            # blkBelow.hpAccess() > 0: c'è un blocco solido sotto.
            # Il blocco non può cadere → resetta lo stato di caduta/tremore.
                
                blk.stopFalling()
                # Chiama stopFalling() che resetta sia lo stato "falling" che "shaking".
                # Così un blocco che era sospeso nell'aria (e tremava) si stabilizza
                # non appena arriva un blocco sotto di lui.

    # =========================================================================
    # AGGIORNAMENTO CONNESSIONI VISIVE
    # =========================================================================
    if change_happened:
    # Aggiorna le connessioni visive SOLO se qualcosa è cambiato (ottimizzazione).
    # Senza questo check, aggiorneremmo la grafica ogni tick anche se nulla si è mosso.
        
        for line in level:
            for el in line:
                if hasattr(el, "updCoText"):
                    el.updCoText(level)
        # Itera su tutti i blocchi della mappa.
        # hasattr(el, "updCoText"): verifica se il blocco ha questo metodo
        # (alcuni tipi speciali potrebbero non averlo).
        # updCoText(level): aggiorna quale texture/sprite connettore usare per
        # i "tubi" visivi tra blocchi dello stesso colore adiacenti.
        # Esempio: un blocco rosso vicino a un altro rosso disegna un connettore tra loro.

    return change_happened
    # True se qualcosa si è mosso/cambiato, False se la mappa era già in equilibrio.
    # Questo valore è usato per ottimizzare ulteriori calcoli nel loop principale.


class SmartFrameStacker:
    """
    Accumula e gestisce gli ultimi N frame visivi dell'AI.
    
    MOTIVAZIONE:
        Una singola immagine della griglia non basta per "capire" il movimento.
        Se vedo un blocco alla riga 5 in un frame e alla riga 6 nel frame successivo,
        so che sta cadendo verso il basso.
        Con un singolo frame, non posso distinguere un blocco fermo da uno in caduta.
    
    FUNZIONAMENTO:
        Mantiene una deque degli ultimi stack_size frame.
        get_state() concatena tutti i frame lungo la dimensione dei canali:
        4 frame × 6 canali = 24 canali totali → tensore (24, 11, 9).
    
    TECNICA:
        Questa è la tecnica degli "frame stack" introdotta da DeepMind nel paper
        "Human-level control through deep reinforcement learning" (2015).
        Usata originariamente per far percepire il movimento all'AI che giocava ad Atari.
    """
    
    def __init__(self, stack_size=4, frame_shape=(6, 11, 9), device="cpu"):
        """
        Argomenti:
            stack_size:  numero di frame da mantenere in memoria (default 4)
            frame_shape: shape di un singolo frame (canali, altezza, larghezza)
            device:      dispositivo PyTorch dove tenere i tensori ("cpu", "cuda", "mps")
        """
        self.stack_size = stack_size
        # Numero di frame da impilare.
        
        self.device     = device
        # Dispositivo di calcolo: i tensori devono vivere tutti sullo stesso device.
        
        self.frames = deque(maxlen=stack_size)
        # Deque con capacità massima = stack_size.
        # Quando si aggiunge un nuovo frame con append(), il più vecchio viene rimosso automaticamente.
        
        self._reset_frames(frame_shape)
        # Inizializza con frame vuoti (tutti zero).

    def _reset_frames(self, shape):
        """
        Riempie la deque con frame vuoti (tensori di zeri).
        Chiamato all'inizializzazione e quando il reset porta forma diversa.
        
        Argomenti:
            shape: shape del frame da creare (es. (6, 11, 9))
        """
        self.frames.clear()
        # Svuota completamente la deque.
        
        for _ in range(self.stack_size):
            self.frames.append(torch.zeros(shape, device=self.device))
        # Aggiunge stack_size frame vuoti (tutti 0.0).
        # Questo rappresenta il "passato" vuoto prima che il gioco inizi.
        # La rete ha comunque dati validi (zeri) anche al primo step.

    def reset(self, initial_frame):
        """
        Resetta lo stack riempiendo tutti gli slot con il primo frame di un nuovo episodio.
        
        Usato all'inizio di ogni nuova partita per evitare che lo stack
        contenga frame del livello precedente.
        
        Argomenti:
            initial_frame: tensore (6, 11, 9) rappresentante lo stato iniziale del livello.
        """
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(initial_frame.clone())
        # Riempie tutti gli slot con una copia del frame iniziale.
        # .clone() crea una copia indipendente del tensore per evitare aliasing.
        # (Se tutti gli slot puntassero allo stesso tensore, modificarne uno modificherebbe tutti.)

    def push(self, frame):
        """
        Aggiunge un nuovo frame allo stack, rimuovendo automaticamente il più vecchio.
        
        Argomenti:
            frame: tensore (6, 11, 9) del frame corrente.
        """
        self.frames.append(frame)
        # Grazie a maxlen=stack_size, la deque rimuove automaticamente il frame più vecchio.

    def get_state(self):
        """
        Restituisce lo stato corrente come tensore unico con tutti i frame concatenati.
        
        Ritorna:
            tensore di shape (N_CHANNELS * stack_size, 11, 9) = (24, 11, 9)
        """
        return torch.cat(list(self.frames), dim=0)
        # torch.cat concatena una lista di tensori lungo una dimensione specificata.
        # dim=0 = concatena lungo i canali (prima dimensione).
        # Es: 4 tensori (6, 11, 9) → 1 tensore (24, 11, 9)
        # list(self.frames) converte la deque in lista (necessario per torch.cat).


def _init_ai_components():
    """
    Istanzia e restituisce i tre componenti ausiliari dell'AI.
    Funzione di utilità per rendere game() più leggibile.
    
    Ritorna:
        (stacker, shaper, drill_tracker): tuple con i tre oggetti inizializzati
    """
    stacker       = SmartFrameStacker(N_FRAMES, (N_CHANNELS, 11, 9), device)
    # Frame stacker: mantiene gli ultimi 4 frame visivi per il senso del movimento.
    
    shaper        = RewardShaper()
    # Reward shaper: calcola il bonus PBRS basato sulla profondità raggiunta.
    
    drill_tracker = DrillTracker()
    # Drill tracker: monitora azioni ripetitive inutili per penalizzare i loop.
    
    return stacker, shaper, drill_tracker


def _reset_ai_episode(player, level, stacker, nstep_buf, shaper, drill_tracker, memory=None):
    """
    Resetta tutti i componenti AI all'inizio di un nuovo episodio (nuova partita).
    
    Questo è cruciale per evitare che informazioni del livello precedente
    inquinino l'apprendimento del livello nuovo.
    
    Argomenti:
        player:       il Character del giocatore (per ottenere posizione e ossigeno iniziali)
        level:        la nuova mappa di gioco
        stacker:      il SmartFrameStacker da resettare
        nstep_buf:    il NStepBuffer da svuotare (o None in PLAY_MODE)
        shaper:       il RewardShaper da resettare con la posizione iniziale
        drill_tracker: il DrillTracker da resettare
        memory:       il ReplayMemory (opzionale: se fornito, riceve le transizioni rimaste nel buffer)
    """
    pY, pX     = player.posAcc()
    # Ottiene la posizione iniziale del giocatore nel nuovo livello.
    
    init_frame = get_local_window_tensor(level, pX, pY).to(device)
    # Genera la finestra visiva iniziale centrata sulla posizione del giocatore.
    # .to(device) sposta il tensore sul dispositivo corretto (CPU/GPU).
    
    stacker.reset(init_frame)
    # Riempie lo stack con il frame iniziale (tutti i 4 slot con lo stesso frame).
    
    if nstep_buf is not None:
        if memory is not None:
            for trans in nstep_buf.drain():
                memory.push(*trans)
            # drain() estrae le transizioni ancora nel buffer N-step (incomplete).
            # Le aggiunge alla memoria principale prima di resettare,
            # così non si perdono esperienze dell'episodio appena concluso.
            # *trans scompatta la tupla: memory.push(s, v, a, s', v', r, done)
        else:
            nstep_buf.flush()
            # In PLAY_MODE non c'è memoria, quindi svuota senza salvare.
    
    shaper.reset(pY, player.oxyAcc(), len(level))
    # Resetta il RewardShaper con:
    #   pY: profondità iniziale del giocatore
    #   player.oxyAcc(): ossigeno iniziale (tipicamente 100%)
    #   len(level): numero totale di righe del livello (per normalizzare la profondità)
    
    drill_tracker.reset()
    # Resetta i contatori del DrillTracker: fallimenti consecutivi, posizioni visitate, ecc.


def _reset_game_state(player, level, levelID):
    """
    Resetta le variabili di stato del gioco tra episodi.
    
    Separato da _reset_ai_episode perché alcune variabili riguardano
    il motore di gioco (telecamera, blocchi in scomparsa) non l'AI.
    
    Argomenti:
        player:  il Character (non usato direttamente qui, ma passato per coerenza)
        level:   la mappa corrente
        levelID: ID del livello corrente
    
    Ritorna:
        tupla (currentBotLine, currentOffset, blocksDisap, ai_state,
               gravity_applied_since_action, backDown, currentClimb)
    """
    clear_cache()
    # Svuota la cache delle texture dei blocchi.
    # I blocchi usano una cache per non ricaricare le immagini ogni frame.
    # Al cambio livello, le texture potrebbero essere diverse → svuota la cache.
    
    return 8, 0, set(), "ACTION", False, False, 0
    # currentBotLine = 8: la riga più in basso visibile inizialmente
    # currentOffset = 0: nessuno scroll della telecamera
    # blocksDisap = set(): nessun blocco "delayed" in attesa di scomparire
    # ai_state = "ACTION": l'AI parte pronta ad agire
    # gravity_applied_since_action = False: la gravità non è ancora stata applicata
    # backDown = False: nessuna risalita in corso
    # currentClimb = 0: il player non si è arrampicato su nessun blocco


# =============================================================================
# FUNZIONE PRINCIPALE DEL GIOCO
# =============================================================================

def game(screen_width=SCREEN_W, screen_height=SCREEN_H):
    """
    Il loop principale del gioco e del training. Questa funzione contiene tutto.
    
    STRUTTURA DEL LOOP:
        1. Inizializzazione (reti neurali, Pygame, assets grafici)
        2. Loop while inProgress:
            a. Gestione menu (se inMenu)
            b. Gestione Game Over screen (se inGameOver)
            c. Check epsilon adattivo
            d. FASE ACTION: l'AI guarda e sceglie un'azione
            e. Pump eventi Headless (se HEADLESS_TRAINING)
            f. Elaborazione eventi Pygame (input, timer, fisica)
            g. FASE WAITING: attesa completamento fisica
            h. FASE LEARNING: calcolo reward e backpropagation
            i. Fine episodio: gestione vittoria/sconfitta
            j. Rendering grafico (solo se non headless)
        3. Cleanup e chiusura
    
    Argomenti:
        screen_width:  larghezza della finestra (default 800)
        screen_height: altezza della finestra (default 600)
    """
    
    TRAIN_AI   = AI_AVAILABLE and not PLAY_MODE
    # True solo se: l'AI è disponibile E siamo in modalità training.
    # Un'AI può essere disponibile (importata) ma usata solo per giocare (PLAY_MODE=True).
    
    steps_done = 0
    # Contatore globale di tutti gli step/azioni eseguiti dall'AI dall'inizio del training.
    # Persiste tra gli episodi: non si resetta mai (salvo ripartire da zero).
    
    n_actions  = 6
    # Numero di azioni disponibili all'agente:
    # 0: IDLE (non fare niente, aspetta), 1: LEFT (vai a sinistra),
    # 2: RIGHT (vai a destra), 3: DRILL_L (trivella sinistra),
    # 4: DRILL_R (trivella destra), 5: DRILL_D (trivella giù)
    
    epsilon    = EPS_START
    # Probabilità iniziale di azione casuale (esplorazione epsilon-greedy).

    # Inizializzazione variabili a None: verranno create solo se serve.
    policy_net = target_net = optimizer = scheduler = None
    # Reti neurali e ottimizzatore: None in PLAY_MODE senza AI o senza training.
    
    memory = stacker = nstep_buf = shaper = drill_tracker = monitor = None
    # Buffer di memoria, componenti AI ausiliari, monitor: None finché non inizializzati.

    curriculum = CurriculumManager(min_episodes=30)
    # Crea il curriculum manager. Anche in PLAY_MODE esiste, ma non viene aggiornato.

    # =========================================================================
    # INIZIALIZZAZIONE TRAINING
    # =========================================================================
    if TRAIN_AI:
        print("[AI] Inizializzazione reti neurali — TRAINING MODE...")

        policy_net = DrillerDuelingDQN(INPUT_SHAPE, n_actions, state_dim=STATE_DIM).to(device)
        # Crea la Policy Net: la rete che viene addestrata e che "gioca" attivamente.
        # INPUT_SHAPE = (24, 11, 9), n_actions = 6, state_dim = 32.
        # .to(device) sposta tutti i parametri della rete su CPU/GPU/MPS.
        
        target_net = DrillerDuelingDQN(INPUT_SHAPE, n_actions, state_dim=STATE_DIM).to(device)
        # Crea la Target Net: copia stabile della Policy Net.
        # Usata nel calcolo della Bellman equation per calcolare Q-target stabili.
        # Non viene aggiornata con backprop: viene COPIATA dalla Policy Net ogni TARGET_UPDATE step.
        # Senza di essa, il training sarebbe instabile (perseguiresti un target che si muove con te).

        optimizer  = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE,
                                 weight_decay=1e-5, amsgrad=True)
        # AdamW: Adam con Weight Decay disaccoppiato.
        # parameters(): restituisce tutti i parametri (pesi + bias) della Policy Net.
        # lr=2e-5: learning rate molto piccolo per aggiornamenti conservativi.
        # weight_decay=1e-5: penalità L2 sui pesi (regolarizzazione, riduce overfitting).
        # amsgrad=True: variante di Adam che usa il massimo dei gradients passati
        #               per garantire convergenza monotona. Più stabile in alcuni casi.
        
        scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500_000, eta_min=1e-6)
        # Scheduler che varia il learning rate seguendo una curva coseno.
        # T_max=500_000: il ciclo completo dura 500.000 step.
        # eta_min=1e-6: il learning rate non scende mai sotto 0.000001.
        # Perché il coseno? Inizia alto (2e-5), scende gradualmente, poi risale leggermente.
        # Questo aiuta a "uscire" da minimi locali e trovare soluzioni migliori.

        steps_done, epsilon = load_checkpoint(policy_net, optimizer, scheduler)
        # Carica il checkpoint se esiste. Ripristina step, epsilon, pesi e ottimizzatore.
        
        epsilon = max(epsilon, EPS_FLOOR)
        # Assicura che epsilon non sia mai sotto il floor, anche se il checkpoint
        # aveva salvato un valore più basso.
        
        _load_curriculum_from_checkpoint(curriculum)
        # Carica separatamente lo stato del curriculum.

        if curriculum.training_complete:
            print("[CURRICULUM] Il training risulta già completato dal checkpoint. Uscita.")
            return
        # Se il curriculum è già completato, non c'è nulla da fare.

        target_net.load_state_dict(policy_net.state_dict())
        # Copia i pesi della Policy Net nella Target Net per allinearle all'inizio.
        # Necessario anche quando si carica da checkpoint (entrambe devono partire uguali).
        
        target_net.eval()
        # Imposta la Target Net in modalità "valutazione".
        # Disabilita: Dropout, Batch Normalization in training mode.
        # La Target Net non viene mai addestrata con backprop → non ha senso usarla in training mode.

        memory = ReplayMemory(MEMORY_SIZE)
        # Crea il Replay Buffer con priorità, capienza MEMORY_SIZE (50.000 transizioni).
        
        memory.load(BUFFER_DIR, max_load=BUFFER_MAX_LOAD)
        # Carica le transizioni salvate precedentemente dal disco.
        # max_load=20.000: carica solo le più recenti per non esaurire la RAM.

        stacker, shaper, drill_tracker = _init_ai_components()
        # Crea i tre componenti ausiliari dell'AI.
        
        nstep_buf = NStepBuffer(n_step=N_STEP, gamma=GAMMA)
        # Crea il buffer N-step con N=8 step e gamma=0.99.

        print(f"[AI] Reti pronte — Device: {device}")
        print(f"[AI] v6.9: LR={LEARNING_RATE}, EPS_FLOOR={EPS_FLOOR}, WARMUP={WARMUP_STEPS}")
        print(f"[CURRICULUM] Livello massimo corrente: {curriculum.current_max_level}")
        # Log di avvio con tutti i parametri e le fix applicate in questa versione.

        if len(memory) < WARMUP_STEPS:
            print(f"[WARMUP] Buffer attuale ({len(memory)}) < {WARMUP_STEPS} — "
                  f"optimize_model disabilitato fino al raggiungimento della soglia.")
        else:
            print(f"[WARMUP] Buffer ({len(memory)}) ≥ {WARMUP_STEPS} — training attivo da subito.")
        # Informa se siamo in warmup o no basandosi sulla dimensione attuale del buffer.

        try:
            from training_monitor import TrainingMonitor
            # Importa il monitor di training (file opzionale nel progetto).
            
            monitor = TrainingMonitor(log_dir="runs/driller_v6")
            # Crea il monitor che salva log CSV e genera grafici.
            # log_dir: cartella dove salvare i file di log.
            
            print("[Monitor] Monitor CSV attivo.")
        except ImportError:
            print("[Monitor] training_monitor.py non trovato.")
        except Exception as e:
            print(f"[Monitor] Errore init: {e}. Disabilitato.")
        # Se il monitor non è disponibile, il training continua senza log.

    # =========================================================================
    # INIZIALIZZAZIONE PYGAME
    # =========================================================================
    
    pygame.init()
    # Inizializza TUTTI i moduli di Pygame (display, input, audio, timer, ecc.).
    # Deve essere chiamato prima di qualsiasi altra funzione Pygame.
    
    try:
        pygame.mixer.init()
        # Inizializza il sistema audio di Pygame.
        # Separato in try/except perché potrebbe fallire su sistemi senza scheda audio.
    except Exception:
        pass
    # Se l'audio fallisce, il gioco continua silenziosamente senza crash.

    FPS      = 60
    # Target frame per secondo. Il clock limiterà il loop a non superare 60 fps.
    
    fpsClock = pygame.time.Clock()
    # Il Clock di Pygame: usato con fpsClock.tick(FPS) per limitare la velocità del loop.
    # Inserisce automaticamente il delay necessario per rispettare il target FPS.

    surface  = pygame.display.set_mode((screen_width, screen_height))
    # Crea la finestra del gioco con le dimensioni specificate (800x600).
    # In modalità headless, questa chiamata va al driver "dummy" e non mostra nulla.
    
    pygame.display.set_caption("Mr. Driller AI — Rainbow DQN")
    # Imposta il titolo della finestra (mostrato nella barra del titolo).

    # =========================================================================
    # CARICAMENTO FONT
    # =========================================================================
    
    font_path = path.join("Assets", "Misc", "police", "Act_Of_Rejection.ttf")
    # Percorso del font personalizzato del gioco.
    
    try:
        ui_font    = pygame.font.Font(font_path, 34)
        # Font per l'interfaccia utente (punteggio, ossigeno), dimensione 34pt.
        
        debug_font = pygame.font.Font(font_path, 20)
        # Font per il pannello di debug AI, dimensione 20pt.
        
        big_font   = pygame.font.Font(font_path, 48)
        # Font per i numeri grandi (punteggio, game over), dimensione 48pt.
        
    except FileNotFoundError:
        # Se il font personalizzato non esiste, usa Arial di sistema.
        ui_font    = pygame.font.SysFont("Arial", 32, bold=True)
        debug_font = pygame.font.SysFont("Arial", 18, bold=True)
        big_font   = pygame.font.SysFont("Arial", 48, bold=True)

    # =========================================================================
    # CARICAMENTO ASSETS GRAFICI
    # =========================================================================
    
    ui_bg_path = path.join("Assets", "Misc", "userinterface.png")
    # Percorso dell'immagine di sfondo dell'interfaccia (pannello laterale).
    
    try:
        raw_bg = pygame.image.load(ui_bg_path).convert()
        # Carica l'immagine e la converte nel formato pixel interno di Pygame.
        # convert() (senza alpha) è più veloce di convert_alpha() per le blittature.
        
        Ui_bg  = pygame.transform.scale(raw_bg, (screen_width, screen_height))
        # Ridimensiona l'immagine alla dimensione della finestra (800x600).
    except FileNotFoundError:
        Ui_bg = pygame.Surface((screen_width, screen_height))
        Ui_bg.fill((30, 30, 30))
        # Se l'immagine non esiste, usa uno sfondo grigio scuro come fallback.

    life_icon_path = path.join("Assets", "Misc", "icon.png")
    # Percorso dell'icona delle vite (la faccina di Mr. Driller).
    
    icon = pygame.image.load(life_icon_path).convert_alpha() if path.exists(life_icon_path) else None
    # Carica l'icona se esiste, altrimenti None.
    
    if icon and icon.get_width() > 40:
        icon = pygame.transform.scale(icon, (32, 32))
    # Se l'icona è troppo grande (>40px), la ridimensiona a 32x32.
    # get_width() restituisce la larghezza in pixel dell'immagine caricata.

    oxy_frames = load_oxy_frames()
    # Precarica tutti i 101 frame dell'animazione dell'ossigeno.

    def play_music(level_id):
        """
        Riproduce la traccia musicale appropriata per il livello.
        
        Argomenti:
            level_id: 0 = menu, 1-10 = livelli del gioco
        """
        if not pygame.mixer: return
        # Se il mixer non è stato inizializzato (fallito all'init), non fare niente.
        
        try:
            pygame.mixer.music.stop()
            # Ferma la traccia corrente prima di caricare quella nuova.
            
            fname = f"Level{level_id}.wav" if level_id > 0 else "menu.wav"
            # Determina il nome del file: "Level1.wav", "Level2.wav", ..., o "menu.wav".
            
            fpath = path.join("Assets", "Music", fname)
            # Costruisce il percorso completo del file audio.
            
            if path.exists(fpath):
                pygame.mixer.music.load(fpath)
                # Carica il file audio nel mixer. Solo un file alla volta può essere caricato.
                
                pygame.mixer.music.set_volume(0.4)
                # Imposta il volume al 40% (0.0 = muto, 1.0 = massimo).
                
                pygame.mixer.music.play(-1)
                # Avvia la riproduzione. -1 = loop infinito.
        except Exception as e:
            print(f"[Audio] Errore: {e}")

    pygame.event.pump()
    # Svuota la coda degli eventi sistema prima di iniziare.
    # Serve per processare gli eventi OS in sospeso (es. il titolo della finestra).

    # =========================================================================
    # CREAZIONE OGGETTI PRINCIPALI DI GIOCO
    # =========================================================================
    
    player = Character(3, 4, 1, 2)
    # Crea il personaggio giocante.
    
    level, levelID, won = restart(player)
    # Genera la mappa del livello 1 e resetta il player alla posizione iniziale.
    # level: matrice 2D di oggetti Block
    # levelID: ID numerico del livello corrente (parte da 1)
    # won: True se il livello è stato completato (qui sempre False all'inizio)
    
    play_music(0)
    # Avvia la musica del menu (level_id=0 → "menu.wav").

    # =========================================================================
    # CARICAMENTO MODELLO PER PLAY MODE
    # =========================================================================
    
    if PLAY_MODE and AI_AVAILABLE:
    # In PLAY_MODE, l'AI gioca visibilmente usando i pesi salvati (senza training).
        
        print("[AI] Modalità PLAY — caricamento modello...")
        policy_net = DrillerDuelingDQN(INPUT_SHAPE, n_actions, state_dim=STATE_DIM).to(device)
        # Crea la rete anche in PLAY_MODE per poter fare l'inferenza (forward pass).
        
        policy_net.eval()
        # Modalità valutazione: disabilita dropout e batchnorm in training mode.
        # In PLAY_MODE non c'è training, quindi non ha senso essere in training mode.
        
        if os.path.exists(MODEL_PATH):
            ckpt = torch.load(MODEL_PATH, map_location=device)
            # Carica il checkpoint.
            
            policy_net.load_state_dict(ckpt["model_state_dict"], strict=True)
            # Carica i pesi salvati nella rete.
            
            steps_done = ckpt.get("steps", 0)
            # Recupera quanti step aveva fatto il modello durante il training.
            
            epsilon    = 0.0
            # In PLAY_MODE, ZERO esplorazione casuale: l'AI usa sempre la rete.
            # Vogliamo vedere la policy appresa, non azioni casuali.
            
            print(f"[AI] Modello caricato — step {steps_done}")
        else:
            print("[AI] ERRORE: nessun modello trovato!")
            # Il modello non esiste ancora → l'AI non può giocare.
        
        stacker, shaper, drill_tracker = _init_ai_components()
        # Crea i componenti ausiliari anche in PLAY_MODE (servono per generare lo stato).
        
        _reset_ai_episode(player, level, stacker, None, shaper, drill_tracker)
        # Resetta i componenti per il primo episodio.
        # None per nstep_buf perché in PLAY_MODE non c'è N-step buffer.

    if TRAIN_AI:
        _reset_ai_episode(player, level, stacker, nstep_buf, shaper, drill_tracker, memory=None)
        # Resetta i componenti per il primo episodio di training.
        # memory=None perché all'inizio non vogliamo svuotare il buffer
        # (potrebbe esserci già della memoria caricata da disco).

    # =========================================================================
    # EVENTI PERSONALIZZATI PYGAME
    # =========================================================================
    # Pygame permette di creare eventi custom oltre a quelli standard (click, tastiera, ecc.).
    # Li usiamo come "timer" per sincronizzare la fisica, le animazioni e il consumo di ossigeno.
    
    EVT_END_BLOCK = pygame.USEREVENT
    # Evento lanciato quando il giocatore rompe il blocco finale del livello.
    # pygame.USEREVENT è il primo ID disponibile per eventi custom (tipicamente 24).
    
    EVTICSEC      = pygame.USEREVENT + 1
    # Evento "secondo di gioco": ogni ~1050ms, toglie ossigeno al player.
    
    EVTICPLY      = pygame.USEREVENT + 2
    # Evento "tick player": ogni ~120ms, aggiorna lo stato idle del player.
    
    EVTIC_GRAVITY = pygame.USEREVENT + 4
    # Evento "tick gravità": ogni ~400ms, avanza di una fase la fisica dei blocchi.
    
    EVTIC_ANIM    = pygame.USEREVENT + 5
    # Evento "tick animazione": ogni ~80ms, cambia il frame dell'animazione sprite.

    if not HEADLESS_TRAINING:
    # In modalità con interfaccia visiva, imposta i timer REALI di Pygame.
        pygame.time.set_timer(EVTIC_GRAVITY, int(_T_GRAVITY))
        # Programma il ripetersi di EVTIC_GRAVITY ogni _T_GRAVITY millisecondi.
        
        pygame.time.set_timer(EVTIC_ANIM,    int(_T_ANIM))
        # Timer animazione.
        
        pygame.time.set_timer(EVTICSEC,      int(_T_SEC))
        # Timer ossigeno (1 secondo di gioco).
        
        pygame.time.set_timer(EVTICPLY,      int(_T_PLY))
        # Timer player idle.

    # =========================================================================
    # VARIABILI DI TEMPORIZZAZIONE HEADLESS
    # =========================================================================
    # In modalità headless non usiamo i timer di Pygame (troppo lenti).
    # Invece "pumpiam" manualmente gli eventi ad ogni iterazione del loop.
    
    v_timer_gravity = 0.0
    v_timer_anim    = 0.0
    v_timer_sec     = 0.0
    v_timer_ply     = 0.0
    # Accumulatori di tempo virtuale (non usati attivamente nel codice attuale,
    # ma mantenuti per potenziale uso futuro o debug).
    
    _hl_tick        = 0
    # Contatore di tick headless: usato per determinare quando inviare eventi meno frequenti.
    
    _DT_HEADLESS    = 1000.0 / FPS
    # Delta time per un frame headless: 1000ms / 60fps ≈ 16.67ms per frame.
    # Non usato attivamente qui, ma utile per calcoli di timing.

    # =========================================================================
    # VARIABILI DI STATO DEL GIOCO
    # =========================================================================
    
    currentBotLine = 8
    # La riga più in basso della griglia visibile sullo schermo.
    # Inizia a 8 (le prime 8 righe sono visibili dall'inizio).
    # Sale man mano che il player scende e la "telecamera" scorre.
    
    currentOffset  = 0
    # Offset globale dello scroll della telecamera.
    # Quanti blocchi di offset ha la griglia rispetto alla posizione iniziale.
    # Aumenta ogni volta che il player scende abbastanza.
    
    currentClimb   = 0
    # Tiene traccia di quanto il player è risalito sopra l'offset attuale.
    # Serve per calcolare la "profondità reale" = currentOffset - currentClimb.
    
    backDown       = False
    # Flag: True se il player aveva risalito ma ora sta tornando giù.
    # Gestisce il meccanismo di "cleanup" della risalita visiva.

    # =========================================================================
    # VARIABILI DI FLUSSO DELL'APPLICAZIONE
    # =========================================================================
    
    inProgress = True
    # Loop principale attivo. Diventa False per chiudere il programma.
    
    inMenu     = not (TRAIN_AI or PLAY_MODE)
    # True solo se NON siamo né in training né in PLAY_MODE (modalità puramente interattiva).
    # In training o PLAY_MODE, salta il menu e inizia subito il gioco.
    
    inGame     = not inMenu
    # True se il gioco è attivo (sia training che play).
    
    inGameOver = False
    # True quando si mostra la schermata di Game Over.
    
    game_won   = False
    # True se il Game Over è dovuto alla vittoria (completamento del gioco).
    
    menuOption = 1
    # Opzione selezionata nel menu principale (1=Start, 2=Quit).

    movKeys = [K_w, K_d, K_a, K_s]
    # Lista dei tasti di movimento: W(su), D(destra), A(sinistra), S(giù).
    # Passata a movementHandle() per sapere quali tasti gestire.
    # In modalità AI, questi tasti vengono "simulati" programmaticamente.

    # =========================================================================
    # VARIABILI DI STATO AI
    # =========================================================================
    
    last_action_idx = 0
    # Indice dell'ultima azione eseguita (0-5). Inizia con IDLE(0).
    # Usato per: reward shaping, log, HUD debug, calcolo stato interno.
    
    ai_state        = "ACTION"
    # Macchina a stati dell'AI: "ACTION" → "WAITING" → "LEARNING" → "ACTION" ...
    # ACTION: l'AI sceglie e esegue un'azione
    # WAITING: attende che la fisica del gioco risponda all'azione
    # LEARNING: calcola la reward e aggiorna la rete

    # Variabili "pt_" (pre-transizione): salvano lo stato PRIMA dell'azione.
    # Necessario per calcolare la reward: confronto tra "prima" e "dopo".
    pt_grid_old = pt_vars_old = pt_action = None
    # pt_grid_old: tensore visivo prima dell'azione
    # pt_vars_old: vettore variabili interne prima dell'azione
    # pt_action: l'azione scelta (tensore PyTorch)
    
    pt_prev_score = pt_prev_oxy = pt_prev_lives = pt_prev_y = pt_prev_x = 0
    # Punteggio, ossigeno, vite, posizione y e x prima dell'azione.
    
    pt_was_hard_block = pt_was_delayed_block = False
    # Flag per il tipo di blocco che l'AI stava per trivellare:
    # hard_block = "unbreakable" (non trivellabile), delayed_block = blocco a timeout.
    
    pt_blocks_around  = None
    # Dizionario con informazioni sui blocchi adiacenti al player (sopra, sotto, sx, dx).
    
    q_values_np       = None
    # Array NumPy con i Q-values calcolati dalla rete per l'azione corrente.
    # Usato per il log del monitor di training.

    pt_post_action_y = 0
    pt_post_action_x = 0
    # Posizione del player SUBITO dopo l'azione, PRIMA che la gravità agisca.
    # Serve per distinguere: "il player si è mosso da solo" vs "è stato spostato da un blocco che cade".

    gravity_applied_since_action = False
    # True se almeno un tick di gravità è avvenuto dall'ultima azione dell'AI.
    # Usato nella logica di transizione WAITING → LEARNING.
    
    blocksDisap: set  = set()
    # Set di coordinate (riga, colonna) di blocchi "delayed" in attesa di scomparire.
    # Viene aggiornato ogni EVTICSEC.
    
    _train_counter    = 0
    # Contatore locale per TRAIN_EVERY: conta le azioni dall'ultimo optimize_model().

    _last_death_cause = "block_fall"
    # Causa della morte dell'episodio corrente:
    # "block_fall": schiacciato da un blocco
    # "oxy": mancanza di ossigeno
    # "win": vittoria (non una morte)
    # Aggiornato quando player.livesAcc() diminuisce.

    _last_eps_check_step = 0
    # Ultimo step in cui è stato fatto il check dell'epsilon adattivo.
    
    _eps_boosted         = False
    # True se epsilon è stato alzato artificialmente dal meccanismo di boost.
    # Usato per rilevare quando il win-rate torna sopra la soglia e rimuovere il boost.

    try:
        from eventHandling import movementHandle, breaking
        # movementHandle: gestisce il movimento del player (sinistra, destra)
        # breaking: gestisce il trivellamento (rompe i blocchi adiacenti)
        
        from level import render
        # render: funzione che disegna tutti i blocchi della mappa sulla surface.
        
    except ImportError:
        def movementHandle(*args): pass
        def breaking(*args): pass
        def render(*args): pass
    # Se i moduli non esistono (es. test unitari), usa stub vuoti che non fanno niente.

    print("[GAME] Entro nel loop principale...")

    def _emergency_save():
        """
        Funzione di salvataggio d'emergenza chiamata in caso di crash o CTRL+C.
        Salva i pesi della rete e il buffer di memoria per non perdere il training.
        """
        if TRAIN_AI and policy_net is not None and optimizer is not None:
            print("[CRASH-SAVE] Salvataggio emergenza...")
            save_checkpoint(policy_net, optimizer, steps_done, epsilon, scheduler,
                            curriculum=curriculum)
            if memory is not None:
                memory.save_async(BUFFER_DIR)
            print("[CRASH-SAVE] Fatto.")

    try:
        # =====================================================================
        # LOOP PRINCIPALE
        # Il cuore di tutto il programma. Gira finché inProgress è True.
        # =====================================================================
        while inProgress:

            # -----------------------------------------------------------------
            # GESTIONE MENU INIZIALE
            # Attivo solo quando inMenu=True (modalità umana interattiva).
            # -----------------------------------------------------------------
            if inMenu:
                mainMenu(surface, menuOption)
                # Disegna il menu principale sull'area di gioco.
                
                for event in pygame.event.get():
                    # Elabora tutti gli eventi in coda (tasti premuti, chiusura finestra).
                    
                    if event.type == QUIT:
                        inProgress = False
                        # L'utente ha chiuso la finestra: esci dal loop.
                    
                    if event.type == KEYDOWN:
                        if event.key == K_UP:
                            menuOption = 1
                            # Freccia su → seleziona "Start Game"
                        elif event.key == K_DOWN:
                            menuOption = 2
                            # Freccia giù → seleziona "Quit"
                        elif event.key == K_RETURN:
                            if menuOption == 1:
                                inMenu = False; inGame = True
                                # Avvia il gioco: disattiva il menu, attiva il gioco.
                                level, levelID, won = restart(player)
                                # Genera il livello 1.
                                play_music(levelID)
                                # Avvia la musica del livello 1.
                            elif menuOption == 2:
                                inProgress = False
                                # Esci dal programma.
                
                pygame.display.update()
                # Aggiorna il display con quanto disegnato (applica le blit al monitor).
                
                continue
                # Salta il resto del loop e torna all'inizio (non processare gioco o AI).

            # -----------------------------------------------------------------
            # GESTIONE GAME OVER SCREEN
            # Attivo quando inGameOver=True (l'episodio è finito e non si sta addestrando).
            # -----------------------------------------------------------------
            if inGameOver:
                
                if TRAIN_AI:
                    # In modalità training, non mostrare la schermata di Game Over.
                    # Resetta immediatamente e ricomincia la prossima partita.
                    inGameOver = False; inGame = True
                    player = Character(3, 4, 1, 2)
                    # Ricrea il player con posizione e vite iniziali.
                    
                    level, levelID, won = restart(player)
                    # Genera una nuova mappa dal livello 1.
                    
                    (currentBotLine, currentOffset, blocksDisap, ai_state,
                     gravity_applied_since_action, backDown, currentClimb) = \
                        _reset_game_state(player, level, levelID)
                    # Resetta le variabili di stato del gioco.
                    
                    _reset_ai_episode(player, level, stacker, nstep_buf, shaper, drill_tracker, memory=memory)
                    # Resetta i componenti AI per il nuovo episodio.
                    
                    _last_death_cause = "block_fall"
                    # Resetta la causa di morte per il nuovo episodio.
                    
                    continue
                    # Salta al prossimo ciclo del loop (non mostrare la schermata).

                if not HEADLESS_TRAINING:
                    draw_gameover(surface, big_font, game_won)
                    # Disegna la schermata di Game Over (solo se la finestra è visibile).
                    pygame.display.update()

                for event in pygame.event.get():
                    if event.type == QUIT:
                        inProgress = False
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            inProgress = False
                            # ESC → esci dal programma.
                        elif event.key == K_r:
                            # R → riavvia una nuova partita.
                            inGameOver = False; inGame = True
                            player = Character(3, 4, 1, 2)
                            level, levelID, won = restart(player)
                            (currentBotLine, currentOffset, blocksDisap, ai_state,
                             gravity_applied_since_action, backDown, currentClimb) = \
                                _reset_game_state(player, level, levelID)
                            play_music(levelID)
                            _last_death_cause = "block_fall"
                            if PLAY_MODE and AI_AVAILABLE:
                                _reset_ai_episode(player, level, stacker, None, shaper, drill_tracker)
                
                if not HEADLESS_TRAINING:
                    fpsClock.tick(FPS)
                    # In PLAY_MODE con Game Over, limita comunque a 60fps.
                    
                continue

            # ═════════════════════════════════════════════════════════════════
            # CHECK EPSILON ADATTIVO
            # Ogni EPS_CHECK_INTERVAL step, valuta le performance e aggiusta epsilon.
            # ═════════════════════════════════════════════════════════════════
            if (TRAIN_AI and
                    steps_done > 0 and
                    steps_done - _last_eps_check_step >= EPS_CHECK_INTERVAL and
                    curriculum.n_episodes() >= EPS_BOOST_MIN_EP):
            # Condizioni:
            # 1. Siamo in training
            # 2. Abbiamo già fatto almeno 1 step
            # 3. Sono passati abbastanza step dall'ultimo check (2000)
            # 4. Abbiamo giocato abbastanza episodi per una statistica affidabile (20)

                short_wr = curriculum.get_short_win_rate(n=20)
                # Calcola il win-rate delle ultime 20 partite (reattivo ai cambi recenti).
                
                _last_eps_check_step = steps_done
                # Aggiorna il timestamp dell'ultimo check.

                if short_wr < EPS_BOOST_THRESHOLD and not _eps_boosted:
                # Win-rate basso E non abbiamo già un boost attivo:
                    epsilon = max(epsilon, EPS_BOOST_VALUE)
                    # Alza epsilon al massimo tra il valore corrente e EPS_BOOST_VALUE (0.15).
                    # max() assicura di non abbassare epsilon se è già più alto di 0.15.
                    
                    _eps_boosted = True
                    # Segnala che il boost è attivo.
                    
                    print(f"[EPS-BOOST] win_rate={short_wr:.1%} < {EPS_BOOST_THRESHOLD:.0%}"
                          f" — epsilon portato a {epsilon:.4f} per favorire esplorazione")
                    
                elif short_wr >= EPS_BOOST_THRESHOLD and _eps_boosted:
                # Win-rate tornato accettabile E il boost era attivo:
                    epsilon = EPS_FLOOR
                    # Riporta epsilon al floor normale (0.05).
                    
                    _eps_boosted = False
                    # Segnala che il boost non è più attivo.
                    
                    print(f"[EPS-BOOST] win_rate={short_wr:.1%} ≥ {EPS_BOOST_THRESHOLD:.0%}"
                          f" — epsilon ripristinato a {epsilon:.4f}")

            # ═════════════════════════════════════════════════════════════════
            # FASE 1: ACTION
            # L'agente osserva lo stato corrente del gioco e sceglie un'azione.
            # Questa fase si attiva ogni volta che ai_state == "ACTION"
            # e il player è fermo (IdlingAcc() == True).
            # ═════════════════════════════════════════════════════════════════
            
            AI_ACTIVE = TRAIN_AI or (PLAY_MODE and AI_AVAILABLE)
            # True se c'è un agente AI attivo (sia in training che in play mode).
            
            if AI_ACTIVE and inGame and ai_state == "ACTION" and player.IdlingAcc():
            # Condizioni per attivare la fase ACTION:
            # 1. AI è attiva
            # 2. Siamo in gioco (non menu, non game over)
            # 3. Lo stato è "ACTION" (non stiamo aspettando o imparando)
            # 4. Il player è idle: ha finito l'animazione della mossa precedente

                # --- OSSERVAZIONE DELLO STATO PRIMA DELL'AZIONE ---
                
                pt_prev_score        = player.scoreAcc()
                # Punteggio attuale (da confrontare dopo l'azione per vedere se è aumentato).
                
                pt_prev_oxy          = player.oxyAcc()
                # Ossigeno attuale (per calcolare la penalità da consumo di ossigeno).
                
                pt_prev_lives        = player.livesAcc()
                # Vite attuali (per rilevare se il player ha perso una vita).
                
                pt_prev_y, pt_prev_x = player.posAcc()
                # Posizione attuale (riga, colonna) nella griglia.

                pt_grid_old = stacker.get_state()
                # Ottiene il tensore visivo attuale: shape (24, 11, 9).
                # Questo è lo "stato" visivo che entra nella rete neurale.
                
                pt_vars_old = get_internal_state_vector(
                    player, len(level), len(level[0]), level, last_action_idx, n_actions
                )
                # Calcola il vettore delle variabili interne (ossigeno, vite, profondità, ecc.).
                # Shape: (STATE_DIM,) = (32,).

                # --- SCANSIONE BLOCCHI ADIACENTI ---
                # Osserva i blocchi vicini per dare informazioni extra al reward shaper.
                pt_blocks_around = {}
                try:
                    lateral_offsets = {
                        "down":  (pt_prev_y + 1, pt_prev_x),
                        # Blocco direttamente sotto.
                        "left":  (pt_prev_y,     pt_prev_x - 1),
                        # Blocco a sinistra.
                        "right": (pt_prev_y,     pt_prev_x + 1),
                        # Blocco a destra.
                    }
                    
                    for direction, (ty, tx) in lateral_offsets.items():
                    # Itera sulle tre direzioni.
                        
                        if 0 <= ty < len(level) and 0 <= tx < len(level[0]):
                        # Verifica che le coordinate siano dentro la griglia.
                            
                            blk = level[ty][tx]
                            # Recupera il blocco in quella posizione.
                            
                            pt_blocks_around[direction] = {
                                "hp":      blk.hpAccess(),
                                # Punti vita del blocco (0 = aria).
                                "type":    blk.typeAccess(),
                                # Tipo del blocco (stringa).
                                "falling": getattr(blk, "isFalling", lambda: False)(),
                                # True se il blocco sta cadendo.
                                # getattr con lambda: safe getter (non crashare se il metodo non esiste).
                                "shaking": getattr(blk, "isShaking", lambda: False)(),
                                # True se il blocco sta tremando (pre-caduta).
                            }

                    for k in range(1, 4):
                    # Scansiona le 3 celle sopra il player (k=1: direttamente sopra, k=2: 2 sopra, k=3: 3 sopra).
                    # Serve per rilevare blocchi in caduta dall'alto PRIMA che colpiscano il player.
                        
                        ty  = pt_prev_y - k
                        # Riga k righe sopra il player.
                        
                        key = "up" if k == 1 else f"above_{k}"
                        # Chiave del dizionario: "up" per 1 sopra, "above_2" e "above_3" per i successivi.
                        
                        if 0 <= ty < len(level):
                        # Verifica che non siamo usciti dalla griglia verso l'alto.
                            
                            blk = level[ty][pt_prev_x]
                            pt_blocks_around[key] = {
                                "hp":      blk.hpAccess(),
                                "type":    blk.typeAccess(),
                                "falling": getattr(blk, "isFalling", lambda: False)(),
                                "shaking": getattr(blk, "isShaking", lambda: False)(),
                            }
                except Exception:
                    pass
                # Se qualsiasi cosa va storta nella scansione (es. accesso fuori bounds),
                # ignora silenziosamente. pt_blocks_around rimarrà con i dati già inseriti.

                # --- SELEZIONE AZIONE ---
                pt_action, q_values_np = select_action(
                    policy_net, pt_grid_old, pt_vars_old, epsilon, n_actions
                )
                # select_action implementa epsilon-greedy:
                # - Con prob epsilon: azione casuale (randint 0-5)
                # - Con prob (1-epsilon): argmax dei Q-values dalla rete
                # Ritorna:
                #   pt_action: tensore scalare con l'indice dell'azione scelta (0-5)
                #   q_values_np: array numpy con i 6 Q-values calcolati (o None se casuale)
                
                action_idx      = pt_action.item()
                # .item() converte il tensore scalare PyTorch in un intero Python.
                
                last_action_idx = action_idx
                # Aggiorna l'ultima azione per il prossimo stato interno e il rendering HUD.

                # --- IDENTIFICAZIONE TIPO DI BLOCCO TARGET ---
                # Prima di eseguire l'azione, controlla su che tipo di blocco l'AI trivellare.
                # Serve per dare reward speciale ai blocchi difficili o a tempo.
                
                pt_was_hard_block = pt_was_delayed_block = False
                
                if action_idx in [3, 4, 5]:
                # Solo per azioni di trivellamento (non per movimento o idle).
                    
                    tx, ty = int(pt_prev_x), int(pt_prev_y)
                    # Posizione corrente del player (da usare come base per il calcolo del target).
                    
                    if action_idx == 3: tx -= 1    # Trivella sinistra → colonna -1
                    elif action_idx == 4: tx += 1  # Trivella destra → colonna +1
                    elif action_idx == 5: ty += 1  # Trivella giù → riga +1
                    # Calcola la posizione del blocco che verrebbe colpito.
                    
                    if 0 <= ty < len(level) and 0 <= tx < len(level[0]):
                    # Verifica che il target sia dentro la griglia.
                        
                        tb = level[ty][tx]
                        # Il blocco target.
                        
                        if tb.hpAccess() > 0:
                        # Solo se il blocco non è già distrutto.
                            
                            if   tb.typeAccess() == "unbreakable": pt_was_hard_block    = True
                            # "unbreakable": blocco marrone indistruttibile.
                            # Trivellarlo non fa niente (azione inutile) → verrà penalizzata.
                            
                            elif tb.typeAccess() == "delayed":     pt_was_delayed_block = True
                            # "delayed": blocco a tempo. Trivellarlo avvia il countdown.
                            # Può dare un bonus perché richiede pianificazione.

                # --- ESECUZIONE DELL'AZIONE ---
                # Traduce l'indice numerico in un evento Pygame simulato.
                
                if action_idx == 1:
                    movementHandle(pygame.event.Event(KEYDOWN, key=movKeys[2]),
                                   surface, player, level, movKeys)
                # Azione 1 = LEFT → simula la pressione del tasto A (movKeys[2]).
                # Crea un evento Pygame KEYDOWN artificialmente e lo processa.
                
                elif action_idx == 2:
                    movementHandle(pygame.event.Event(KEYDOWN, key=movKeys[1]),
                                   surface, player, level, movKeys)
                # Azione 2 = RIGHT → simula la pressione del tasto D (movKeys[1]).
                
                elif action_idx == 3:
                    breaking(pygame.event.Event(KEYDOWN, key=K_LEFT),
                             surface, player, level, currentBotLine)
                # Azione 3 = DRILL_L → simula la pressione di freccia sinistra.
                # breaking() rompe il blocco alla sinistra del player se possibile.
                
                elif action_idx == 4:
                    breaking(pygame.event.Event(KEYDOWN, key=K_RIGHT),
                             surface, player, level, currentBotLine)
                # Azione 4 = DRILL_R → freccia destra.
                
                elif action_idx == 5:
                    breaking(pygame.event.Event(KEYDOWN, key=K_DOWN),
                             surface, player, level, currentBotLine)
                # Azione 5 = DRILL_D → freccia giù.
                # action_idx == 0 (IDLE) non fa niente: il player aspetta.

                pt_post_action_y, pt_post_action_x = player.posAcc()
                # Cattura la posizione SUBITO dopo l'azione, PRIMA della gravità.
                # Se il player si è mosso, le coordinate cambieranno.
                # Se l'azione non ha avuto effetto (es. muro), resteranno uguali a pt_prev_y/x.

                steps_done += 1
                # Incrementa il contatore globale di step. NON si resetta mai tra episodi.
                
                gravity_applied_since_action = False
                # Resetta il flag: la gravità non è ancora stata applicata dopo questa azione.
                
                ai_state = "WAITING"
                # Transizione di stato: ora aspettiamo che la fisica risponda all'azione.

            # Pump eventi headless: in training headless, invece di aspettare i timer reali,
            # iniettiamo manualmente gli eventi nella coda di Pygame.
            if HEADLESS_TRAINING and inGame and ai_state == "WAITING":
                _hl_tick += 1
                # Incrementa il contatore del tick headless.
                
                pygame.event.post(pygame.event.Event(EVTIC_GRAVITY))
                # Inietta il tick di gravità. I blocchi avanzano di una fase.
                
                pygame.event.post(pygame.event.Event(EVTICPLY))
                pygame.event.post(pygame.event.Event(EVTICPLY))
                pygame.event.post(pygame.event.Event(EVTICPLY))
                # Inietta 3 tick player. 3x per simulare il passare di ~360ms
                # (3 × 120ms) che normalmente servono al player per tornare idle.
                
                if _hl_tick % 3 == 0:
                    pygame.event.post(pygame.event.Event(EVTICSEC))
                # Ogni 3 tick headless, inietta 1 tick ossigeno.
                # Rapporto approssimato per simulare il consumo di ossigeno reale.

            # =========================================================================
            # ELABORAZIONE EVENTI PYGAME
            # Processa tutti gli eventi nella coda: input utente, timer, segnali di sistema.
            # =========================================================================
            for event in pygame.event.get():
            # pygame.event.get() svuota la coda degli eventi e restituisce una lista.
            # Ogni iterazione del loop principale processa tutti gli eventi accumulati.

                if event.type == QUIT:
                # L'utente ha chiuso la finestra del programma.
                    
                    if TRAIN_AI:
                        save_checkpoint(policy_net, optimizer, steps_done, epsilon, scheduler,
                                        curriculum=curriculum)
                        # Salva il modello prima di uscire (per non perdere il training).
                        
                        memory.save_async(BUFFER_DIR)
                        # Salva il buffer di replay su disco (asincrono).
                    
                    inProgress = False
                    # Ferma il loop principale.

                if event.type == EVT_END_BLOCK:
                # Il giocatore ha rotto il blocco finale del livello (blocco obiettivo).
                    won = True
                    # Imposta il flag di vittoria. Verrà processato nella sezione "Fine Episodio".

                if inGame:
                # Tutti i seguenti eventi vengono processati solo se il gioco è attivo.

                    if event.type == EVTICSEC:
                    # Tick del "secondo di gioco": ogni ~1050ms.
                        
                        player.updateOxygen(1, surface, level)
                        # Sottrae 1 punto di ossigeno al player.
                        # Se arriva a 0, il player muore (gestito internamente da updateOxygen).

                        dead_delayed: set = set()
                        # Set locale dei blocchi "delayed" da rimuovere in questo tick.
                        
                        for (r, c) in blocksDisap:
                        # Itera su tutti i blocchi "delayed" in attesa di scomparire.
                            
                            if level[r][c].hpAccess() > 0:
                            # Il blocco è ancora vivo.
                                
                                if hasattr(level[r][c],'timeout') and level[r][c].timeout():
                                # timeout() restituisce True quando il timer del blocco è scaduto.
                                # hasattr: verifica che il metodo esista (sicurezza).
                                    dead_delayed.add((r, c))
                                    # Il timer è scaduto: questo blocco deve scomparire.
                            else:
                                dead_delayed.add((r, c))
                                # L'HP è già 0 (blocco già distrutto): rimuovi dal tracking.

                        for (r, c) in dead_delayed:
                        # Per ogni blocco scaduto/distrutto:
                            
                            if level[r][c].hpAccess() == 0:
                                nb = Classic(c, r, 1, 0)
                                # Crea un nuovo blocco "classic" nella stessa posizione.
                                
                                nb._hp = 0
                                # Imposta HP a 0 → è aria/vuoto.
                                
                                nb.changeBG(levelID)
                                # Applica lo sfondo visivo corretto per il livello corrente.
                                
                                level[r][c] = nb
                                # Sostituisce il blocco nella mappa con questo placeholder vuoto.

                        blocksDisap -= dead_delayed
                        # Rimuove i blocchi scomparsi dal set di tracking.
                        # L'operatore -= su set è l'operazione "differenza": rimuove tutti gli elementi di dead_delayed.

                        if dead_delayed:
                        # Se almeno un blocco è scomparso, applica la gravità per far cadere quelli sopra.
                            applyGravity(level, player, surface)
                            player.fall(surface, level)
                            # player.fall() gestisce il caso in cui il player si trovi nel vuoto
                            # e debba "cadere" di conseguenza.

                    if event.type == EVTIC_ANIM:
                    # Tick animazione: aggiorna il frame corrente dello sprite del player.
                        player.Anim(surface)
                        # Cambia il frame dell'animazione del player (es. gambe che camminano).

                    if event.type == EVTIC_GRAVITY:
                    # Il tick più importante: avanza di una fase la fisica dei blocchi.
                        
                        applyGravity(level, player, surface)
                        # Esegue un ciclo completo della fisica: tutti i blocchi avanzano di una fase.
                        
                        player.fall(surface, level)
                        # Se il player è nel vuoto (blocco sotto distrutto), lo fa cadere di una riga.
                        
                        gravity_applied_since_action = True
                        # Segnala che la gravità è stata applicata dopo l'ultima azione dell'AI.
                        # Usato nella logica di transizione WAITING → LEARNING.

                        if player.blocksFallenAcc() != currentOffset:
                        # blocksFallenAcc() restituisce quanti livelli il player è sceso in totale.
                        # Se è cambiato rispetto all'offset corrente della telecamera, scrolliamo.
                            
                            currentOffset  += 1
                            # Aumenta l'offset della telecamera di 1 riga.
                            
                            currentBotLine += 1
                            # La riga più in basso visibile scende di 1.
                            
                            for line in level:
                                for el in line:
                                    el.updOffset(currentOffset)
                            # Aggiorna tutti i blocchi con il nuovo offset
                            # (serve per il rendering corretto della posizione visiva).

                        if AI_ACTIVE and inGame and stacker is not None:
                        # Se l'AI è attiva, aggiunge il frame corrente allo stack visivo.
                            
                            _gy, _gx = player.posAcc()
                            # Posizione attuale del player dopo la gravità.
                            
                            _gf = get_local_window_tensor(level, _gx, _gy).to(device)
                            # Genera il frame visivo centrato sulla nuova posizione.
                            
                            stacker.push(_gf)
                            # Aggiunge il frame allo stack, rimuovendo il più vecchio.
                            # Così l'AI "vede" il movimento della scena durante la caduta.

                    if event.type == EVTICPLY:
                    # Tick player: verifica se il player ha finito di muoversi.
                        player.NeedToIdle(surface)
                        # Aggiorna lo stato del player: se ha finito l'animazione corrente,
                        # torna allo stato "idle" (pronto per la prossima azione).

                    if event.type in (EVTIC_GRAVITY, EVTICSEC):
                    # Dopo ogni tick di gravità o di ossigeno, aggiorna il tracking dei delayed blocks.
                        
                        for r, row in enumerate(level):
                        # Itera su tutte le righe.
                            
                            for c, blk in enumerate(row):
                            # Itera su tutti i blocchi della riga.
                                
                                if blk.typeAccess()=="delayed" and blk.idAcc() and blk.hpAccess()>0:
                                # Se il blocco è di tipo "delayed", ha un ID timer attivo, ed è ancora vivo...
                                    blocksDisap.add((r, c))
                                    # Aggiunge al set di tracking per il countdown.

            # Gestione della risalita della telecamera (quando il player sale su blocchi).
            if backDown and player.climbAcc() < currentClimb:
            # Se stavamo tracciando una risalita (backDown=True) e il climb è diminuito:
                
                if currentClimb == 0:
                    backDown = False
                    # Se il climb è tornato a 0, la risalita è completata.
                
                player.backDownCleanup(surface)
                # Esegue la pulizia visiva della risalita (riposiziona la griglia visiva).

            currentClimb = player.climbAcc()
            # Aggiorna il valore corrente del climb dal player.
            
            if player.climbAcc() > 0:
                backDown = True
            # Se il player si è arrampicato (climb > 0), attiva il tracking della risalita.

            # ═════════════════════════════════════════════════════════════════
            # FASE 2: WAITING
            # L'AI ha eseguito un'azione e aspetta che la fisica del gioco abbia
            # risposto completamente prima di valutare il risultato.
            # ═════════════════════════════════════════════════════════════════
            if AI_ACTIVE and inGame and ai_state == "WAITING":
                
                is_terminal = (player.livesAcc() < 0) or won
                # True se l'episodio deve finire: player morto (lives < 0) o livello completato (won).
                
                idle_action = (last_action_idx == 0)
                # True se l'ultima azione era IDLE: non aspettare la gravità, vai subito a LEARNING.

                action_moved_immediate = (
                    (pt_post_action_y != pt_prev_y) or
                    (pt_post_action_x != pt_prev_x)
                )
                # True se il player si è effettivamente spostato subito dopo l'azione.
                # Confronta la posizione SUBITO dopo l'azione (pt_post_action) con quella PRIMA (pt_prev).
                # Se sono uguali → l'azione non ha avuto effetto immediato (muro o blocco indistruttibile).

                if last_action_idx != 0:
                # Per azioni non-IDLE:
                    effective_fails = drill_tracker.consecutive_fails + (
                        0 if action_moved_immediate else 1
                    )
                    # Conta i fallimenti effettivi:
                    # Se l'azione si è mossa → add 0 (non è un fallimento)
                    # Se non si è mossa → add 1 (questa azione era inutile)
                    # consecutive_fails: i fallimenti precedenti già tracciati
                else:
                    effective_fails = 0
                # IDLE non è un "fallimento".

                # --- LOGICA DI TRANSIZIONE A LEARNING ---
                
                if is_terminal or idle_action:
                    ai_state = "LEARNING"
                    # Se terminale o IDLE: passa subito a Learning, senza aspettare la gravità.
                    
                elif effective_fails >= MAX_CONSEC_FAILS:
                # Troppi fallimenti consecutivi: l'AI è bloccata in un loop di azioni inutili.
                    if gravity_applied_since_action and player.IdlingAcc():
                        ai_state = "LEARNING"
                        # Forza il Learning anche se non saremmo pronti, per sbloccare il loop.
                        
                elif gravity_applied_since_action and player.IdlingAcc():
                # Condizione normale: la gravità è stata applicata E il player è idle.
                    ai_state = "LEARNING"
                    # L'azione ha avuto effetti completi: ora possiamo valutare il risultato.

            # ═════════════════════════════════════════════════════════════════
            # FASE 3: LEARNING
            # L'AI valuta il risultato della sua azione, calcola la reward,
            # e aggiorna i pesi della rete neurale.
            # ═════════════════════════════════════════════════════════════════
            if TRAIN_AI and inGame and ai_state == "LEARNING":

                new_pY, new_pX = player.posAcc()
                # Posizione attuale del player DOPO che tutto è avvenuto.

                if not gravity_applied_since_action:
                # Se la gravità non è stata applicata (es. azione IDLE o terminale immediata),
                # aggiungiamo manualmente un frame allo stack per tenerlo aggiornato.
                    _ff = get_local_window_tensor(level, new_pX, new_pY).to(device)
                    stacker.push(_ff)

                grid_new = stacker.get_state()
                # Tensore visivo del NUOVO stato (dopo l'azione). Shape: (24, 11, 9).
                
                vars_new = get_internal_state_vector(
                    player, len(level), len(level[0]), level, last_action_idx, n_actions
                )
                # Vettore variabili interne del NUOVO stato. Shape: (32,).

                shaping_bonus = shaper.step(new_pY, player.oxyAcc(), len(level), GAMMA)
                # Calcola il bonus PBRS per questo step:
                # Φ(s') - γ * Φ(s), dove Φ è il potenziale (funzione della profondità).
                # Positivo se il player è sceso, negativo se è salito.
                # Questo guida l'agente verso il basso senza cambiare la policy ottimale teorica.

                if player.livesAcc() < pt_prev_lives:
                # Se le vite sono diminuite, determina la causa della morte.
                    _last_death_cause = (
                        "oxy" if pt_prev_oxy <= OXY_DEATH_THRESHOLD else "block_fall"
                    )
                    # Se l'ossigeno era basso (≤2) prima della morte → causa: mancanza ossigeno.
                    # Altrimenti → causa: schiacciato da un blocco.
                    # Questa distinzione serve per i log del monitor.

                # --- CALCOLO REWARD ---
                reward = calculate_reward(
                    prev_y=pt_prev_y, prev_x=pt_prev_x,
                    # Posizione prima dell'azione.
                    new_y=new_pY, new_x=new_pX,
                    # Posizione dopo l'azione.
                    post_action_y=pt_post_action_y,
                    post_action_x=pt_post_action_x,
                    # Posizione subito dopo l'azione (prima della gravità).
                    prev_oxy=pt_prev_oxy, new_oxy=player.oxyAcc(),
                    # Ossigeno prima e dopo.
                    prev_score=pt_prev_score, new_score=player.scoreAcc(),
                    # Punteggio prima e dopo.
                    prev_lives=pt_prev_lives, new_lives=player.livesAcc(),
                    # Vite prima e dopo.
                    action_idx=last_action_idx,
                    # L'azione eseguita (0-5).
                    is_hard_block=pt_was_hard_block,
                    # Se ha trivellato un blocco indistruttibile.
                    is_delayed_block=pt_was_delayed_block,
                    # Se ha trivellato un blocco a tempo.
                    drill_tracker=drill_tracker,
                    # Il tracker per penalizzare loop e oscillazioni.
                    total_rows=len(level),
                    # Altezza totale del livello (per normalizzare la profondità).
                    shaping_bonus=shaping_bonus,
                    # Il bonus di reward shaping PBRS.
                    is_level_complete=won,
                    # True se ha appena vinto il livello.
                    blocks_around=pt_blocks_around,
                    # Dizionario con i blocchi adiacenti.
                    is_game_over=(player.livesAcc() < 0),
                    # True se ha perso tutte le vite.
                )
                # calculate_reward restituisce un tensore PyTorch scalare con la reward totale.
                # Questo numero (es. +2.5, -1.0, +10.0) è il "voto" assegnato all'azione.

                done = torch.tensor([player.livesAcc() < 0 or won], device=device)
                # Tensore booleano che indica se l'episodio è terminato dopo questa transizione.
                # True se: morto (lives < 0) OPPURE livello completato (won).
                # Usato nell'equazione di Bellman: se done=True, non si considera il futuro.

                nstep_buf.push(pt_grid_old, pt_vars_old, pt_action, grid_new, vars_new, reward, done)
                # Aggiunge la transizione (s, a, s', r, done) al buffer N-step.
                # Il buffer accumula N transizioni, poi calcola la reward accumulata
                # e crea la transizione finale per la memoria PER.
                # (pt_grid_old, pt_vars_old): stato VECCHIO
                # pt_action: azione eseguita
                # (grid_new, vars_new): stato NUOVO
                # reward: reward calcolata
                # done: flag terminale

                if steps_done % 100 == 0:
                # Log ogni 100 step (non ogni step per non riempire il terminale).
                    in_warmup = len(memory) < WARMUP_STEPS
                    # True se siamo ancora nel periodo di warmup.
                    
                    warmup_tag = f" [WARMUP {len(memory)}/{WARMUP_STEPS}]" if in_warmup else ""
                    # Stringa di log per il warmup.
                    
                    eps_tag    = " [BOOSTED]" if _eps_boosted else ""
                    # Stringa di log per il boost epsilon.
                    
                    print(f"[STEP {steps_done:6d}] R: {reward.item():+7.2f} | dy: {new_pY-pt_prev_y} | "
                          f"Oxy: {int(player.oxyAcc()):3d}% | e: {epsilon:.4f}{eps_tag} | "
                          f"buf: {len(memory)} | CurrLv: {curriculum.current_max_level}{warmup_tag}")
                    # Log dettagliato:
                    # R: reward con segno (positiva o negativa)
                    # dy: variazione di profondità (positivo = sceso, negativo = salito)
                    # Oxy: ossigeno rimanente in %
                    # e: epsilon corrente
                    # buf: dimensione del buffer di memoria
                    # CurrLv: livello massimo corrente del curriculum

                if nstep_buf.is_ready():
                # Se il buffer ha accumulato N=8 transizioni, è pronto a produrre output.
                    
                    nstep_trans = nstep_buf.get()
                    # Estrae la transizione N-step: (s0, v0, a0, sN, vN, R_nstep, done)
                    # dove R_nstep = r0 + γ*r1 + ... + γ^7*r7 + γ^8 * Q(sN)
                    
                    if nstep_trans:
                        memory.push(*nstep_trans)
                    # Aggiunge la transizione N-step al Replay Buffer principale.
                    # *nstep_trans scompatta la tupla come argomenti separati.

                _train_counter += 1
                # Incrementa il contatore per TRAIN_EVERY.
                
                opt_result = None
                # Risultato dell'ottimizzazione (loss, q_mean). None se non ottimizzato.

                if _train_counter >= TRAIN_EVERY and len(memory) >= WARMUP_STEPS:
                # Condizioni per eseguire la backpropagation:
                # 1. Sono passati TRAIN_EVERY (3) step dall'ultimo training
                # 2. Il buffer ha abbastanza esperienze (almeno WARMUP_STEPS)
                    
                    opt_result = optimize_model(
                        policy_net, target_net, memory, optimizer, BATCH_SIZE, GAMMA
                    )
                    # Esegue un passo di ottimizzazione:
                    # 1. Campiona BATCH_SIZE transizioni dalla memoria PER
                    # 2. Calcola Q-values predetti (policy net forward pass)
                    # 3. Calcola Q-values target (target net forward pass)
                    # 4. Calcola Huber Loss (robusto agli outliers rispetto a MSE)
                    # 5. Backpropagation (calcola gradienti)
                    # 6. Gradient clipping (norm massima 10 per stabilità)
                    # 7. Optimizer step (aggiorna i pesi)
                    # 8. Aggiorna le priorità nel PER buffer
                    
                    if opt_result is not None:
                        scheduler.step()
                    # Avanza il learning rate scheduler di un passo (solo se l'ottimizzazione ha avuto luogo).
                    
                    _train_counter = 0
                    # Resetta il contatore TRAIN_EVERY.
                    
                elif _train_counter >= TRAIN_EVERY:
                # Se è ora di ottimizzare ma siamo ancora in warmup:
                    _train_counter = 0
                    # Resetta comunque il contatore.

                if monitor:
                # Salva le metriche di training nel CSV (solo se il monitor è attivo).
                    
                    loss_val    = opt_result[0] if opt_result else None
                    # Valore della loss dell'ultimo batch (None se non ottimizzato in questo step).
                    
                    q_mean_safe = q_values_np if q_values_np is not None else np.zeros(n_actions, dtype=np.float32)
                    # Q-values correnti (o array di zeri come fallback).
                    
                    monitor.log_step(
                        reward=reward.item(),
                        # Reward di questo step.
                        action_idx=last_action_idx,
                        # Azione eseguita (0-5).
                        depth=currentOffset-currentClimb,
                        # Profondità reale = offset telecamera - risalita.
                        oxy=player.oxyAcc(),
                        # Ossigeno corrente.
                        loss=loss_val,
                        # Loss dell'ultimo batch.
                        epsilon=epsilon,
                        # Epsilon corrente.
                        steps_done=steps_done,
                        # Step totali.
                        q_mean=q_mean_safe,
                        # Q-values per tutte le azioni.
                        lr=scheduler.get_last_lr()[0] if scheduler else LEARNING_RATE,
                        # Learning rate corrente dello scheduler.
                        mem_size=len(memory),
                        # Dimensione del buffer.
                        was_hard_block=pt_was_hard_block,
                        was_delayed_block=pt_was_delayed_block,
                        # Tipo di blocco trivellato.
                        player_x=new_pX, player_y=new_pY
                        # Posizione finale del player.
                    )

                if steps_done % TARGET_UPDATE == 0:
                # Ogni TARGET_UPDATE (1000) step: aggiorna la Target Net.
                    target_net.load_state_dict(policy_net.state_dict())
                    # Copia esattamente i pesi della Policy Net nella Target Net.
                    # Dopo questa copia, i Q-values target saranno calcolati con la rete aggiornata.

                if steps_done % 1000 == 0 and steps_done > 0:
                # Salva il checkpoint ogni 1000 step (e non allo step 0).
                    save_checkpoint(policy_net, optimizer, steps_done, epsilon, scheduler,
                                    curriculum=curriculum)

                if steps_done % BUFFER_SAVE_EVERY == 0 and steps_done > 0:
                # Salva il buffer su disco ogni BUFFER_SAVE_EVERY (5000) step.
                    memory.save_async(BUFFER_DIR)
                    # save_async: salva in background senza bloccare il gioco.

                ai_state = "ACTION"
                # Transizione finale: torna alla fase ACTION per scegliere la prossima mossa.

            if PLAY_MODE and inGame and ai_state == "LEARNING":
            # In PLAY_MODE non c'è learning: salta direttamente ad ACTION.
                ai_state = "ACTION"

            # ═════════════════════════════════════════════════════════════════
            # FINE EPISODIO
            # Gestisce la transizione tra un episodio e il successivo.
            # Attivato quando il player muore O quando vince il livello.
            # ═════════════════════════════════════════════════════════════════
            if player.livesAcc() < 0 or won:
            # Condizione doppia: vite esaurite (< 0, non == 0) OPPURE livello completato.

                effective_max = curriculum.current_max_level if TRAIN_AI else MAX_LEVEL
                # In training: livello massimo del curriculum. In play: il livello 10 assoluto.

                lives_end = max(player.livesAcc(), 0)
                # Le vite finali. max(lives, 0) evita di mostrare numeri negativi nei log.
                
                ep_tag    = "WIN " if won else "LOSS"
                # Tag per il log: "WIN " (con spazio per allineamento) o "LOSS".
                
                print(f"[EP END] {ep_tag} | Lv{levelID} | lives={lives_end} | "
                      f"score={player.scoreAcc()} | oxy={int(player.oxyAcc()):3d}% | "
                      f"step={steps_done} | curr_max={effective_max}")
                # Log di fine episodio con tutte le statistiche principali.

                if TRAIN_AI and monitor:
                # Registra l'episodio nel CSV del monitor.
                    death_cause = "win" if won else _last_death_cause
                    monitor.log_episode(
                        steps_done=steps_done, level_id=levelID, won=won,
                        death_cause=death_cause,
                        # "win", "oxy", o "block_fall".
                        depth=currentOffset-currentClimb,
                        final_oxy=player.oxyAcc(),
                        score=player.scoreAcc(),
                        lives_left=lives_end
                    )

                if won:
                # --- GESTIONE VITTORIA ---
                    
                    if levelID >= effective_max:
                    # Ha vinto il livello massimo sbloccato dal curriculum.
                        
                        if TRAIN_AI:
                            curriculum.record_episode(completed_max=True)
                            # Registra questa vittoria nell'history del curriculum.
                            
                            training_done = curriculum.check_and_advance()
                            # Controlla se il win-rate è sufficiente per avanzare di livello.

                            if training_done:
                            # Tutto il curriculum è completato! Training finito.
                                save_checkpoint(policy_net, optimizer, steps_done, epsilon, scheduler,
                                                curriculum=curriculum)
                                # Salvataggio finale.
                                
                                memory.save_async(BUFFER_DIR)
                                # Salvataggio finale del buffer.
                                
                                if monitor:
                                    monitor.print_summary()
                                    # Stampa un riassunto statistico.
                                    monitor.plot_all()
                                    # Genera i grafici PDF.
                                    monitor.close()
                                    # Chiude i file di log.
                                
                                print("[CURRICULUM] Training completato. Chiusura.")
                                inProgress = False
                                # Ferma il loop: il training è finito!
                                continue

                            # Non ancora completato: resetta e ricomincia dal livello 1.
                            level, levelID, won = restart(player)
                            play_music(levelID)
                            (currentBotLine, currentOffset, blocksDisap, ai_state,
                             gravity_applied_since_action, backDown, currentClimb) = \
                                _reset_game_state(player, level, levelID)
                            _reset_ai_episode(player, level, stacker, nstep_buf, shaper, drill_tracker, memory=memory)
                            _last_death_cause = "block_fall"
                            continue

                        else:
                        # PLAY_MODE: mostra la schermata di vittoria finale.
                            inGame = False; inGameOver = True; game_won = True; won = False
                            continue

                    # Ha vinto ma NON è al livello massimo: avanza al prossimo livello.
                    level, levelID, game_completed = changeLvl(levelID, player, is_ai=(TRAIN_AI or PLAY_MODE))
                    # changeLvl crea la mappa del livello successivo.
                    # game_completed=True se levelID era già l'ultimo livello assoluto.
                    
                    if game_completed:
                        level, levelID, _ = restart(player)
                        # Se il gioco è completato (tutti i 10 livelli), riavvia dal livello 1.

                    if TRAIN_AI and levelID > effective_max:
                    # Protezione bug: se per qualche motivo il nuovo levelID supera il massimo del curriculum...
                        print(f"[BUG-K] levelID {levelID} > effective_max {effective_max} "
                              f"dopo changeLvl — forzo restart.")
                        level, levelID, won = restart(player)
                        # Forza il restart per non uscire dai confini del curriculum.

                    won = False
                    # Resetta il flag won per il nuovo livello.

                else:
                # --- GESTIONE SCONFITTA ---
                    
                    if TRAIN_AI:
                        if levelID >= effective_max:
                            curriculum.record_episode(completed_max=False)
                            # Registra questa sconfitta nell'history del curriculum.
                            # (Solo se era al livello massimo: le morti ai livelli intermedi non contano.)

                    level, levelID, won = restart(player)
                    # Torna al livello 1 dopo una morte.

                play_music(levelID)
                # Avvia la musica del nuovo livello (o livello 1 se è stato un restart).

                (currentBotLine, currentOffset, blocksDisap, ai_state,
                 gravity_applied_since_action, backDown, currentClimb) = \
                    _reset_game_state(player, level, levelID)
                # Resetta le variabili di stato del gioco.

                if TRAIN_AI:
                    _reset_ai_episode(player, level, stacker, nstep_buf, shaper, drill_tracker, memory=memory)
                    # Resetta i componenti AI per il nuovo episodio (con flush del nstep_buf nel memory).
                elif PLAY_MODE and AI_AVAILABLE:
                    _reset_ai_episode(player, level, stacker, None, shaper, drill_tracker)
                    # In PLAY_MODE: resetta senza buffer (nstep_buf=None).

                pt_post_action_y, pt_post_action_x = player.posAcc()
                # Aggiorna la posizione "post-azione" con quella iniziale del nuovo episodio.
                
                _last_death_cause = "block_fall"
                # Resetta la causa di morte per il nuovo episodio (default: blocco).

                continue
                # Salta il rendering di questo frame e vai al prossimo ciclo del loop.

            # ═════════════════════════════════════════════════════════════════
            # RENDERING VISIVO
            # Disegna il frame corrente solo se la finestra è visibile.
            # Saltato completamente in modalità headless per massima velocità.
            # ═════════════════════════════════════════════════════════════════
            if inGame and not HEADLESS_TRAINING:
                
                surface.fill((10, 5, 20))
                # Riempie tutta la finestra con un colore blu scurissimo (quasi nero).
                # Questo cancella il frame precedente. (10, 5, 20) = RGB very dark blue/purple.
                
                surface.blit(Ui_bg, (0, 0))
                # Disegna l'immagine dell'interfaccia (HUD destro) su tutta la finestra.
                # Il pannello del gioco sovrascriverà la parte sinistra.

                render(surface, level, currentOffset)
                # Disegna tutti i blocchi della mappa di gioco sulla parte sinistra della finestra.
                # currentOffset determina quali righe sono visibili (scorrimento verticale).
                
                player.display(surface)
                # Disegna lo sprite del player nella sua posizione corrente.

                # --- RENDERING PUNTEGGIO ---
                score_val  = player.scoreAcc()
                # Punteggio numerico attuale.
                
                score_text = f"{score_val}" if score_val < 1000 else f"{score_val/1000:.1f} k"
                # Formattazione intelligente: sotto 1000 mostra il numero intero,
                # sopra 1000 mostra in formato "1.2 k" (kilo = migliaia).
                
                surface.blit(big_font.render(score_text, True, (220, 0, 255)), (640, 107))
                # Renderizza il testo del punteggio in viola e lo disegna alle coordinate (640, 107).
                # (220, 0, 255) = RGB viola brillante.

                # --- RENDERING OSSIGENO ---
                oxy_val = max(0, min(100, int(player.oxyAcc())))
                # Valore ossigeno clampato tra 0 e 100 e convertito in intero.
                # max(0, min(100, v)) = clamp: assicura che sia nel range valido.
                
                surface.blit(big_font.render(str(oxy_val), True, (220, 0, 255)), (640, 200))
                # Renderizza il numero dell'ossigeno in viola.
                
                if oxy_frames and oxy_val in oxy_frames and oxy_frames[oxy_val]:
                    surface.blit(oxy_frames[oxy_val], (537, 252))
                # Se esiste il frame dell'animazione corrispondente a questo valore di ossigeno,
                # disegnalo alle coordinate (537, 252) (posizione della barra animata dell'O2).

                # --- RENDERING PROFONDITÀ ---
                real_depth = currentOffset - currentClimb
                # Profondità reale = quanto abbiamo scrollato verso il basso meno quanto è risalito il player.
                
                surface.blit(big_font.render(str(real_depth), True, (220, 0, 255)), (640, 377))
                # Renderizza la profondità in viola.

                # --- RENDERING ICONE VITE ---
                if icon:
                    for i in range(player.livesAcc()):
                    # Una icona per ogni vita rimanente.
                        surface.blit(icon, (700 - i*70, 500))
                    # Posiziona le icone da destra a sinistra: 700, 630, 560 px sull'asse X.
                    # Tutte alla riga Y=500 (vicino al fondo della finestra).

                # --- RENDERING HUD AI ---
                if AI_ACTIVE:
                    in_warmup = TRAIN_AI and len(memory) < WARMUP_STEPS
                    # True se siamo in training E ancora nel warmup.
                    
                    consec_display = drill_tracker.consecutive_fails if drill_tracker else 0
                    # Numero di fallimenti consecutivi da mostrare (0 se drill_tracker è None).
                    
                    draw_ai_debug(surface, debug_font, epsilon, steps_done, last_action_idx,
                                  ai_state, curriculum=curriculum if TRAIN_AI else None,
                                  in_warmup=in_warmup,
                                  consec_fails=consec_display)
                    # Disegna il pannello di debug AI nell'angolo in alto a sinistra.

                pygame.display.update()
                # Trasferisce tutto il contenuto della surface al buffer del display.
                # Senza questa chiamata, nulla di quello disegnato sopra sarebbe visibile.

            if not HEADLESS_TRAINING:
                fpsClock.tick(FPS)
                # Limita il loop a FPS (60) iterazioni al secondo.
                # Se il loop è più veloce di 60fps, aggiunge un sleep per pareggiare.
                # Garantisce velocità costante e non usa il 100% della CPU.

    # =========================================================================
    # GESTIONE ECCEZIONI E CHIUSURA
    # =========================================================================
    
    except KeyboardInterrupt:
        print("\n[INTERRUPT] KeyboardInterrupt ricevuto.")
        _emergency_save()
        # Intercetta CTRL+C: salva il modello prima di uscire bruscamente.

    finally:
    # Il blocco finally viene SEMPRE eseguito, sia in uscita normale che per eccezione.
        
        if TRAIN_AI and monitor:
            monitor.print_summary()
            # Stampa le statistiche finali del training.
            
            monitor.plot_all()
            # Genera tutti i grafici e li salva come PDF.
            
            monitor.close()
            # Chiude i file di log aperti (CSV, ecc.).

        pygame.quit()
        # Deinizializza tutti i moduli di Pygame correttamente.
        # Senza questo, la finestra potrebbe rimanere aperta o causare errori.
        
        sys.exit()
        # Termina il processo Python con codice di uscita 0 (successo).
        # Necessario perché dopo pygame.quit(), alcune librerie potrebbero lasciare thread aperti.


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Questo blocco viene eseguito SOLO quando il file viene lanciato direttamente:
    #   python main.py
    # NON viene eseguito se il file viene importato come modulo da un altro script:
    #   import main  → il blocco NON esegue
    # È la convenzione Python standard per distinguere "script principale" da "modulo riutilizzabile".
    
    game(SCREEN_W, SCREEN_H)
    # Avvia il gioco con le dimensioni dello schermo configurate (800x600).
    # Da qui in poi, tutto è controllato dal loop while dentro game().