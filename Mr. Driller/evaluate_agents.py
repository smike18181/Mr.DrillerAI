"""
evaluate_agents.py — Valutazione comparativa dei tre agenti Mr. Driller
========================================================================
Confronta in modo sistematico tre tipi di agente che giocano a Mr. Driller
in modalità "headless" (senza finestra grafica):

  1. dqn      → Rainbow DQN puro (rete neurale addestrata con Reinforcement Learning)
  2. llm_act  → LLM usato direttamente come policy (sceglie le azioni)
  3. llm_rew  → DQN addestrato con reward stimato da un LLM (pesi separati)

Per ogni agente vengono eseguiti N episodi del gioco e misurate metriche come:
win rate, profondità raggiunta, passi eseguiti, ossigeno rimasto, punteggio,
distribuzione delle azioni e intervalli di confidenza bootstrap.
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORT E SETUP AMBIENTE
# ──────────────────────────────────────────────────────────────────────────────

import os         # per variabili d'ambiente e path di file
import sys        # per modificare il sys.path
import csv        # per salvare i risultati in formato CSV
import json       # per salvare i riepiloghi in formato JSON
import math       # per log2 (entropia di Shannon), sqrt (deviazione standard)
import time       # per misurare la durata degli episodi
import random     # per generare numeri casuali (seed)
import argparse   # per leggere argomenti da riga di comando
import datetime   # per generare timestamp nei nomi dei file di output
from collections import Counter         # per contare le cause di morte
from dataclasses import dataclass, field, asdict  # per strutture dati leggibili
from typing import Optional             # per annotazioni di tipo opzionali

# ── Silenzia output indesiderati da librerie esterne ─────────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # disabilita ottimizzazioni oneDNN di TensorFlow
os.environ["KMP_DUPLICATE_LIB_OK"]  = "True" # evita crash con OpenMP su macOS
os.environ["OMP_NUM_THREADS"]       = "1"    # limita i thread OpenMP a 1
os.environ["SDL_VIDEODRIVER"]       = "dummy" # dice a SDL (pygame) di non aprire finestre reali
os.environ["SDL_AUDIODRIVER"]       = "dummy" # dice a SDL di non usare audio reale

import numpy as np   # per operazioni numeriche su array e seed casuale
import pygame        # motore di gioco (gestione eventi, superficie grafica virtuale)
from pygame.locals import *  # importa costanti come K_w, K_a, K_s, K_d, KEYDOWN, ecc.

# ── Import classi di gioco ────────────────────────────────────────────────────
from character import Character   # classe del giocatore reale di Mr. Driller
from block import Classic, clear_cache  # classe blocco e funzione per pulire la cache
from menu import restart           # funzione che inizializza un nuovo livello di gioco

# ── Import agente DQN (opzionale) ────────────────────────────────────────────
# Se il modulo ai_agent non è installato, il benchmark procede senza DQN
try:
    from ai_agent import (
        DrillerDuelingDQN,           # architettura della rete neurale Dueling DQN
        select_action,               # funzione che usa la rete per scegliere un'azione
        device,                      # dispositivo di calcolo (CPU o CUDA GPU)
        get_local_window_tensor,     # estrae la finestra locale intorno al giocatore come tensore
        get_internal_state_vector,   # estrae il vettore di stato interno (ossigeno, vite, ecc.)
    )
    import torch          # framework per reti neurali
    AI_AVAILABLE = True   # flag: il DQN è disponibile
except ImportError as e:
    print(f"[EVAL] ai_agent non disponibile: {e}")
    AI_AVAILABLE = False  # il DQN non è disponibile

# ── Import LLMAgent (opzionale, con fallback) ─────────────────────────────────
# Tenta prima llm_agent (ottimizzata), poi llm_agent standard
try:
    from llm_agent import LLMAgent
    LLM_AGENT_AVAILABLE = True
    print("[EVAL] LLMAgent (fast) caricato correttamente")
except ImportError:
    try:
        from llm_agent import LLMAgent           # fallback alla versione standard
        LLM_AGENT_AVAILABLE = True
        print("[EVAL] llm_agent_fast non trovato — uso llm_agent standard")
    except ImportError as e:
        print(f"[EVAL] llm_agent non disponibile: {e}")
        LLM_AGENT_AVAILABLE = False  # nessun LLM disponibile

# ──────────────────────────────────────────────────────────────────────────────
# COSTANTI DI CONFIGURAZIONE
# ──────────────────────────────────────────────────────────────────────────────

# Forma dell'input per la rete DQN:
# (N_CHANNELS * N_FRAMES, altezza_finestra, larghezza_finestra) = (24, 11, 9)
# 6 canali per frame × 4 frame impilati = 24 canali totali
INPUT_SHAPE  = (6 * 4, 11, 9)

STATE_DIM    = 32   # dimensione del vettore di stato interno
N_ACTIONS    = 6    # numero totale di azioni: IDLE, LEFT, RIGHT, DRILL_L, DRILL_R, DRILL_D
MODEL_PATH   = "RenfLearningAgent/driller_ai_v31.pth"  # path al checkpoint del DQN

MAX_STEPS        = 2000  # numero massimo di azioni per episodio
MAX_STUCK_STEPS  = 80    # iterazioni del loop senza cambio posizione → episodio terminato
MAX_CONSEC_FAILS = 4     # fallimenti consecutivi prima di forzare la transizione WAITING→ACTION

LLM_MODEL_KEY = "llama"  # modello LLM di default per gli agenti llm_act e llm_rew

# Proporzioni timer del gioco reale (documentano le frequenze degli eventi)
GAME_SPEED  = 1.0
_T_GRAVITY  = 400.0  / GAME_SPEED   # 400 ms: intervallo tra due applicazioni di gravità
_T_ANIM     = 80.0   / GAME_SPEED   # 80 ms: intervallo tra frame di animazione
_T_SEC      = 1050.0 / GAME_SPEED   # 1050 ms: intervallo tra decrementi di ossigeno
_T_PLY      = 120.0  / GAME_SPEED   # 120 ms: intervallo tra tick del giocatore
_DT_60FPS   = 1000.0 / 60.0         # 16.67 ms: delta-time a 60 FPS

# ── ID degli eventi pygame personalizzati ────────────────────────────────────
EVT_END_BLOCK = pygame.USEREVENT       # evento: blocco finale raggiunto (vittoria)
EVTICSEC      = pygame.USEREVENT + 1   # tick: scorre il tempo (ossigeno diminuisce)
EVTICPLY      = pygame.USEREVENT + 2   # tick: aggiorna lo stato del giocatore
EVTIC_GRAVITY = pygame.USEREVENT + 4   # tick: applica la gravità ai blocchi
EVTIC_ANIM    = pygame.USEREVENT + 5   # tick: aggiorna le animazioni

EPS_EVAL     = 0.0   # epsilon per la politica ε-greedy: 0 = sempre greedy

# Nomi leggibili delle 6 azioni possibili
ACTION_NAMES = ["IDLE", "LEFT", "RIGHT", "DRILL_L", "DRILL_R", "DRILL_D"]

# Etichette leggibili per i tre tipi di agente
AGENT_LABELS = {
    "dqn"    : "Rainbow DQN",
    "llm_act": f"LLM Agente ({LLM_MODEL_KEY})",
    "llm_rew": f"RL+LLM Reward ({LLM_MODEL_KEY})",
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. STRUTTURE DATI
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """
    Risultato di un singolo episodio per un agente.
    Raccoglie tutte le metriche rilevanti di fine partita.
    """
    agent:                   str    # nome dell'agente ("dqn", "llm_act", "llm_rew")
    episode:                 int    # indice dell'episodio (0-based)
    won:                     bool   # True se il giocatore ha raggiunto il fondo del livello
    depth:                   int    # profondità massima raggiunta (in unità di livello)
    steps:                   int    # numero di azioni eseguite nell'episodio
    final_oxy:               float  # ossigeno residuo alla fine dell'episodio (%)
    final_score:             int    # punteggio finale
    lives_left:              int    # vite rimaste alla fine
    death_cause:             str    # causa di fine episodio: "win"|"oxy"|"block_fall"|"timeout"|"stuck"
    duration_s:              float  = 0.0   # durata in secondi reali
    intended_action_counts:  list   = field(default_factory=lambda: [0]*N_ACTIONS)
    # intended: quante volte l'agente ha SCELTO ciascuna azione
    exec_action_counts:      list   = field(default_factory=lambda: [0]*N_ACTIONS)
    # exec: quante volte ciascuna azione è stata FISICAMENTE ESEGUITA


@dataclass
class ConfidenceInterval:
    """
    Intervallo di confidenza bootstrap al 95%.
    Contiene media stimata e limiti inferiore/superiore.
    """
    mean:  float   # stima puntuale della media
    lower: float   # limite inferiore dell'IC al 95%
    upper: float   # limite superiore dell'IC al 95%

    def __str__(self):
        return f"{self.mean:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


@dataclass
class AgentSummary:
    """
    Riepilogo statistico aggregato di tutti gli episodi per un agente.
    Contiene medie, deviazioni standard, IC bootstrap, distribuzioni azioni e LLM stats.
    """
    agent:                str    # nome dell'agente
    n_episodes:           int    # numero totale di episodi
    win_rate:             float  # frazione di episodi vinti [0–1]
    avg_depth:            float  # profondità media raggiunta
    avg_steps:            float  # numero medio di azioni per episodio
    avg_oxy:              float  # ossigeno medio a fine episodio
    avg_score:            float  # punteggio medio
    std_depth:            float = 0.0   # deviazione standard della profondità
    std_steps:            float = 0.0   # deviazione standard dei passi
    std_oxy:              float = 0.0   # deviazione standard dell'ossigeno
    std_score:            float = 0.0   # deviazione standard del punteggio
    pct_win:              float = 0.0   # % episodi terminati per vittoria
    pct_oxy:              float = 0.0   # % episodi terminati per ossigeno esaurito
    pct_block_fall:       float = 0.0   # % episodi terminati per blocco caduto sul giocatore
    pct_timeout:          float = 0.0   # % episodi terminati per timeout (MAX_STEPS raggiunto)
    pct_stuck:            float = 0.0   # % episodi terminati per stallo di posizione
    ci_win_rate:          dict  = field(default_factory=dict)
    ci_avg_depth:         dict  = field(default_factory=dict)
    ci_avg_steps:         dict  = field(default_factory=dict)
    intended_action_dist: list  = field(default_factory=lambda: [0.0]*N_ACTIONS)  # dist. azioni intese (%)
    exec_action_dist:     list  = field(default_factory=lambda: [0.0]*N_ACTIONS)  # dist. azioni eseguite (%)
    intended_entropy:     float = 0.0   # entropia di Shannon delle azioni intese (bit)
    exec_entropy:         float = 0.0   # entropia di Shannon delle azioni eseguite (bit)
    llm_stats:            dict  = field(default_factory=dict)   # statistiche LLM (latenza, cache, ecc.)


# ──────────────────────────────────────────────────────────────────────────────
# 2. UTILITY FISICA
# ──────────────────────────────────────────────────────────────────────────────

def applyGravity(level, player, surf):
    """
    Applica la gravità a tutti i blocchi della griglia.
    Un blocco cade se la cella sotto di lui è vuota (HP == 0).

    Meccanismo dettagliato:
      - Scansiona la griglia dal basso verso l'alto (per evitare che un blocco
        che sta cadendo venga processato più volte nello stesso tick)
      - Per i blocchi "classic": controlla se sono "incollati" a un blocco dello
        stesso colore adiacente (in tal caso non cadono)
      - Un blocco passa attraverso 3 stati: stazionario → shaking → falling
      - Se un blocco "falling" atterra sulla posizione del giocatore, lo uccide

    Parametri:
      level  : griglia 2D di blocchi (lista di liste)
      player : oggetto Character del giocatore
      surf   : superficie pygame

    Ritorna:
      True se almeno un blocco ha cambiato stato.
    """
    change_happened  = False
    rows, cols       = len(level), len(level[0])
    playerY, playerX = player.posAcc()

    for y in range(rows - 2, -1, -1):  # scansione dal basso verso l'alto
        for x in range(cols):

            blk      = level[y][x]
            blkBelow = level[y + 1][x]

            falling_types = ["classic", "solo", "delayed"]
            if blk.typeAccess() not in falling_types or blk.hpAccess() <= 0:
                continue

            if blkBelow.hpAccess() == 0:
                is_glued = False

                # Controlla l'incollamento solo per i blocchi "classic"
                if blk.typeAccess() == "classic":
                    myColor = blk.ColorAccess()

                    if (x > 0 and level[y][x-1].typeAccess() == "classic" and
                            level[y][x-1].ColorAccess() == myColor and
                            level[y][x-1].hpAccess() > 0):
                        is_glued = True

                    if (x < cols-1 and level[y][x+1].typeAccess() == "classic" and
                            level[y][x+1].ColorAccess() == myColor and
                            level[y][x+1].hpAccess() > 0):
                        is_glued = True

                    if (y > 0 and level[y-1][x].typeAccess() == "classic" and
                            level[y-1][x].ColorAccess() == myColor and
                            level[y-1][x].hpAccess() > 0):
                        is_glued = True

                if is_glued:
                    blk.stopFalling()
                    continue

                if blk.isFalling():
                    # Blocco già in caduta: scambia con la cella sotto
                    if (y + 1) == playerY and x == playerX:
                        player.revive(surf, level)  # toglie una vita al giocatore
                    level[y+1][x], level[y][x] = blk, blkBelow
                    blk.updatePos(x, y + 1)
                    blkBelow.updatePos(x, y)
                    change_happened = True

                elif blk.isShaking():
                    # Blocco che sta per cadere (fase di tremore)
                    if y == playerY and x == playerX:
                        player.revive(surf, level)
                    blk.tickShake()
                    change_happened = True

                else:
                    # Blocco stazionario con spazio sotto → inizia a tremare
                    blk.startShaking()
                    change_happened = True

            else:
                blk.stopFalling()

    if change_happened:
        for line in level:
            for el in line:
                if hasattr(el, "updCoText"):
                    el.updCoText(level)

    return change_happened


def check_won(player, level, won_flag: bool) -> bool:
    """
    Controlla se il giocatore ha vinto l'episodio.
    Usa una strategia "defense in depth": se won_flag è già True, ritorna True
    immediatamente; altrimenti tenta vari metodi dell'oggetto player e
    come ultima risorsa controlla se il giocatore è all'ultima riga.

    Parametri:
      player   : oggetto Character del giocatore
      level    : griglia 2D del livello
      won_flag : flag di vittoria precedentemente impostato (es. da EVT_END_BLOCK)

    Ritorna:
      True se il giocatore ha vinto, False altrimenti.
    """
    if won_flag:
        return True

    for m in ("wonAcc", "hasWon", "isWon", "checkWin"):
        fn = getattr(player, m, None)
        if callable(fn):
            try:
                if fn(): return True
            except Exception:
                pass

    pY, _ = player.posAcc()
    return pY >= len(level) - 1  # True se il giocatore è sceso fino all'ultima riga


# ──────────────────────────────────────────────────────────────────────────────
# 3. FRAME STACKER
# ──────────────────────────────────────────────────────────────────────────────

class SmartFrameStacker:
    """
    Accumula gli ultimi `stack_size` frame del gioco come tensori PyTorch.
    La rete DQN riceve sempre un pacchetto di N frame consecutivi (invece
    di un frame singolo) per percepire il movimento e la direzione dei blocchi.

    Funzionamento:
      - reset()     : inizializza la deque con `stack_size` copie dello stesso frame
      - push(frame) : aggiunge un nuovo frame (il più vecchio viene scartato automaticamente)
      - get_state() : concatena tutti i frame lungo l'asse dei canali e restituisce il tensore
    """

    def __init__(self, stack_size: int = 4, frame_shape: tuple = (6, 11, 9)):
        from collections import deque
        self.stack_size = stack_size
        # deque con maxlen: quando piena, elimina automaticamente il frame più vecchio
        self.frames     = deque(maxlen=stack_size)
        for _ in range(stack_size):
            self.frames.append(torch.zeros(frame_shape, device=device))

    def reset(self, init_frame):
        """Azzera lo stack e lo riempie con `stack_size` copie di init_frame."""
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(init_frame.clone())

    def push(self, f):
        """Aggiunge il frame `f` allo stack, scartando il più vecchio se pieno."""
        self.frames.append(f)

    def get_state(self):
        """
        Restituisce lo stato completo concatenando tutti i frame lungo il primo asse.
        Forma del risultato: (stack_size * canali, altezza, larghezza) = (24, 11, 9)
        """
        return torch.cat(list(self.frames), dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# 4. LOOP DI EPISODIO
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(
    agent_name:     str,
    policy_net      = None,   # rete DQN (usata se agent_name è "dqn" o "llm_rew")
    llm_agent_obj   = None,   # oggetto LLMAgent (usato se agent_name è "llm_act")
    surface         = None,   # superficie pygame virtuale
    ep_idx:         int           = 0,
    rng:            random.Random = None,
) -> EpisodeResult:
    """
    Esegue un singolo episodio headless del gioco Mr. Driller.

    Il loop replica esattamente il comportamento di main.py v6.9 in
    modalità headless, organizzato in 7 fasi per ogni iterazione:

    ① FASE ACTION     — se l'agente è pronto (ai_state=="ACTION" e player.IdlingAcc()),
                        seleziona ed esegue un'azione.
    ② Headless tick   — mentre aspetta (ai_state=="WAITING"), posta eventi pygame
                        direttamente (1 gravity + 3 ply + 1 sec ogni 3 tick).
    ③ Event loop      — processa tutti gli eventi: gravità, ossigeno, animazioni, vittoria.
    ④ backDown/climb  — gestisce la logica di scalata/discesa del livello.
    ⑤ WAITING→ACTION  — decide se è ora di tornare ad ACTION.
    ⑥ Stallo          — se la posizione non cambia per MAX_STUCK_STEPS → "stuck".
    ⑦ Terminazione    — esce dal loop se il giocatore muore, vince o finisce l'ossigeno.
    """
    if rng is None:
        rng = random.Random()

    t_start = time.time()
    pygame.event.clear()

    # ── Inizializzazione del gioco ─────────────────────────────────────────────
    player = Character(3, 4, 1, 2)
    level, levelID, won = restart(player)
    clear_cache()

    # ── Variabili di stato del gioco ───────────────────────────────────────────
    currentBotLine  = 8
    currentOffset   = 0
    currentClimb    = 0      # quante volte il giocatore ha "scalato" (variabile locale)
    backDown        = False
    blocksDisap: set = set()
    gravity_applied_since_action = False

    # ── Macchina a stati dell'agente ──────────────────────────────────────────
    ai_state        = "ACTION"   # stato iniziale: l'agente deve subito scegliere un'azione
    last_action_idx = 0

    # ── Variabili per il rilevamento di fallimenti consecutivi ─────────────────
    pt_prev_y = pt_prev_x = 0             # posizione PRIMA dell'azione
    pt_post_action_y = pt_post_action_x = 0  # posizione DOPO l'azione (prima della fisica)
    consec_fails = 0                       # contatore fallimenti consecutivi

    # ── Metriche dell'episodio ────────────────────────────────────────────────
    steps           = 0
    max_depth       = 0
    intended_counts = [0] * N_ACTIONS  # quante volte ogni azione è stata SCELTA
    exec_counts     = [0] * N_ACTIONS  # quante volte ogni azione è stata ESEGUITA

    # ── Rilevamento stallo di posizione ───────────────────────────────────────
    stuck_counter = 0
    last_pos      = None

    # ── Headless tick counter ─────────────────────────────────────────────────
    # Conta le iterazioni del loop mentre si aspetta che la fisica risolva un'azione
    _hl_tick = 0

    # ── Inizializzazione frame stacker DQN ───────────────────────────────────
    stacker = None
    if AI_AVAILABLE and policy_net is not None:
        stacker = SmartFrameStacker(4, (6, 11, 9))
        pY, pX  = player.posAcc()
        stacker.reset(get_local_window_tensor(level, pX, pY).to(device))

    # ── Import delle funzioni di controllo del gioco ─────────────────────────
    movKeys = [K_w, K_d, K_a, K_s]  # tasti: su, destra, sinistra, giù
    try:
        from eventHandling import movementHandle, breaking
    except ImportError:
        def movementHandle(*a): pass
        def breaking(*a): pass

    # ═══════════════════════════════════════════════════════════════════════════
    # LOOP PRINCIPALE
    # ═══════════════════════════════════════════════════════════════════════════
    while steps < MAX_STEPS:

        # ══════════════════════════════════════════════════════════════════════
        # ① FASE ACTION
        # L'agente sceglie ed esegue un'azione solo quando:
        #   - ai_state è "ACTION" (il sistema è pronto per una nuova azione)
        #   - player.IdlingAcc() è True (il giocatore non sta eseguendo animazioni)
        # ══════════════════════════════════════════════════════════════════════
        if ai_state == "ACTION" and player.IdlingAcc():
            pY, pX     = player.posAcc()
            pt_prev_y, pt_prev_x = pY, pX  # salva la posizione prima dell'azione

            # Aggiorna la profondità massima raggiunta
            cur_depth = currentOffset - currentClimb
            max_depth = max(max_depth, cur_depth)

            # ── Selezione dell'azione ──────────────────────────────────────────
            if agent_name == "llm_act" and llm_agent_obj is not None:
                # Modalità LLM: chiede all'LLM di scegliere l'azione
                # select_action ritorna (indice_azione, motivazione_testuale)
                intended_idx, _ = llm_agent_obj.select_action(
                    player, level, last_action_idx
                )

            elif AI_AVAILABLE and policy_net is not None and stacker is not None:
                # Modalità DQN: usa la rete neurale per scegliere l'azione
                grid    = stacker.get_state()  # tensore (24, 11, 9)
                vars_   = get_internal_state_vector(
                    player, len(level), len(level[0]), level, last_action_idx, N_ACTIONS
                )
                act_t, _ = select_action(policy_net, grid, vars_, EPS_EVAL, N_ACTIONS)
                intended_idx = act_t.item()

            else:
                # Fallback casuale: nessun modello disponibile
                intended_idx = rng.randint(0, N_ACTIONS - 1)

            # Registra l'azione che l'agente intende eseguire
            intended_counts[intended_idx] += 1

            # ── Esecuzione dell'azione ─────────────────────────────────────────
            # breaking() e movementHandle() vengono chiamati incondizionatamente,
            # esattamente come fa main.py, anche se l'azione non produce effetti.
            if intended_idx == 1:
                movementHandle(
                    pygame.event.Event(KEYDOWN, key=movKeys[2]),  # K_a = sinistra
                    surface, player, level, movKeys
                )
            elif intended_idx == 2:
                movementHandle(
                    pygame.event.Event(KEYDOWN, key=movKeys[1]),  # K_d = destra
                    surface, player, level, movKeys
                )
            elif intended_idx == 3:
                breaking(
                    pygame.event.Event(KEYDOWN, key=K_LEFT),
                    surface, player, level, currentBotLine
                )
            elif intended_idx == 4:
                breaking(
                    pygame.event.Event(KEYDOWN, key=K_RIGHT),
                    surface, player, level, currentBotLine
                )
            elif intended_idx == 5:
                breaking(
                    pygame.event.Event(KEYDOWN, key=K_DOWN),
                    surface, player, level, currentBotLine
                )
            # intended_idx == 0: IDLE → nessuna chiamata

            exec_counts[intended_idx] += 1

            # Cattura la posizione subito dopo l'azione, prima della fisica
            pt_post_action_y, pt_post_action_x = player.posAcc()

            last_action_idx              = intended_idx
            steps                       += 1
            gravity_applied_since_action = False
            ai_state                     = "WAITING"

        # ══════════════════════════════════════════════════════════════════════
        # ② HEADLESS TICK
        # Simula il passaggio del tempo postando eventi pygame nella coda.
        # Proporzioni timer di main.py:
        #   - 1 EVTIC_GRAVITY per tick (gravità dei blocchi)
        #   - 3 EVTICPLY per tick (aggiornamento stato giocatore)
        #   - 1 EVTICSEC ogni 3 tick (consumo ossigeno)
        # ══════════════════════════════════════════════════════════════════════
        if ai_state == "WAITING":
            _hl_tick += 1

            pygame.event.post(pygame.event.Event(EVTIC_GRAVITY))
            pygame.event.post(pygame.event.Event(EVTICPLY))
            pygame.event.post(pygame.event.Event(EVTICPLY))
            pygame.event.post(pygame.event.Event(EVTICPLY))

            if _hl_tick % 3 == 0:
                pygame.event.post(pygame.event.Event(EVTICSEC))

        # ══════════════════════════════════════════════════════════════════════
        # ③ EVENT LOOP
        # Processa tutti gli eventi nella coda pygame.
        # ══════════════════════════════════════════════════════════════════════
        for event in pygame.event.get():

            if event.type == EVT_END_BLOCK:
                won = True

            if event.type == EVTICSEC:
                # Diminuisce l'ossigeno del giocatore di 1 unità
                player.updateOxygen(1, surface, level)

                # Processa i blocchi "delayed" (scompaiono dopo un ritardo)
                dead_delayed: set = set()
                for (r, c) in blocksDisap:
                    if level[r][c].hpAccess() > 0:
                        if hasattr(level[r][c], "timeout") and level[r][c].timeout():
                            dead_delayed.add((r, c))
                    else:
                        dead_delayed.add((r, c))

                # Sostituisce i blocchi scaduti con celle vuote
                for (r, c) in dead_delayed:
                    if level[r][c].hpAccess() == 0:
                        nb = Classic(c, r, 1, 0)
                        nb._hp = 0
                        nb.changeBG(levelID)
                        level[r][c] = nb

                blocksDisap -= dead_delayed

                if dead_delayed:
                    applyGravity(level, player, surface)
                    player.fall(surface, level)

            if event.type == EVTIC_ANIM:
                player.Anim(surface)  # avanza le animazioni del giocatore

            if event.type == EVTIC_GRAVITY:
                applyGravity(level, player, surface)
                player.fall(surface, level)
                gravity_applied_since_action = True

                # Se il livello è scrollato, aggiorna l'offset e le posizioni
                if player.blocksFallenAcc() != currentOffset:
                    currentOffset  += 1
                    currentBotLine += 1
                    for row in level:
                        for el in row:
                            el.updOffset(currentOffset)

                max_depth = max(max_depth, currentOffset - currentClimb)

                # Push frame nello stacker ad ogni EVTIC_GRAVITY
                if stacker is not None:
                    gy, gx = player.posAcc()
                    gf = get_local_window_tensor(level, gx, gy).to(device)
                    stacker.push(gf)

            if event.type == EVTICPLY:
                player.NeedToIdle(surface)  # aggiorna il flag di idling

            # Aggiunge nuovi blocchi delayed attivi al set da monitorare
            if event.type in (EVTIC_GRAVITY, EVTICSEC):
                for r, row in enumerate(level):
                    for c, blk in enumerate(row):
                        if (blk.typeAccess() == "delayed" and
                                blk.idAcc() and blk.hpAccess() > 0):
                            blocksDisap.add((r, c))

        # ══════════════════════════════════════════════════════════════════════
        # ④ backDown / climb
        # Gestione della logica di scalata e discesa del livello.
        # ══════════════════════════════════════════════════════════════════════
        if backDown and player.climbAcc() < currentClimb:
            if currentClimb == 0:
                backDown = False
            player.backDownCleanup(surface)

        currentClimb = player.climbAcc()
        if player.climbAcc() > 0:
            backDown = True

        # ══════════════════════════════════════════════════════════════════════
        # ⑤ FASE WAITING → ACTION
        # Decide se è il momento di tornare allo stato ACTION.
        #
        # Condizioni di transizione (qualsiasi è sufficiente):
        #   A. is_terminal: il giocatore è morto, ha vinto o ha finito l'ossigeno
        #   B. idle_action: l'ultima azione era IDLE (non richiede attesa fisica)
        #   C. effective_fails >= MAX_CONSEC_FAILS: troppe azioni inutili → forza transizione
        #   D. gravity_applied AND player.IdlingAcc(): la fisica si è stabilizzata
        # ══════════════════════════════════════════════════════════════════════
        if ai_state == "WAITING":
            is_terminal = (player.livesAcc() < 0) or won or (player.oxyAcc() <= 0)

            # IDLE bypassa l'attesa: se l'agente non ha fatto nulla, non serve
            # aspettare che la fisica si stabilizzi
            idle_action = (last_action_idx == 0)

            # Confronta la posizione pre-azione con quella post-azione (prima della fisica)
            action_moved = (
                (pt_post_action_y != pt_prev_y) or
                (pt_post_action_x != pt_prev_x)
            )

            if last_action_idx != 0:
                effective_fails = consec_fails + (0 if action_moved else 1)
            else:
                effective_fails = 0

            ready = False

            if is_terminal or idle_action:
                ready        = True
                consec_fails = effective_fails

            elif effective_fails >= MAX_CONSEC_FAILS:
                # Troppi fallimenti consecutivi: forza la transizione
                if gravity_applied_since_action and player.IdlingAcc():
                    ready        = True
                    consec_fails = 0

            elif gravity_applied_since_action and player.IdlingAcc():
                # Caso normale: la gravità ha fatto il suo lavoro e il giocatore è a riposo
                ready        = True
                consec_fails = effective_fails

            if ready:
                # Se non c'è stata gravità, push un frame extra per mantenere lo stack aggiornato
                if stacker is not None and not gravity_applied_since_action:
                    new_pY, new_pX = player.posAcc()
                    ff = get_local_window_tensor(level, new_pX, new_pY).to(device)
                    stacker.push(ff)

                ai_state = "ACTION"

        # ══════════════════════════════════════════════════════════════════════
        # ⑥ Rilevamento stallo di posizione
        # Se il giocatore rimane alla stessa posizione per MAX_STUCK_STEPS
        # iterazioni consecutive, l'episodio viene terminato forzatamente.
        # ══════════════════════════════════════════════════════════════════════
        won     = check_won(player, level, won)
        cur_pos = player.posAcc()

        if cur_pos == last_pos:
            stuck_counter += 1
            if stuck_counter >= MAX_STUCK_STEPS:
                break
        else:
            stuck_counter = 0

        last_pos = cur_pos

        # ══════════════════════════════════════════════════════════════════════
        # ⑦ Terminazione normale
        # ══════════════════════════════════════════════════════════════════════
        if player.livesAcc() < 0 or won or player.oxyAcc() <= 0:
            break

    # ── Controlla eventi residui nella coda ───────────────────────────────────
    for event in pygame.event.get():
        if event.type == EVT_END_BLOCK:
            won = True

    won = check_won(player, level, won)

    # ── Determinazione della causa di fine episodio ───────────────────────────
    if won:
        death_cause = "win"
    elif player.oxyAcc() <= 0:
        death_cause = "oxy"
    elif stuck_counter >= MAX_STUCK_STEPS:
        death_cause = "stuck"
    elif steps >= MAX_STEPS:
        death_cause = "timeout"
    else:
        death_cause = "block_fall"

    return EpisodeResult(
        agent                   = agent_name,
        episode                 = ep_idx,
        won                     = won,
        depth                   = max_depth,
        steps                   = steps,
        final_oxy               = float(player.oxyAcc()),
        final_score             = int(player.scoreAcc()),
        lives_left              = max(0, int(player.livesAcc())),
        death_cause             = death_cause,
        duration_s              = round(time.time() - t_start, 2),
        intended_action_counts  = intended_counts,
        exec_action_counts      = exec_counts,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. STATISTICHE
# ──────────────────────────────────────────────────────────────────────────────

def _mean(lst):
    """Calcola la media aritmetica di una lista. Ritorna 0.0 se vuota."""
    return sum(lst) / len(lst) if lst else 0.0


def _std(lst):
    """
    Calcola la deviazione standard (popolazione) di una lista.
    Ritorna 0.0 se la lista ha meno di 2 elementi.
    """
    if len(lst) < 2: return 0.0
    m = _mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))


def _entropy(counts: list) -> float:
    """
    Calcola l'entropia di Shannon (in bit) di una distribuzione discreta.

    - Entropia = log2(N_ACTIONS) ≈ 2.585 bit: azioni perfettamente uniformi
    - Entropia = 0: l'agente sceglie sempre la stessa azione
    """
    total = sum(counts)
    if not total: return 0.0

    ent = 0.0
    for c in counts:
        if c > 0:
            p    = c / total
            ent -= p * math.log2(p)
    return round(ent, 4)


def bootstrap_ci(
    data:        list,
    stat_fn      = None,
    n_bootstrap: int   = 2000,
    alpha:       float = 0.05,
    seed:        int   = 42,
) -> ConfidenceInterval:
    """
    Calcola un intervallo di confidenza al (1-alpha)% tramite bootstrap.

    Il bootstrap:
      1. Campiona con rimpiazzo `n_bootstrap` volte dall'insieme di dati
      2. Calcola la statistica (es. media) su ogni campione
      3. Prende i percentili alpha/2 e (1-alpha/2) come limiti dell'IC
    """
    if stat_fn is None:
        stat_fn = _mean

    if not data:
        return ConfidenceInterval(0.0, 0.0, 0.0)

    rng  = random.Random(seed)
    n    = len(data)

    boot = sorted(
        stat_fn([rng.choice(data) for _ in range(n)])
        for _ in range(n_bootstrap)
    )

    lo = boot[int(alpha / 2 * n_bootstrap)]
    hi = boot[min(int((1 - alpha / 2) * n_bootstrap), n_bootstrap - 1)]

    return ConfidenceInterval(round(stat_fn(data), 4), round(lo, 4), round(hi, 4))


def compute_summary(
    agent_name:    str,
    results:       list,
    llm_stats_obj  = None,
    n_bootstrap:   int = 2000,
    seed:          int = 42,
) -> AgentSummary:
    """
    Aggrega i risultati di tutti gli episodi di un agente in un riepilogo statistico.
    Calcola: medie, deviazioni standard, IC bootstrap al 95%, distribuzioni
    delle azioni, entropie e statistiche LLM.
    """
    n = len(results)
    if n == 0:
        return AgentSummary(agent=agent_name, n_episodes=0,
                            win_rate=0, avg_depth=0, avg_steps=0,
                            avg_oxy=0, avg_score=0)

    depths  = [r.depth       for r in results]
    steps_  = [r.steps       for r in results]
    oxys    = [r.final_oxy   for r in results]
    scores  = [r.final_score for r in results]
    wins    = [1 if r.won else 0 for r in results]

    dc = Counter(r.death_cause for r in results)

    # Somma le distribuzioni di azioni su tutti gli episodi
    int_tot = [0] * N_ACTIONS
    exc_tot = [0] * N_ACTIONS
    for r in results:
        for i in range(N_ACTIONS):
            int_tot[i] += r.intended_action_counts[i]
            exc_tot[i] += r.exec_action_counts[i]

    int_sum = max(sum(int_tot), 1)
    exc_sum = max(sum(exc_tot), 1)

    ci_w = bootstrap_ci(wins,   n_bootstrap=n_bootstrap, seed=seed)
    ci_d = bootstrap_ci(depths, n_bootstrap=n_bootstrap, seed=seed)
    ci_s = bootstrap_ci(steps_, n_bootstrap=n_bootstrap, seed=seed)

    return AgentSummary(
        agent                = agent_name,
        n_episodes           = n,
        win_rate             = round(_mean(wins),   4),
        avg_depth            = round(_mean(depths), 2),
        avg_steps            = round(_mean(steps_), 1),
        avg_oxy              = round(_mean(oxys),   1),
        avg_score            = round(_mean(scores), 1),
        std_depth            = round(_std(depths),  2),
        std_steps            = round(_std(steps_),  1),
        std_oxy              = round(_std(oxys),    1),
        std_score            = round(_std(scores),  1),
        pct_win              = round(dc.get("win",        0) / n * 100, 1),
        pct_oxy              = round(dc.get("oxy",        0) / n * 100, 1),
        pct_block_fall       = round(dc.get("block_fall", 0) / n * 100, 1),
        pct_timeout          = round(dc.get("timeout",    0) / n * 100, 1),
        pct_stuck            = round(dc.get("stuck",      0) / n * 100, 1),
        ci_win_rate          = {"mean": ci_w.mean,  "lower": ci_w.lower,  "upper": ci_w.upper},
        ci_avg_depth         = {"mean": ci_d.mean,  "lower": ci_d.lower,  "upper": ci_d.upper},
        ci_avg_steps         = {"mean": ci_s.mean,  "lower": ci_s.lower,  "upper": ci_s.upper},
        intended_action_dist = [round(int_tot[i] / int_sum * 100, 2) for i in range(N_ACTIONS)],
        exec_action_dist     = [round(exc_tot[i] / exc_sum * 100, 2) for i in range(N_ACTIONS)],
        intended_entropy     = _entropy(int_tot),
        exec_entropy         = _entropy(exc_tot),
        # Stats LLM: estratte dall'oggetto se disponibile, altrimenti dict vuoto
        llm_stats            = llm_stats_obj.stats() if llm_stats_obj is not None else {},
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6. OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def print_table(summaries: list):
    """
    Stampa sul terminale la tabella comparativa completa di tutti gli agenti.
    Struttura:
      ① Statistiche principali (win rate, profondità, passi, ossigeno, punteggio)
      ② Intervalli di confidenza al 95% (bootstrap)
      ③ Cause di fine episodio (%)
      ④ Distribuzione azioni intended vs executed + entropia
      ⑤ Statistiche LLM (solo se disponibili)
    """
    W   = 112
    sep = "=" * W

    print(f"\n{sep}")
    print("  VALUTAZIONE COMPARATIVA — Mr. Driller Agent Benchmark")
    print(sep)

    # ── ① Statistiche principali ──────────────────────────────────────────────
    fmt = "{:<22} {:>7} {:>9} {:>16} {:>16} {:>8} {:>10}"
    print(fmt.format("Agente", "Ep", "Win%", "Depth avg±std", "Steps avg±std", "Oxy%", "Score"))
    print("-" * W)
    for s in summaries:
        lbl = AGENT_LABELS.get(s.agent, s.agent)
        print(fmt.format(
            lbl, s.n_episodes,
            f"{s.win_rate*100:.1f}%",
            f"{s.avg_depth:.1f}±{s.std_depth:.1f}",
            f"{s.avg_steps:.1f}±{s.std_steps:.1f}",
            f"{s.avg_oxy:.1f}",
            f"{s.avg_score:.0f}",
        ))
    print(sep)

    # ── ② Intervalli di confidenza al 95% ─────────────────────────────────────
    print("\n  INTERVALLI DI CONFIDENZA 95%  (bootstrap, 2000 campioni)")
    print("-" * W)
    cf = "{:<22} {:>28} {:>26} {:>26}"
    print(cf.format("Agente", "Win Rate CI", "Depth CI", "Steps CI"))
    print("-" * W)
    for s in summaries:
        lbl = AGENT_LABELS.get(s.agent, s.agent)
        w, d, t = s.ci_win_rate, s.ci_avg_depth, s.ci_avg_steps
        print(cf.format(
            lbl,
            f"{w['mean']*100:.1f}% [{w['lower']*100:.1f}, {w['upper']*100:.1f}]",
            f"{d['mean']:.1f}  [{d['lower']:.1f}, {d['upper']:.1f}]",
            f"{t['mean']:.1f}  [{t['lower']:.1f}, {t['upper']:.1f}]",
        ))
    print(sep)

    # ── ③ Cause di fine episodio ──────────────────────────────────────────────
    print("\n  CAUSE DI FINE EPISODIO  (%)")
    print("-" * W)
    df = "{:<22} {:>7} {:>9} {:>13} {:>9} {:>8}"
    print(df.format("Agente", "Win", "Ossigeno", "Blocco caduto", "Timeout", "Stuck"))
    print("-" * W)
    for s in summaries:
        lbl = AGENT_LABELS.get(s.agent, s.agent)
        print(df.format(
            lbl,
            f"{s.pct_win:.1f}%",
            f"{s.pct_oxy:.1f}%",
            f"{s.pct_block_fall:.1f}%",
            f"{s.pct_timeout:.1f}%",
            f"{s.pct_stuck:.1f}%",
        ))
    print(sep)

    # ── ④ Distribuzione azioni ────────────────────────────────────────────────
    print("\n  DISTRIBUZIONE AZIONI")
    print("  INTENDED = scelta dell'agente  |  EXECUTED = fisicamente tentata")
    ah = "{:<24}" + (" {:>9}" * N_ACTIONS) + "  {:>10}"
    print(ah.format("", *ACTION_NAMES, "Entr.(bit)"))
    print("-" * W)
    for s in summaries:
        lbl = AGENT_LABELS.get(s.agent, s.agent)
        iv  = [f"{v:.1f}%" for v in s.intended_action_dist]
        ev  = [f"({v:.1f}%)" for v in s.exec_action_dist]
        print(f"  {lbl} — INTENDED")
        print(ah.format("", *iv, f"{s.intended_entropy:.4f}"))
        print(f"  {lbl} — EXECUTED")
        print(ah.format("", *ev, f"{s.exec_entropy:.4f}"))
    print(sep)

    # ── ⑤ Statistiche LLM (solo per agenti che usano LLM) ────────────────────
    llm_rows = [s for s in summaries if s.llm_stats]
    if llm_rows:
        print(f"\n  STATISTICHE LLM  ({LLM_MODEL_KEY} via OpenRouter)")
        print("-" * W)
        for s in llm_rows:
            lbl = AGENT_LABELS.get(s.agent, s.agent)
            ls  = s.llm_stats
            print(f"  {lbl}:")
            print(f"    Chiamate API : {ls.get('call_count', 0)}")
            print(f"    Cache hits   : {ls.get('cache_hits', 0)}"
                  f"  ({ls.get('cache_hit_rate', 0)*100:.1f}%)")
            print(f"    Errori       : {ls.get('error_count', 0)}"
                  f"  ({ls.get('error_rate', 0)*100:.1f}%)")
            print(f"    Lat. media   : {ls.get('avg_latency_ms', 0):.1f} ms")
            print(f"    Consistenza  : {ls.get('consistency', 0)*100:.1f}%")
        print(sep)

    print()


def save_csv(all_results: list, out_dir: str, ts: str) -> str:
    """
    Salva tutti i risultati episodio per episodio in un file CSV.

    Struttura del CSV:
      - Una riga per ogni episodio di ogni agente
      - Colonne di base: agent, episode, won, depth, steps, ...
      - Colonne azioni intended: int_IDLE, int_LEFT, ...
      - Colonne azioni executed: exc_IDLE, exc_LEFT, ...
    """
    os.makedirs(out_dir, exist_ok=True)
    path_ = os.path.join(out_dir, f"results_{ts}.csv")

    base  = ["agent", "episode", "won", "depth", "steps", "final_oxy",
             "final_score", "lives_left", "death_cause", "duration_s"]

    int_f = [f"int_{n}" for n in ACTION_NAMES]
    exc_f = [f"exc_{n}" for n in ACTION_NAMES]

    with open(path_, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base + int_f + exc_f)
        w.writeheader()

        for r in all_results:
            row = {k: getattr(r, k) for k in base}
            for i, n in enumerate(ACTION_NAMES):
                row[f"int_{n}"] = r.intended_action_counts[i]
                row[f"exc_{n}"] = r.exec_action_counts[i]
            w.writerow(row)

    print(f"[EVAL] CSV → {path_}")
    return path_


def save_json(summaries: list, out_dir: str, ts: str) -> str:
    """
    Salva il riepilogo statistico di tutti gli agenti in un file JSON.
    Il JSON contiene una lista di dizionari, uno per agente,
    con tutte le metriche aggregate.
    """
    os.makedirs(out_dir, exist_ok=True)
    path_ = os.path.join(out_dir, f"summary_{ts}.json")

    with open(path_, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2, ensure_ascii=False)

    print(f"[EVAL] JSON → {path_}")
    return path_


def save_plot(summaries: list, out_dir: str, ts: str) -> Optional[str]:
    """
    Genera e salva un grafico PNG con 6 subplot comparativi:
      ① Win rate con IC al 95%
      ② Profondità media ± std
      ③ Passi medi ± std
      ④ Cause di morte (stacked bar)
      ⑤ Distribuzione azioni intended vs executed
      ⑥ Entropia delle azioni

    Richiede matplotlib. Se non installato, stampa un avviso e ritorna None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # backend non interattivo (non apre finestre)
        import matplotlib.pyplot as plt

        labels = [AGENT_LABELS.get(s.agent, s.agent) for s in summaries]
        n      = len(summaries)
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"][:n]
        x      = np.arange(n)
        bw     = 0.55

        fig, axes = plt.subplots(2, 3, figsize=(20, 11))
        fig.suptitle(
            "Mr. Driller — Agent Benchmark\n"
            "breaking() incondizionato  |  loop headless hl_tick  |  "
            "push frame in EVTIC_GRAVITY  |  MAX_CONSEC_FAILS=4",
            fontsize=11, fontweight="bold"
        )

        # ── ① Win Rate con IC bootstrap ───────────────────────────────────────
        ax  = axes[0, 0]
        wr  = [s.win_rate * 100 for s in summaries]
        lo  = [s.win_rate*100 - s.ci_win_rate["lower"]*100 for s in summaries]
        hi  = [s.ci_win_rate["upper"]*100 - s.win_rate*100 for s in summaries]
        ax.bar(x, wr, color=colors, width=bw, zorder=2)
        ax.errorbar(x, wr, yerr=[lo, hi], fmt="none", color="black",
                    capsize=7, lw=1.5, zorder=3)
        ax.set_title("Win Rate %  (95% CI bootstrap)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=13, ha="right")
        ax.set_ylim(0, 118)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(wr):
            ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

        # ── ② Profondità media ± std ──────────────────────────────────────────
        ax = axes[0, 1]
        ax.bar(x, [s.avg_depth for s in summaries],
               yerr=[s.std_depth for s in summaries],
               color=colors, width=bw, capsize=5, zorder=2)
        ax.set_title("Profondità Media ± Std")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=13, ha="right")
        ax.grid(axis="y", alpha=0.3)

        # ── ③ Passi medi ± std ────────────────────────────────────────────────
        ax = axes[0, 2]
        ax.bar(x, [s.avg_steps for s in summaries],
               yerr=[s.std_steps for s in summaries],
               color=colors, width=bw, capsize=5, zorder=2)
        ax.set_title("Passi Medi ± Std (azioni)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=13, ha="right")
        ax.grid(axis="y", alpha=0.3)

        # ── ④ Cause di morte (stacked bar) ────────────────────────────────────
        ax  = axes[1, 0]
        bot = [0.0] * n
        for vals, lbl, col in [
            ([s.pct_win        for s in summaries], "Win",          "#4CAF50"),
            ([s.pct_oxy        for s in summaries], "Ossigeno",     "#FF5722"),
            ([s.pct_block_fall for s in summaries], "Blocco",       "#9C27B0"),
            ([s.pct_timeout    for s in summaries], "Timeout",      "#607D8B"),
            ([s.pct_stuck      for s in summaries], "Stuck",        "#FF9800"),
        ]:
            ax.bar(x, vals, width=bw, label=lbl, color=col, bottom=bot)
            bot = [bot[i] + vals[i] for i in range(n)]
        ax.set_title("Cause di fine episodio (%)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=13, ha="right")
        ax.legend(fontsize=8); ax.set_ylim(0, 118)

        # ── ⑤ Distribuzione azioni intended vs executed ───────────────────────
        ax        = axes[1, 1]
        act_colors = ["#9E9E9E","#2196F3","#03A9F4","#4CAF50","#8BC34A","#CDDC39"]
        bw_a      = 0.07

        for j in range(N_ACTIONS):
            iv   = [s.intended_action_dist[j] for s in summaries]
            ev   = [s.exec_action_dist[j]     for s in summaries]
            off_i = (j - N_ACTIONS / 2) * bw_a * 2.1
            off_e = off_i + bw_a
            ax.bar(x + off_i, iv, width=bw_a,
                   color=act_colors[j], label=ACTION_NAMES[j])
            ax.bar(x + off_e, ev, width=bw_a,
                   color=act_colors[j], alpha=0.4, hatch="//")
        ax.set_title("Azioni: Intended (pieno) vs Executed (tratt.)")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=13, ha="right")
        ax.legend(fontsize=6, ncol=3)

        # ── ⑥ Entropia delle azioni ───────────────────────────────────────────
        ax      = axes[1, 2]
        int_ent = [s.intended_entropy for s in summaries]
        exc_ent = [s.exec_entropy     for s in summaries]
        bw_e    = 0.28
        ax.bar(x - bw_e/2, int_ent, width=bw_e, color=colors, label="Intended")
        ax.bar(x + bw_e/2, exc_ent, width=bw_e, color=colors, alpha=0.45,
               hatch="//", label="Executed")

        max_e = math.log2(N_ACTIONS)  # ~2.585 bit per 6 azioni
        ax.axhline(max_e, color="red", ls="--", lw=0.8,
                   label=f"Max {max_e:.2f} bit")
        ax.set_title("Entropia azioni (Shannon, bit)")
        ax.set_ylabel("bit")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=13, ha="right")
        ax.legend(fontsize=8)

        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        path_ = os.path.join(out_dir, f"comparison_{ts}.png")
        plt.savefig(path_, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[EVAL] Plot → {path_}")
        return path_

    except ImportError:
        print("[EVAL] matplotlib non disponibile — plot saltato.")
        return None
    except Exception as e:
        print(f"[EVAL] Errore plot: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Funzione principale: gestisce argomenti CLI, caricamento modelli,
    esecuzione episodi per ogni agente, raccolta statistiche e output.
    """
    global MAX_STUCK_STEPS

    parser = argparse.ArgumentParser(
        description="Mr. Driller — Agent Benchmark"
    )
    parser.add_argument("--episodes",     type=int,  default=10)
    parser.add_argument("--agents",       nargs="+", default=["dqn", "llm_act", "llm_rew"],
                        choices=["dqn", "llm_act", "llm_rew"])
    parser.add_argument("--out_dir",      type=str,  default="eval_results")
    parser.add_argument("--seed",         type=int,  default=42)
    parser.add_argument("--bootstrap",    type=int,  default=2000)
    parser.add_argument("--dqn_path",     type=str,  default=MODEL_PATH)
    parser.add_argument("--llm_rew_path", type=str,  default=None)
    parser.add_argument("--llm_model",    type=str,  default=LLM_MODEL_KEY,
                        choices=["llama", "deepseek"])
    parser.add_argument("--stuck_steps",  type=int,  default=MAX_STUCK_STEPS,
                        help="Iterazioni senza cambio posizione → terminazione stuck")
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args()

    # ── Impostazione dei seed per riproducibilità ─────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    if AI_AVAILABLE:
        torch.manual_seed(args.seed)

    MAX_STUCK_STEPS = args.stuck_steps

    # ── Inizializzazione pygame ───────────────────────────────────────────────
    pygame.init()
    # Crea una superficie virtuale 800×600 (nessuna finestra reale)
    surface = pygame.Surface((800, 600))

    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rng     = random.Random(args.seed)

    # ── Caricamento dei modelli ───────────────────────────────────────────────
    policy_dqn     = None
    policy_llm_rew = None
    llm_agent_obj  = None

    if "dqn" in args.agents or "llm_rew" in args.agents:
        if AI_AVAILABLE:

            if "dqn" in args.agents:
                print(f"[EVAL] Caricamento DQN da {args.dqn_path} ...")
                policy_dqn = DrillerDuelingDQN(INPUT_SHAPE, N_ACTIONS, STATE_DIM).to(device)
                policy_dqn.eval()  # modalità evaluation (disabilita dropout)

                if os.path.exists(args.dqn_path):
                    ckpt = torch.load(args.dqn_path, map_location=device)
                    policy_dqn.load_state_dict(ckpt["model_state_dict"], strict=True)
                    print(f"[EVAL] DQN caricato  (step {ckpt.get('steps', '?')})")
                else:
                    print(f"[EVAL] ATTENZIONE: {args.dqn_path} non trovato — pesi casuali!")

            if "llm_rew" in args.agents:
                llm_rew_path = (args.llm_rew_path or
                                args.dqn_path.replace(".pth", "_llmrew.pth"))

                if not os.path.exists(llm_rew_path):
                    llm_rew_path = args.dqn_path
                    print(f"[EVAL] llm_rew_path non trovato — uso {llm_rew_path}")

                print(f"[EVAL] Caricamento DQN+LLMRew da {llm_rew_path} ...")
                policy_llm_rew = DrillerDuelingDQN(INPUT_SHAPE, N_ACTIONS, STATE_DIM).to(device)
                policy_llm_rew.eval()

                if os.path.exists(llm_rew_path):
                    ckpt = torch.load(llm_rew_path, map_location=device)
                    policy_llm_rew.load_state_dict(ckpt["model_state_dict"], strict=True)
                    print(f"[EVAL] DQN+LLMRew caricato  (step {ckpt.get('steps', '?')})")
                else:
                    print("[EVAL] ATTENZIONE: checkpoint llm_rew assente — pesi casuali!")
        else:
            args.agents = [a for a in args.agents if a not in ("dqn", "llm_rew")]

    if "llm_act" in args.agents:
        if LLM_AGENT_AVAILABLE:
            print(f"[EVAL] Inizializzazione LLMAgent (model={args.llm_model}) ...")
            try:
                llm_agent_obj = LLMAgent(model_key=args.llm_model)
                print(f"[EVAL] LLMAgent pronto")
            except Exception as e:
                print(f"[EVAL] LLMAgent init fallita: {e}  — llm_act saltato.")
                args.agents = [a for a in args.agents if a != "llm_act"]
        else:
            args.agents = [a for a in args.agents if a != "llm_act"]

    if not args.agents:
        print("[EVAL] Nessun agente disponibile. Uscita.")
        pygame.quit()
        return

    print()
    print(f"[EVAL] ══════════════════════════════════════════════════════════")
    print(f"[EVAL]  Seed={args.seed} | Bootstrap={args.bootstrap} | "
          f"Episodi={args.episodes} | LLM={args.llm_model}")
    print(f"[EVAL]  GAME_SPEED={GAME_SPEED} | MAX_CONSEC_FAILS={MAX_CONSEC_FAILS}")
    print(f"[EVAL]  Stuck-timeout dopo {MAX_STUCK_STEPS} iter invariate")
    print(f"[EVAL] ══════════════════════════════════════════════════════════")
    print()

    all_results: list = []
    summaries:   list = []

    # ── Loop principale: per ogni agente, esegue tutti gli episodi ────────────
    for agent_name in args.agents:
        lbl = AGENT_LABELS.get(agent_name, agent_name)

        # Seleziona la rete corretta per questo agente
        policy  = (policy_dqn     if agent_name == "dqn"     else
                   policy_llm_rew if agent_name == "llm_rew" else None)

        llm_obj = llm_agent_obj if agent_name == "llm_act" else None

        print(f"[EVAL] ── {lbl}  ({args.episodes} episodi) ──")
        agent_results = []

        for ep in range(args.episodes):
            if args.verbose:
                print(f"[EVAL]   ep {ep+1:3d}/{args.episodes} ...", end=" ", flush=True)

            res = run_episode(
                agent_name    = agent_name,
                policy_net    = policy,
                llm_agent_obj = llm_obj,
                surface       = surface,
                ep_idx        = ep,
                rng           = rng,
            )
            agent_results.append(res)

            if args.verbose:
                tag = "WIN " if res.won else f"DEAD({res.death_cause[:4]})"
                it  = ACTION_NAMES[res.intended_action_counts.index(
                    max(res.intended_action_counts))]
                ex  = ACTION_NAMES[res.exec_action_counts.index(
                    max(res.exec_action_counts))] if any(res.exec_action_counts) else "—"
                print(f"{tag} | d={res.depth:3d} | s={res.steps:4d} | "
                      f"oxy={res.final_oxy:5.1f}% | "
                      f"int={it} exc={ex} | {res.duration_s:.1f}s")

        llm_src = llm_agent_obj if agent_name == "llm_act" else None
        summ    = compute_summary(
            agent_name,
            agent_results,
            llm_stats_obj  = llm_src,
            n_bootstrap    = args.bootstrap,
            seed           = args.seed,
        )
        summaries.append(summ)
        all_results.extend(agent_results)

        ci = summ.ci_win_rate
        print(f"[EVAL]   → Win {summ.win_rate*100:.1f}%"
              f"  CI [{ci['lower']*100:.1f}, {ci['upper']*100:.1f}]"
              f"  depth {summ.avg_depth:.1f}±{summ.std_depth:.1f}"
              f"  steps {summ.avg_steps:.1f}±{summ.std_steps:.1f}"
              f"  int_H={summ.intended_entropy:.3f}bit"
              f"  exc_H={summ.exec_entropy:.3f}bit")
        print()

    print_table(summaries)

    if all_results:
        save_csv(all_results, args.out_dir, ts)
        save_json(summaries,  args.out_dir, ts)
        save_plot(summaries,  args.out_dir, ts)

    if llm_agent_obj is not None:
        print(f"\n[LLM_AGENT] Stats finali: {llm_agent_obj.stats()}")

    pygame.quit()


if __name__ == "__main__":
    main()