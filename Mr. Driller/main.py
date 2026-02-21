"""
main.py — Mr. Driller AI Training Loop (Rainbow DQN) v5.4 — Reward Rebalancing

CHANGELOG v5.4:
  LEARNING_RATE: 3e-4 -> 1.5e-4
  TARGET_UPDATE: 300  -> 200
  TRAIN_EVERY:   4    -> 3
  N_STEP:        5    (invariato)

AGGIUNTO v5.4-PER-PERSIST:
  BUFFER_PATH: percorso file npz per la persistenza del replay buffer PER.
  Al riavvio: memory.load(BUFFER_PATH) ripristina tutte le transizioni e
  le loro priorità originali. Niente più "lobotomia" ad ogni sessione Colab.

  Salvataggio buffer:
    - Ogni BUFFER_SAVE_EVERY step (default 5000) → asincrono, non blocca
    - All'uscita (QUIT) → asincrono
    - Al termine del training (levelID >= MAX_LEVEL) → asincrono

IPERPARAMETRO PRINCIPALE:
  PLAY_MODE = True  → carica il modello e gioca (con display, nessun training)
  PLAY_MODE = False → training headless (senza display)
"""

import os, sys, math, threading
from os import path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"]  = "True"
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"

# ── UNICO IPERPARAMETRO DA CAMBIARE ──────────────────────────────────────────
PLAY_MODE = False   # True = gioca col modello (display ON)
                   # False = training headless (display OFF)
# ─────────────────────────────────────────────────────────────────────────────

HEADLESS_TRAINING = not PLAY_MODE   # derivato automaticamente da PLAY_MODE

if HEADLESS_TRAINING:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"
    print("[INIT] Modalita HEADLESS (no finestra) — TRAINING MODE")
else:
    print("[INIT] Modalita DISPLAY (finestra visibile) — PLAY MODE")

import numpy as np
try:
    import torch
    import torch.optim as optim
except ImportError:
    pass

import pygame
from pygame.locals import *

from character import Character
from block import Classic, Pill, Solo, Delayed, clear_cache
from menu import changeLvl, restart, mainMenu

try:
    from ai_agent import (
        DrillerDuelingDQN, PrioritizedReplayMemory as ReplayMemory,
        NStepBuffer, RewardShaper, DrillTracker,
        select_action, optimize_model, calculate_reward, device,
        get_local_window_tensor, get_internal_state_vector
    )
    AI_AVAILABLE = True
    print("[INIT] AI caricata correttamente")
except ImportError as e:
    print(f"[INIT] AI non disponibile: {e}")
    AI_AVAILABLE = False

# ── Iperparametri v5.4 ────────────────────────────────────────────────────────
BATCH_SIZE    = 128
GAMMA         = 0.99
EPS_START     = 0.30
EPS_END       = 0.02
EPS_DECAY     = 40_000
TARGET_UPDATE = 200
MEMORY_SIZE   = 50_000
LEARNING_RATE = 1.5e-4
N_STEP        = 5
MODEL_PATH    = "RenfLearningAgent/driller_ai_v15.pth"
MAX_LEVEL     = 1
N_CHANNELS    = 6
N_FRAMES      = 4
INPUT_SHAPE   = (N_CHANNELS * N_FRAMES, 11, 9)
STATE_DIM     = 32
TRAIN_EVERY   = 3

# ── Persistenza buffer PER ────────────────────────────────────────────────────
BUFFER_PATH       = "RenfLearningAgent/driller_buffer_v15.npz"
BUFFER_SAVE_EVERY = 5000
# ─────────────────────────────────────────────────────────────────────────────

sys.setrecursionlimit(5000)
GAME_W, UI_W       = 600, 200
SCREEN_W, SCREEN_H = GAME_W + UI_W, 600


def save_checkpoint(model, optimizer, steps, epsilon, scheduler=None):
    ckpt = {"model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "steps": steps, "epsilon": epsilon}
    if scheduler:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    def _write():
        torch.save(ckpt, MODEL_PATH)
        print(f"[AI] Checkpoint salvato — step {steps}, e {epsilon:.4f}")
    threading.Thread(target=_write, daemon=True).start()


def load_checkpoint(model, optimizer, scheduler=None):
    start_steps, start_eps = 0, EPS_START
    if os.path.exists(MODEL_PATH):
        print(f"[AI] Caricamento da {MODEL_PATH}...")
        try:
            ckpt = torch.load(MODEL_PATH, map_location=device)
            try:
                model.load_state_dict(ckpt["model_state_dict"], strict=True)
            except RuntimeError:
                print("[AI] Architettura cambiata, riparto da zero")
                return 0, EPS_START
            if optimizer:
                try: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception: pass
            if scheduler and "scheduler_state_dict" in ckpt:
                try: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception: pass
            start_steps = ckpt.get("steps", 0)
            start_eps   = ckpt.get("epsilon", EPS_START)
            print(f"[AI] Caricato! Step: {start_steps}, e: {start_eps:.4f}")
        except Exception as e:
            print(f"[AI] Errore: {e}, riparto da zero")
    else:
        print("[AI] Nessun salvataggio, pesi casuali")
    return start_steps, start_eps


def load_oxy_frames():
    frames, base_path = {}, path.join("Assets", "Misc", "oxyAnim")
    for i in range(150):
        p = path.join(base_path, f"{i}.png")
        frames[i] = pygame.image.load(p).convert_alpha() if path.exists(p) else None
    return frames


def draw_ai_debug(surface, debug_font, epsilon, steps_done, last_action_idx, ai_state):
    overlay = pygame.Surface((260, 140), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    surface.blit(overlay, (5, 5))
    action_names = ["IDLE", "LEFT", "RIGHT", "DRILL_L", "DRILL_R", "DRILL_D"]
    act_label = action_names[last_action_idx] if last_action_idx < len(action_names) else str(last_action_idx)
    infos = [
        (f"EPS:   {epsilon:.4f}", (0, 255, 0)),
        (f"STEP:  {steps_done}",   (0, 255, 255)),
        (f"ACT:   {act_label}",    (255, 255, 0)),
        (f"MODE:  {'PLAY' if PLAY_MODE else 'TRAIN'}", (100, 200, 255)),
        (f"PHASE: {ai_state}", (255, 100, 100) if ai_state != "ACTION" else (100, 255, 100)),
    ]
    for i, (txt, col) in enumerate(infos):
        surface.blit(debug_font.render(txt, True, col), (10, 10 + i * 25))


def draw_gameover(surface, font, won_game=False):
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    surface.blit(overlay, (0, 0))
    msg1 = "MISSION COMPLETE!" if won_game else "GAME OVER"
    color = (255, 215, 0) if won_game else (255, 50, 50)
    s1 = font.render(msg1, True, color)
    s2 = font.render("Press R to Restart", True, (200, 200, 200))
    s3 = font.render("Press ESC to Quit",  True, (150, 150, 150))
    cx, cy = SCREEN_W // 2, SCREEN_H // 2
    surface.blit(s1, s1.get_rect(center=(cx, cy - 50)))
    surface.blit(s2, s2.get_rect(center=(cx, cy + 20)))
    surface.blit(s3, s3.get_rect(center=(cx, cy + 60)))


def applyGravity(level, player, surf):
    change_happened  = False
    rows, cols       = len(level), len(level[0])
    playerY, playerX = player.posAcc()
    for y in range(rows - 2, -1, -1):
        for x in range(cols):
            blk      = level[y][x]
            blkBelow = level[y + 1][x]
            falling_types = ["classic", "solo", "delayed"]
            if blk.typeAccess() not in falling_types or blk.hpAccess() <= 0:
                continue
            if blkBelow.hpAccess() == 0:
                is_glued = False
                if blk.typeAccess() == "classic":
                    myColor = blk.ColorAccess()
                    if (x > 0 and level[y][x-1].typeAccess() == "classic" and
                            level[y][x-1].ColorAccess() == myColor and level[y][x-1].hpAccess() > 0):
                        is_glued = True
                    if (x < cols-1 and level[y][x+1].typeAccess() == "classic" and
                            level[y][x+1].ColorAccess() == myColor and level[y][x+1].hpAccess() > 0):
                        is_glued = True
                    if (y > 0 and level[y-1][x].typeAccess() == "classic" and
                            level[y-1][x].ColorAccess() == myColor and level[y-1][x].hpAccess() > 0):
                        is_glued = True
                if is_glued:
                    blk.stopFalling(); continue
                if blk.isFalling():
                    if (y + 1) == playerY and x == playerX:
                        player.revive(surf, level)
                    else:
                        level[y+1][x], level[y][x] = blk, blkBelow
                        blk.updatePos(x, y + 1); blkBelow.updatePos(x, y)
                        change_happened = True
                elif blk.isShaking():
                    blk.tickShake(); change_happened = True
                else:
                    blk.startShaking(); change_happened = True
            else:
                blk.stopFalling()
    if change_happened:
        for line in level:
            for el in line:
                if hasattr(el, "updCoText"): el.updCoText(level)
    return change_happened


class SmartFrameStacker:
    def __init__(self, stack_size=4, frame_shape=(6, 11, 9), device="cpu"):
        self.stack_size = stack_size
        self.device     = device
        from collections import deque
        self.frames = deque(maxlen=stack_size)
        self._reset_frames(frame_shape)

    def _reset_frames(self, shape):
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(torch.zeros(shape, device=self.device))

    def reset(self, initial_frame):
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(initial_frame.clone())

    def push(self, frame): self.frames.append(frame)

    def get_state(self): return torch.cat(list(self.frames), dim=0)


def _reset_ai_episode(player, level, stacker, nstep_buf, shaper, drill_tracker, memory=None):
    pY, pX     = player.posAcc()
    init_frame = get_local_window_tensor(level, pX, pY).to(device)
    stacker.reset(init_frame)
    if nstep_buf is not None:
        if memory is not None:
            for trans in nstep_buf.drain(): memory.push(*trans)
        else:
            nstep_buf.flush()
    shaper.reset(pY, player.oxyAcc(), len(level))
    drill_tracker.reset()


def game(screen_width=SCREEN_W, screen_height=SCREEN_H):

    TRAIN_AI   = AI_AVAILABLE and not PLAY_MODE
    steps_done = 0
    n_actions  = 6
    epsilon    = EPS_START
    policy_net = target_net = optimizer = scheduler = None
    memory = stacker = nstep_buf = shaper = drill_tracker = monitor = None

    # ── TRAINING MODE ────────────────────────────────────────────────────────
    if TRAIN_AI:
        print("[AI] Inizializzazione reti neurali — TRAINING MODE...")
        policy_net = DrillerDuelingDQN(INPUT_SHAPE, n_actions, state_dim=STATE_DIM).to(device)
        target_net = DrillerDuelingDQN(INPUT_SHAPE, n_actions, state_dim=STATE_DIM).to(device)
        optimizer  = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE,
                                 weight_decay=1e-5, amsgrad=True)
        scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500_000, eta_min=1e-5)
        steps_done, epsilon = load_checkpoint(policy_net, optimizer, scheduler)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        memory        = ReplayMemory(MEMORY_SIZE)
        memory.load(BUFFER_PATH)
        stacker       = SmartFrameStacker(N_FRAMES, (N_CHANNELS, 11, 9), device)
        nstep_buf     = NStepBuffer(n_step=N_STEP, gamma=GAMMA)
        shaper        = RewardShaper()
        drill_tracker = DrillTracker()
        print(f"[AI] Reti pronte — Device: {device}")
        print(f"[AI] v5.4: LR={LEARNING_RATE}, TARGET_UPDATE={TARGET_UPDATE}, TRAIN_EVERY={TRAIN_EVERY}")
        try:
            from training_monitor import TrainingMonitor
            monitor = TrainingMonitor(log_dir="runs/driller_v5")
            print("[Monitor] Monitor CSV attivo.")
        except ImportError:
            print("[Monitor] training_monitor.py non trovato.")
        except Exception as e:
            print(f"[Monitor] Errore init: {e}. Disabilitato.")

    if pygame.mixer is None:
        pygame.mixer.pre_init(44100, -16, 2, 2048)
    pygame.init()
    pygame.mixer.init()
    if TRAIN_AI and monitor:
        monitor.finalize_writer()

    FPS      = 60
    fpsClock = pygame.time.Clock()
    surface  = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Mr. Driller AI — Rainbow DQN v5.4")

    font_path = path.join("Assets", "Misc", "police", "Act_Of_Rejection.ttf")
    try:
        ui_font    = pygame.font.Font(font_path, 34)
        debug_font = pygame.font.Font(font_path, 20)
        big_font   = pygame.font.Font(font_path, 48)
    except FileNotFoundError:
        ui_font    = pygame.font.SysFont("Arial", 32, bold=True)
        debug_font = pygame.font.SysFont("Arial", 18, bold=True)
        big_font   = pygame.font.SysFont("Arial", 48, bold=True)

    ui_bg_path = path.join("Assets", "Misc", "userinterface.png")
    try:
        raw_bg = pygame.image.load(ui_bg_path).convert()
        Ui_bg  = pygame.transform.scale(raw_bg, (screen_width, screen_height))
    except FileNotFoundError:
        Ui_bg = pygame.Surface((screen_width, screen_height))
        Ui_bg.fill((30, 30, 30))

    life_icon_path = path.join("Assets", "Misc", "icon.png")
    icon = pygame.image.load(life_icon_path).convert_alpha() if path.exists(life_icon_path) else None
    if icon and icon.get_width() > 40:
        icon = pygame.transform.scale(icon, (32, 32))

    oxy_frames = load_oxy_frames()

    def play_music(level_id):
        if not pygame.mixer: return
        try:
            pygame.mixer.music.stop()
            fname = f"Level{level_id}.wav" if level_id > 0 else "menu.wav"
            fpath = path.join("Assets", "Music", fname)
            if path.exists(fpath):
                pygame.mixer.music.load(fpath)
                pygame.mixer.music.set_volume(0.4)
                pygame.mixer.music.play(-1)
        except Exception as e:
            print(f"[Audio] Errore: {e}")

    pygame.event.pump()
    player = Character(3, 4, 1, 2)
    level, levelID, won = restart(player)
    play_music(0)

    # ── PLAY MODE ────────────────────────────────────────────────────────────
    if PLAY_MODE and AI_AVAILABLE:
        print("[AI] Modalità PLAY — caricamento modello...")
        policy_net = DrillerDuelingDQN(INPUT_SHAPE, n_actions, state_dim=STATE_DIM).to(device)
        policy_net.eval()
        if os.path.exists(MODEL_PATH):
            ckpt = torch.load(MODEL_PATH, map_location=device)
            policy_net.load_state_dict(ckpt["model_state_dict"], strict=True)
            steps_done = ckpt.get("steps", 0)
            epsilon    = 0.0   # nessuna esplorazione casuale
            print(f"[AI] Modello caricato — step {steps_done}")
        else:
            print("[AI] ERRORE: nessun modello trovato!")
        stacker       = SmartFrameStacker(N_FRAMES, (N_CHANNELS, 11, 9), device)
        shaper        = RewardShaper()
        drill_tracker = DrillTracker()
        _reset_ai_episode(player, level, stacker, None, shaper, drill_tracker)

    # ── reset episodio training ───────────────────────────────────────────────
    if TRAIN_AI:
        _reset_ai_episode(player, level, stacker, nstep_buf, shaper, drill_tracker, memory=None)

    EVT_END_BLOCK = pygame.USEREVENT
    EVTICSEC      = pygame.USEREVENT + 1
    EVTICPLY      = pygame.USEREVENT + 2
    EVTIC_GRAVITY = pygame.USEREVENT + 4
    EVTIC_ANIM    = pygame.USEREVENT + 5

    GAME_SPEED = 2.0

    if not HEADLESS_TRAINING:
        pygame.time.set_timer(EVTIC_GRAVITY, int(400  / GAME_SPEED))
        pygame.time.set_timer(EVTIC_ANIM,    int(80   / GAME_SPEED))
        pygame.time.set_timer(EVTICSEC,      int(1050 / GAME_SPEED))
        pygame.time.set_timer(EVTICPLY,      int(120  / GAME_SPEED))

    v_timer_gravity = 0
    v_timer_anim    = 0
    v_timer_sec     = 0
    v_timer_ply     = 0

    currentBotLine = 8
    currentOffset  = 0
    currentClimb   = 0
    backDown       = False
    inProgress = True
    inMenu     = not (TRAIN_AI or PLAY_MODE)   # salta menu in entrambe le modalità AI
    inGame     = not inMenu
    inGameOver = False
    game_won   = False
    menuOption = 1
    movKeys    = [K_w, K_d, K_a, K_s]
    last_action_idx = 0
    ai_state   = "ACTION"
    pt_grid_old = pt_vars_old = pt_action = None
    pt_prev_score = pt_prev_oxy = pt_prev_lives = pt_prev_y = pt_prev_x = 0
    pt_was_hard_block = pt_was_delayed_block = False
    pt_blocks_around  = None
    q_values_np       = None
    gravity_applied_since_action = False
    blocksDisap: set  = set()
    _train_counter    = 0

    try:
        from eventHandling import movementHandle, breaking
        from level import render
    except ImportError:
        def movementHandle(*args): pass
        def breaking(*args): pass
        def render(*args): pass

    print("[GAME] Entro nel loop principale...")
    while inProgress:

        if inMenu:
            mainMenu(surface, menuOption)
            for event in pygame.event.get():
                if event.type == QUIT: inProgress = False
                if event.type == KEYDOWN:
                    if event.key == K_UP: menuOption = 1
                    elif event.key == K_DOWN: menuOption = 2
                    elif event.key == K_RETURN:
                        if menuOption == 1:
                            inMenu = False; inGame = True
                            level, levelID, won = restart(player)
                            play_music(levelID)
                        elif menuOption == 2: inProgress = False
            pygame.display.update()
            continue

        if inGameOver:
            if TRAIN_AI:
                # in training si riparte automaticamente senza fermarsi
                inGameOver = False; inGame = True
                player = Character(3, 4, 1, 2)
                level, levelID, won = restart(player)
                currentBotLine = 8; currentOffset = 0; blocksDisap = set()
                clear_cache()
                ai_state = "ACTION"; gravity_applied_since_action = False
                _reset_ai_episode(player, level, stacker, nstep_buf, shaper, drill_tracker, memory=memory)
                continue
            # in play mode si mostra game over e si aspetta input
            if not HEADLESS_TRAINING:
                draw_gameover(surface, big_font, game_won)
                pygame.display.update()
            for event in pygame.event.get():
                if event.type == QUIT: inProgress = False
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE: inProgress = False
                    elif event.key == K_r:
                        inGameOver = False; inGame = True
                        player = Character(3, 4, 1, 2)
                        level, levelID, won = restart(player)
                        currentBotLine = 8; currentOffset = 0; blocksDisap = set()
                        clear_cache(); play_music(levelID)
                        if PLAY_MODE and AI_AVAILABLE:
                            _reset_ai_episode(player, level, stacker, None, shaper, drill_tracker)
            if not HEADLESS_TRAINING: fpsClock.tick(FPS)
            continue

        # ── FASE ACTION (training E play mode) ───────────────────────────────
        AI_ACTIVE = TRAIN_AI or (PLAY_MODE and AI_AVAILABLE)
        if AI_ACTIVE and inGame and ai_state == "ACTION" and player.IdlingAcc():
            pt_prev_score = player.scoreAcc()
            pt_prev_oxy   = player.oxyAcc()
            pt_prev_lives = player.livesAcc()
            pt_prev_y, pt_prev_x = player.posAcc()
            pt_grid_old = stacker.get_state()
            pt_vars_old = get_internal_state_vector(player, len(level), len(level[0]), level, last_action_idx, n_actions)
            pt_blocks_around = {}
            try:
                if pt_prev_y > 0:               pt_blocks_around["up"]    = level[pt_prev_y-1][pt_prev_x]
                if pt_prev_y+1 < len(level):    pt_blocks_around["down"]  = level[pt_prev_y+1][pt_prev_x]
                if pt_prev_x > 0:               pt_blocks_around["left"]  = level[pt_prev_y][pt_prev_x-1]
                if pt_prev_x+1 < len(level[0]): pt_blocks_around["right"] = level[pt_prev_y][pt_prev_x+1]
            except Exception: pass

            if TRAIN_AI and epsilon > EPS_END:
                epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)

            pt_action, q_values_np = select_action(policy_net, pt_grid_old, pt_vars_old, epsilon, n_actions)
            action_idx = pt_action.item()
            last_action_idx = action_idx

            pt_was_hard_block = pt_was_delayed_block = False
            if action_idx in [3, 4, 5]:
                tx, ty = int(pt_prev_x), int(pt_prev_y)
                if action_idx == 3: tx -= 1
                elif action_idx == 4: tx += 1
                elif action_idx == 5: ty += 1
                if 0 <= ty < len(level) and 0 <= tx < len(level[0]):
                    tb = level[ty][tx]
                    if tb.hpAccess() > 0:
                        if   tb.typeAccess() == "unbreakable": pt_was_hard_block    = True
                        elif tb.typeAccess() == "delayed":     pt_was_delayed_block = True

            if   action_idx == 1: movementHandle(pygame.event.Event(KEYDOWN, key=movKeys[2]), surface, player, level, movKeys)
            elif action_idx == 2: movementHandle(pygame.event.Event(KEYDOWN, key=movKeys[1]), surface, player, level, movKeys)
            elif action_idx == 3: breaking(pygame.event.Event(KEYDOWN, key=K_LEFT),  surface, player, level, currentBotLine)
            elif action_idx == 4: breaking(pygame.event.Event(KEYDOWN, key=K_RIGHT), surface, player, level, currentBotLine)
            elif action_idx == 5: breaking(pygame.event.Event(KEYDOWN, key=K_DOWN),  surface, player, level, currentBotLine)

            steps_done += 1
            gravity_applied_since_action = False
            ai_state = "WAITING"

        # ── TIMER VIRTUALE HEADLESS ───────────────────────────────────────────
        if HEADLESS_TRAINING and inGame:
            dt = 16.6
            v_timer_gravity += dt; v_timer_anim += dt
            v_timer_sec     += dt; v_timer_ply  += dt

            if v_timer_gravity >= int(400 / GAME_SPEED):
                pygame.event.post(pygame.event.Event(EVTIC_GRAVITY))
                v_timer_gravity -= int(400 / GAME_SPEED)
            if v_timer_anim >= int(80 / GAME_SPEED):
                pygame.event.post(pygame.event.Event(EVTIC_ANIM))
                v_timer_anim -= int(80 / GAME_SPEED)
            if v_timer_sec >= int(1050 / GAME_SPEED):
                pygame.event.post(pygame.event.Event(EVTICSEC))
                v_timer_sec -= int(1050 / GAME_SPEED)
            if v_timer_ply >= int(120 / GAME_SPEED):
                pygame.event.post(pygame.event.Event(EVTICPLY))
                v_timer_ply -= int(120 / GAME_SPEED)

        for event in pygame.event.get():
            if event.type == QUIT:
                if TRAIN_AI:
                    save_checkpoint(policy_net, optimizer, steps_done, epsilon, scheduler)
                    memory.save_async(BUFFER_PATH)
                inProgress = False
            if event.type == EVT_END_BLOCK: won = True
            if inGame:
                if event.type == EVTICSEC:
                    player.updateOxygen(1, surface, level)
                    dead_delayed: set = set()
                    for (r, c) in blocksDisap:
                        if level[r][c].hpAccess() > 0:
                            if hasattr(level[r][c],'timeout') and level[r][c].timeout():
                                dead_delayed.add((r, c))
                        else: dead_delayed.add((r, c))
                    for (r, c) in dead_delayed:
                        if level[r][c].hpAccess() == 0:
                            nb = Classic(c, r, 1, 0); nb._hp = 0; nb.changeBG(levelID); level[r][c] = nb
                    blocksDisap -= dead_delayed
                    if dead_delayed:
                        applyGravity(level, player, surface); player.fall(surface, level)
                if event.type == EVTIC_ANIM: player.Anim(surface)
                if event.type == EVTIC_GRAVITY:
                    applyGravity(level, player, surface)
                    gravity_applied_since_action = True
                    if player.blocksFallenAcc() != currentOffset:
                        currentOffset += 1; currentBotLine += 1
                        for line in level:
                            for el in line: el.updOffset(currentOffset)
                if event.type == EVTICPLY: player.NeedToIdle(surface)
                if event.type in (EVTIC_GRAVITY, EVTICSEC):
                    for r, row in enumerate(level):
                        for c, blk in enumerate(row):
                            if blk.typeAccess()=="delayed" and blk.idAcc() and blk.hpAccess()>0:
                                blocksDisap.add((r, c))

        if backDown and player.climbAcc() < currentClimb:
            if currentClimb == 0: backDown = False
            player.backDownCleanup(surface)
        currentClimb = player.climbAcc()
        if player.climbAcc() > 0: backDown = True

        if inGame:
            player.fall(surface, level)
            if player.blocksFallenAcc() != currentOffset:
                currentOffset += 1; currentBotLine += 1
                for line in level:
                    for el in line: el.updOffset(currentOffset)

        if AI_ACTIVE and inGame and ai_state == "WAITING":
            is_terminal = (player.livesAcc() < 0) or won
            if is_terminal or (gravity_applied_since_action and player.IdlingAcc()):
                ai_state = "LEARNING"

        # ── FASE LEARNING (solo training) ─────────────────────────────────────
        if TRAIN_AI and inGame and ai_state == "LEARNING":
            new_pY, new_pX = player.posAcc()
            cur_frame = get_local_window_tensor(level, new_pX, new_pY).to(device)
            stacker.push(cur_frame)
            grid_new = stacker.get_state()
            vars_new = get_internal_state_vector(player, len(level), len(level[0]), level, last_action_idx, n_actions)
            shaping_bonus = shaper.step(new_pY, player.oxyAcc(), len(level), GAMMA)
            reward = calculate_reward(
                prev_y=pt_prev_y, prev_x=pt_prev_x, new_y=new_pY, new_x=new_pX,
                prev_oxy=pt_prev_oxy, new_oxy=player.oxyAcc(),
                prev_score=pt_prev_score, new_score=player.scoreAcc(),
                prev_lives=pt_prev_lives, new_lives=player.livesAcc(),
                action_idx=last_action_idx, is_hard_block=pt_was_hard_block,
                is_delayed_block=pt_was_delayed_block, drill_tracker=drill_tracker,
                total_rows=len(level), shaping_bonus=shaping_bonus,
                is_level_complete=won, blocks_around=pt_blocks_around,
            )
            done = torch.tensor([player.livesAcc() < 0 or won], device=device)
            nstep_buf.push(pt_grid_old, pt_vars_old, pt_action, grid_new, vars_new, reward, done)

            if steps_done % 100 == 0:
                print(f"[STEP {steps_done:6d}] R: {reward.item():+7.2f} | dy: {new_pY-pt_prev_y} | "
                      f"Oxy: {int(player.oxyAcc()):3d}% | e: {epsilon:.4f} | "
                      f"buf: {len(memory)}")

            if nstep_buf.is_ready():
                nstep_trans = nstep_buf.get()
                if nstep_trans: memory.push(*nstep_trans)

            _train_counter += 1
            opt_result = None
            if _train_counter >= TRAIN_EVERY:
                opt_result = optimize_model(policy_net, target_net, memory, optimizer, BATCH_SIZE, GAMMA)
                if opt_result is not None: scheduler.step()
                _train_counter = 0

            if monitor:
                loss_val    = opt_result[0] if opt_result else None
                q_mean_safe = q_values_np if q_values_np is not None else np.zeros(n_actions, dtype=np.float32)
                monitor.log_step(reward=reward.item(), action_idx=last_action_idx,
                    depth=currentOffset-currentClimb, oxy=player.oxyAcc(),
                    loss=loss_val, epsilon=epsilon, steps_done=steps_done, q_mean=q_mean_safe,
                    lr=scheduler.get_last_lr()[0] if scheduler else LEARNING_RATE,
                    mem_size=len(memory), was_hard_block=pt_was_hard_block,
                    was_delayed_block=pt_was_delayed_block, player_x=new_pX, player_y=new_pY)

            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if steps_done % 1000 == 0 and steps_done > 0:
                save_checkpoint(policy_net, optimizer, steps_done, epsilon, scheduler)

            if steps_done % BUFFER_SAVE_EVERY == 0 and steps_done > 0:
                memory.save_async(BUFFER_PATH)

            ai_state = "ACTION"

        # In play mode, dopo WAITING si torna subito ad ACTION (nessun learning)
        if PLAY_MODE and inGame and ai_state == "LEARNING":
            new_pY, new_pX = player.posAcc()
            cur_frame = get_local_window_tensor(level, new_pX, new_pY).to(device)
            stacker.push(cur_frame)
            ai_state = "ACTION"

        if player.livesAcc() < 0 or won:
            if TRAIN_AI and monitor:
                death_cause = "win" if won else ("oxy" if player.oxyAcc() <= 0 else "block_fall")
                monitor.log_episode(steps_done=steps_done, level_id=levelID, won=won,
                    death_cause=death_cause, depth=currentOffset-currentClimb,
                    final_oxy=player.oxyAcc(), score=player.scoreAcc(), lives_left=max(player.livesAcc(), 0))
            if won:
                if levelID >= MAX_LEVEL:
                    if TRAIN_AI:
                        save_checkpoint(policy_net, optimizer, steps_done, epsilon, scheduler)
                        memory.save_async(BUFFER_PATH)
                    inGame = False; inGameOver = True; game_won = True; won = False; continue
                level, levelID, game_completed = changeLvl(levelID, player, is_ai=(TRAIN_AI or PLAY_MODE))
                if game_completed: level, levelID, _ = restart(player)
                won = False
            else:
                level, levelID, won = restart(player)
            play_music(levelID)
            currentBotLine = 8; currentOffset = 0; blocksDisap = set()
            clear_cache()
            ai_state = "ACTION"; gravity_applied_since_action = False
            if TRAIN_AI:
                _reset_ai_episode(player, level, stacker, nstep_buf, shaper, drill_tracker, memory=memory)
            elif PLAY_MODE and AI_AVAILABLE:
                _reset_ai_episode(player, level, stacker, None, shaper, drill_tracker)
            continue

        if inGame and not HEADLESS_TRAINING:
            surface.fill((10, 5, 20))
            surface.blit(Ui_bg, (0, 0))
            render(surface, level, currentOffset)
            player.display(surface)
            score_val  = player.scoreAcc()
            score_text = f"{score_val}" if score_val < 1000 else f"{score_val/1000:.1f} k"
            surface.blit(big_font.render(score_text, True, (220, 0, 255)), (640, 107))
            oxy_val = max(0, min(100, int(player.oxyAcc())))
            surface.blit(big_font.render(str(oxy_val), True, (220, 0, 255)), (640, 200))
            if oxy_frames and oxy_val in oxy_frames and oxy_frames[oxy_val]:
                surface.blit(oxy_frames[oxy_val], (537, 252))
            real_depth = currentOffset - currentClimb
            surface.blit(big_font.render(str(real_depth), True, (220, 0, 255)), (640, 377))
            if icon:
                for i in range(player.livesAcc()): surface.blit(icon, (700 - i*70, 500))
            if AI_ACTIVE: draw_ai_debug(surface, debug_font, epsilon, steps_done, last_action_idx, ai_state)
            pygame.display.update()
        if not HEADLESS_TRAINING: fpsClock.tick(FPS)

    if TRAIN_AI and monitor:
        monitor.print_summary(); monitor.plot_all(); monitor.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    game(SCREEN_W, SCREEN_H)