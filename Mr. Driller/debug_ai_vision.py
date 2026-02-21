import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pygame  # Necessario per catturare lo schermo

def save_ai_view(frame_tensor, step_num, player_pos, screen_surface=None):
    """
    Salva un'immagine che include lo screenshot del gioco + i 6 canali dell'AI.
    """
    os.makedirs("debug", exist_ok=True)
    
    if frame_tensor.is_cuda:
        frame_tensor = frame_tensor.cpu()
    grid = frame_tensor.numpy()
    
    # Prepariamo la figura: 3 righe, 3 colonne
    # (1 slot per il gioco reale, 6 slot per i canali, 2 slot vuoti o per statistiche)
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    fig.suptitle(f"AI Perception Analysis — Step {step_num} — Pos: {player_pos}", 
                 fontsize=20, fontweight='bold', y=0.95)

    # ── 0. SCREENSHOT DEL GIOCO REALE ──────────────────────────────────────
    ax_game = fig.add_subplot(gs[0, 0])
    if screen_surface is not None:
        # Cattura i pixel da Pygame
        view = pygame.surfarray.array3d(screen_surface)
        # Pygame usa (width, height, RGB), Matplotlib vuole (height, width, RGB)
        view = view.transpose([1, 0, 2])
        ax_game.imshow(view)
        ax_game.set_title("REAL GAME SCREEN", fontsize=14, color='darkred', fontweight='bold')
    else:
        ax_game.text(0.5, 0.5, "Screen not provided", ha='center', va='center')
    ax_game.axis('off')

    # ── CANALI AI ──────────────────────────────────────────────────────────
    channel_info = [
        ("CH0: Strategic Value", "RdYlGn"),
        ("CH1: Color + Combo", "viridis"),
        ("CH2: HP/State", "RdYlGn"),
        ("CH3: Urgency (Gravity)", "hot"),
        ("CH4: Geometry (Distance)", "cool"),
        ("CH5: Movability", "gray"),
    ]

    # Mappiamo i canali sugli slot rimanenti della griglia 3x3
    # Saltiamo lo slot [0,0] che è occupato dal gioco reale
    plot_indices = [(0,1), (0,2), (1,0), (1,1), (1,2), (2,0)]
    
    for idx, (ax_pos, (title, cmap)) in enumerate(zip(plot_indices, channel_info)):
        ax = fig.add_subplot(gs[ax_pos])
        im = ax.imshow(grid[idx], cmap=cmap, vmin=0, vmax=1.2, 
                       interpolation='nearest', origin='upper')
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Overlay griglia tecnica
        ax.set_xticks(np.arange(-0.5, grid.shape[2], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Segna il centro (Player) - Assumendo finestra 11x9 (centro 5,4) o 9x7 (centro 4,3)
        cy, cx = grid.shape[1]//2, grid.shape[2]//2
        ax.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    # ── LEGENDA E INFO ──────────────────────────────────────────────────────
    ax_info = fig.add_subplot(gs[2, 1:]) # Occupa gli ultimi due slot in basso
    ax_info.axis('off')
    info_text = (
        f"STEP: {step_num}\n"
        f"PLAYER POS: {player_pos}\n\n"
        "LEGEND:\n"
        "CH0: Green=Target, Red=Danger | CH3: Brighter=Falling Block!\n"
        "CH5: White=Walkable/Air, Black=Solid Wall"
    )
    ax_info.text(0.0, 0.5, info_text, transform=ax_info.transAxes, 
                 fontsize=13, verticalalignment='center', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))

    # Salvataggio
    filepath = os.path.join("debug", f"ai_debug_full_{step_num:06d}.png")
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close(fig) # Importante per non consumare memoria
    print(f"  >> [DEBUG IMAGE SAVED]: {filepath}")