"""
training_monitor.py — Monitor CSV+Plot per Mr. Driller Rainbow DQN
====================================================================
VERSIONE SENZA TENSORBOARD — usa solo CSV + matplotlib.

Motivazione: SummaryWriter (torch.utils.tensorboard e tensorboardX)
crasha con segfault su questo sistema a causa di conflitti tra le
librerie gRPC/protobuf di TensorBoard e SDL/X11 di pygame.
Il crash avviene su SummaryWriter.__init__() indipendentemente
dall'ordine di inizializzazione.

Questa versione:
  - Scrive su CSV (sempre disponibile, zero dipendenze)
  - Genera grafici PNG con matplotlib a fine training (o ogni N episodi)
  - API identica alla versione TensorBoard → nessuna modifica a main.py
  - finalize_writer() è uno stub vuoto per compatibilità

Come vedere i dati durante il training:
  - Console: stampa [EP ...] ogni episodio
  - Grafici PNG aggiornati ogni plot_every episodi in runs/.../plots/

Come vedere i dati a fine training:
  - monitor.plot_all() genera PNG in runs/.../plots/
  - training_log.csv apribile con Excel/LibreOffice/pandas
"""

import os
import csv
import time
import numpy as np
from collections import deque


# ─────────────────────────────────────────────────────────────────────
# CSV WRITER
# ─────────────────────────────────────────────────────────────────────
class CSVWriter:
    """
    Scrive metriche su CSV con flush periodico.
    Formato colonne: step, tag, value
    """
    def __init__(self, path: str):
        self.path  = path
        self._buf  = []
        self._file = open(path, 'w', newline='', encoding='utf-8')
        self._csv  = csv.writer(self._file)
        self._csv.writerow(['step', 'tag', 'value'])
        self._file.flush()
        print(f"[Monitor] 📄 Log CSV: {path}")

    def add_scalar(self, tag: str, value: float, step: int):
        self._buf.append((step, tag, float(value)))
        if len(self._buf) >= 200:
            self._flush_buf()

    def _flush_buf(self):
        for row in self._buf:
            self._csv.writerow(row)
        self._buf.clear()
        self._file.flush()

    def add_histogram(self, *args, **kwargs):
        pass  # Non supportato — ignorato silenziosamente

    def flush(self):
        self._flush_buf()

    def close(self):
        self._flush_buf()
        self._file.close()


# ─────────────────────────────────────────────────────────────────────
# TRAINING MONITOR
# ─────────────────────────────────────────────────────────────────────
class TrainingMonitor:
    """
    Monitor centralizzato per training Rainbow DQN.
    Backend: CSV + matplotlib (no TensorBoard, no crash SDL/X11).
    """

    ACTION_NAMES = ["IDLE", "LEFT", "RIGHT", "DRILL_L", "DRILL_R", "DRILL_D"]

    def __init__(self, log_dir="runs/driller_dqn", log_every=50,
                 window=100, plot_every=500, **kwargs):
        self.log_every  = log_every
        self.window     = window
        self.plot_every = plot_every

        # Crea cartelle run
        self.run_name  = f"{log_dir}_{time.strftime('%m%d_%H%M%S')}"
        self.plots_dir = os.path.join(self.run_name, "plots")
        os.makedirs(self.run_name,  exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Writer CSV
        csv_path    = os.path.join(self.run_name, "training_log.csv")
        self.writer = CSVWriter(csv_path)
        self.csv_path = csv_path

        # ── Stato episodio corrente ────────────────────────────────────
        self._ep_reward       = 0.0
        self._ep_steps        = 0
        self._ep_max_depth    = 0
        self._ep_min_oxy      = 100.0
        self._ep_wall_hits    = 0
        self._ep_delayed_hits = 0
        self._ep_action_hist  = np.zeros(6, dtype=np.int32)

        # ── Rolling windows ────────────────────────────────────────────
        self._ep_rewards    = deque(maxlen=window)
        self._ep_depths     = deque(maxlen=window)
        self._ep_lengths    = deque(maxlen=window)
        self._ep_won        = deque(maxlen=window)
        self._ep_wall_rates = deque(maxlen=window)

        # ── Storico completo (per grafici) ─────────────────────────────
        self._all_ep_rewards = []
        self._all_ep_depths  = []
        self._all_ep_lengths = []
        self._all_ep_won     = []

        # ── Step buffers ───────────────────────────────────────────────
        self._losses  = deque(maxlen=100)
        self._q_means = deque(maxlen=100)
        self._rewards = deque(maxlen=100)

        self._current_lr  = None
        self._current_mem = None
        self.episode_count = 0

        print(f"[Monitor] 🚀 TrainingMonitor avviato (CSV+matplotlib)")
        print(f"[Monitor] 📁 Run: {self.run_name}")
        print(f"[Monitor] 📊 Grafici PNG aggiornati ogni {plot_every} episodi")

    def finalize_writer(self):
        """
        Stub per compatibilità con main.py che chiama finalize_writer()
        dopo pygame.init(). In questa versione CSV non serve fare nulla.
        """
        print("[Monitor] ✅ CSV writer già attivo (nessun SummaryWriter da finalizzare)")

    # ──────────────────────────────────────────────────────────────────
    # LOG STEP
    # ──────────────────────────────────────────────────────────────────
    def log_step(self, reward=0.0, action_idx=0, depth=0, oxy=100.0,
                 loss=None, q_mean=None, epsilon=None, steps_done=0,
                 was_hard_block=False, was_delayed_block=False,
                 lr=None, mem_size=None, player_x=None, player_y=None,
                 q_values=None, **kwargs):
        """Chiamare ogni step AI (FASE ACTION e FASE LEARNING)."""

        # Alias retrocompatibilità q_values → q_mean
        if q_mean is None and q_values is not None:
            q_mean = q_values

        # Accumulo episodio
        self._ep_reward    += reward
        self._ep_steps     += 1
        self._ep_max_depth  = max(self._ep_max_depth, depth)
        self._ep_min_oxy    = min(self._ep_min_oxy, oxy)
        if was_hard_block:    self._ep_wall_hits    += 1
        if was_delayed_block: self._ep_delayed_hits += 1
        if 0 <= action_idx < 6:
            self._ep_action_hist[action_idx] += 1

        self._rewards.append(reward)
        if loss   is not None: self._losses.append(float(loss))
        if q_mean is not None:
            val = float(q_mean.mean()) if hasattr(q_mean, 'mean') else float(q_mean)
            self._q_means.append(val)
        if lr       is not None: self._current_lr  = lr
        if mem_size is not None: self._current_mem = mem_size

        if steps_done > 0 and steps_done % self.log_every == 0:
            self._flush_step(steps_done, epsilon)

    def _flush_step(self, step: int, epsilon):
        w = self.writer
        if self._rewards: w.add_scalar('Step/Reward_Mean',  np.mean(self._rewards), step)
        if self._losses:  w.add_scalar('Step/Loss_Mean',    np.mean(self._losses),  step)
        if self._q_means: w.add_scalar('Step/Q_Mean',       np.mean(self._q_means), step)
        if epsilon is not None:
            w.add_scalar('Step/Epsilon', epsilon, step)
        if self._current_lr is not None:
            w.add_scalar('Step/Learning_Rate', self._current_lr, step)
        if self._current_mem is not None:
            w.add_scalar('Step/Memory_Size', self._current_mem, step)

        tot = self._ep_action_hist.sum() or 1
        for i, name in enumerate(self.ACTION_NAMES):
            w.add_scalar(f'Actions/{name}_pct', self._ep_action_hist[i] / tot * 100, step)

        drill_d = int(self._ep_action_hist[5])
        drill_l = int(self._ep_action_hist[3]) + int(self._ep_action_hist[4])
        w.add_scalar('Pathology/DrillDown_ratio', drill_d / max(drill_l + drill_d, 1), step)
        w.add_scalar('Pathology/IDLE_pct',        self._ep_action_hist[0] / tot * 100, step)
        w.add_scalar('Pathology/WallHits',        self._ep_wall_hits,    step)
        w.add_scalar('Pathology/DelayedHits',     self._ep_delayed_hits, step)

    # ──────────────────────────────────────────────────────────────────
    # LOG EPISODE
    # ──────────────────────────────────────────────────────────────────
    def log_episode(self, steps_done=0, level_id=0, won=False,
                    death_cause="unknown", depth=0, final_oxy=None,
                    score=0, lives_left=0, **kwargs):
        """Chiamare su morte o fine livello."""
        self.episode_count += 1
        ep = self.episode_count
        w  = self.writer

        ep_depth   = max(self._ep_max_depth, depth)
        ep_min_oxy = self._ep_min_oxy
        if final_oxy is not None:
            ep_min_oxy = min(ep_min_oxy, final_oxy)

        # Rolling e storico
        self._ep_rewards.append(self._ep_reward)
        self._ep_depths.append(ep_depth)
        self._ep_lengths.append(self._ep_steps)
        self._ep_won.append(float(won))
        wall_rate = self._ep_wall_hits / max(self._ep_steps, 1)
        self._ep_wall_rates.append(wall_rate)

        self._all_ep_rewards.append(self._ep_reward)
        self._all_ep_depths.append(ep_depth)
        self._all_ep_lengths.append(self._ep_steps)
        self._all_ep_won.append(float(won))

        # CSV
        w.add_scalar('Episode/Total_Reward',  self._ep_reward, ep)
        w.add_scalar('Episode/Length_Steps',  self._ep_steps,  ep)
        w.add_scalar('Episode/Max_Depth',     ep_depth,        ep)
        w.add_scalar('Episode/Min_Oxygen',    ep_min_oxy,      ep)
        w.add_scalar('Episode/Level_ID',      level_id,        ep)
        w.add_scalar('Episode/Won',           float(won),      ep)
        w.add_scalar('Episode/WallBash_Rate', wall_rate,       ep)
        w.add_scalar('Episode/Score',         score,           ep)

        drill_d = int(self._ep_action_hist[5])
        drill_l = int(self._ep_action_hist[3]) + int(self._ep_action_hist[4])
        drill_ratio = drill_d / max(drill_l + drill_d, 1)
        w.add_scalar('Episode/DrillDown_ratio', drill_ratio, ep)

        tot        = self._ep_action_hist.sum() or 1
        idle_ratio = self._ep_action_hist[0] / tot * 100
        w.add_scalar('Episode/IDLE_pct', idle_ratio, ep)

        for i, name in enumerate(self.ACTION_NAMES):
            w.add_scalar(f'EpActions/{name}_pct', self._ep_action_hist[i] / tot * 100, ep)

        if len(self._ep_rewards) >= 5:
            w.add_scalar('Rolling/Reward_Avg',   np.mean(self._ep_rewards),    ep)
            w.add_scalar('Rolling/Depth_Avg',    np.mean(self._ep_depths),     ep)
            w.add_scalar('Rolling/WinRate',      np.mean(self._ep_won),        ep)
            w.add_scalar('Rolling/WallRate_Avg', np.mean(self._ep_wall_rates), ep)

        cause_map = {"oxy": 0, "block_fall": 1, "unknown": 2, "win": 3}
        w.add_scalar('Death/Cause_code', cause_map.get(death_cause, 2), ep)

        # Console
        avg_r  = f"{np.mean(self._ep_rewards):+.1f}" if self._ep_rewards else "---"
        status = "WIN" if won else f"DEAD({death_cause})"
        print(
            f"[EP {ep:5d}|Step {steps_done:7d}] "
            f"R:{self._ep_reward:+8.2f}  Avg:{avg_r:>8}  "
            f"Depth:{ep_depth:3d}  Oxy:{int(ep_min_oxy):3d}%  "
            f"IDLE:{idle_ratio:4.1f}%  DrillD:{drill_ratio*100:4.1f}%  "
            f"Lv{level_id} {status}"
        )

        # Grafici periodici automatici
        if ep % self.plot_every == 0:
            self.plot_all(silent=True)

        # Reset stato episodio
        self._ep_reward       = 0.0
        self._ep_steps        = 0
        self._ep_max_depth    = 0
        self._ep_min_oxy      = 100.0
        self._ep_wall_hits    = 0
        self._ep_delayed_hits = 0
        self._ep_action_hist  = np.zeros(6, dtype=np.int32)

    # ──────────────────────────────────────────────────────────────────
    # GRAFICI PNG
    # ──────────────────────────────────────────────────────────────────
    def plot_all(self, output_dir: str = None, silent: bool = False):
        """
        Genera grafici PNG dallo storico degli episodi.
        Richiede matplotlib. Se non disponibile, stampa avviso e continua.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Backend non-interattivo
            import matplotlib.pyplot as plt
        except ImportError:
            if not silent:
                print("[Monitor] ⚠️  matplotlib non disponibile.")
                print("[Monitor] 💡 Installa con: pip install matplotlib")
            return

        out = output_dir or self.plots_dir
        os.makedirs(out, exist_ok=True)

        def smooth(data, k=20):
            if len(data) < k: return list(data)
            return list(np.convolve(data, np.ones(k)/k, mode='valid'))

        # Reward
        if self._all_ep_rewards:
            fig, ax = plt.subplots(figsize=(12, 4))
            r = self._all_ep_rewards
            ax.plot(r, alpha=0.25, color='steelblue', lw=0.8, label='Per episodio')
            if len(r) >= 20:
                ax.plot(range(19, len(r)), smooth(r, 20), color='steelblue',
                        lw=2, label='Media (20 ep)')
            ax.axhline(0, color='gray', lw=0.5, ls='--')
            ax.set_title('Reward per Episodio')
            ax.set_xlabel('Episodio'); ax.set_ylabel('Reward')
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            p = os.path.join(out, 'reward_curve.png')
            plt.savefig(p, dpi=120); plt.close()
            if not silent: print(f"[Monitor] 📈 {p}")

        # Depth
        if self._all_ep_depths:
            fig, ax = plt.subplots(figsize=(12, 4))
            d = self._all_ep_depths
            ax.plot(d, alpha=0.3, color='green', lw=0.8)
            if len(d) >= 20:
                ax.plot(range(19, len(d)), smooth(d, 20), color='darkgreen',
                        lw=2, label='Media (20 ep)')
            ax.set_title('Profondità Massima per Episodio')
            ax.set_xlabel('Episodio'); ax.set_ylabel('Righe')
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            p = os.path.join(out, 'depth_curve.png')
            plt.savefig(p, dpi=120); plt.close()
            if not silent: print(f"[Monitor] 📈 {p}")

        # Win rate
        if self._all_ep_won:
            fig, ax = plt.subplots(figsize=(12, 4))
            wn = self._all_ep_won; k = 20
            wr = [np.mean(wn[max(0,i-k):i+1]) * 100 for i in range(len(wn))]
            ax.plot(wr, color='gold', lw=2)
            ax.fill_between(range(len(wr)), wr, alpha=0.25, color='gold')
            ax.set_title(f'Win Rate Rolling (finestra {k} ep)')
            ax.set_xlabel('Episodio'); ax.set_ylabel('Win Rate (%)')
            ax.set_ylim(0, 100); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            p = os.path.join(out, 'win_rate.png')
            plt.savefig(p, dpi=120); plt.close()
            if not silent: print(f"[Monitor] 📈 {p}")

        # Action distribution
        if self._ep_action_hist.sum() > 0:
            fig, ax = plt.subplots(figsize=(7, 5))
            sizes  = self._ep_action_hist.astype(float)
            labels = [n for n, s in zip(self.ACTION_NAMES, sizes) if s > 0]
            vals   = [s for s in sizes if s > 0]
            colors = ['#e74c3c','#3498db','#2ecc71','#9b59b6','#f39c12','#1abc9c']
            ax.pie(vals, labels=labels, autopct='%1.1f%%', colors=colors[:len(vals)])
            ax.set_title('Distribuzione Azioni (sessione corrente)')
            plt.tight_layout()
            p = os.path.join(out, 'action_dist.png')
            plt.savefig(p, dpi=120); plt.close()
            if not silent: print(f"[Monitor] 📈 {p}")

        if not silent:
            print(f"[Monitor] ✅ Grafici in: {out}")

    # ──────────────────────────────────────────────────────────────────
    # SUMMARY & CLOSE
    # ──────────────────────────────────────────────────────────────────
    def print_summary(self):
        print("\n" + "═"*60)
        print("  🏁  RIEPILOGO FINALE TRAINING")
        print("═"*60)
        print(f"  Episodi totali       : {self.episode_count}")
        if self._all_ep_depths:
            print(f"  Profondità massima   : {max(self._all_ep_depths)}")
            print(f"  Profondità media     : {np.mean(self._all_ep_depths):.1f}")
        if self._all_ep_rewards:
            print(f"  Reward media totale  : {np.mean(self._all_ep_rewards):.1f}")
        if self._ep_rewards:
            print(f"  Reward ultima window : {np.mean(self._ep_rewards):.1f}")
        if self._all_ep_won:
            print(f"  Win rate totale      : {np.mean(self._all_ep_won)*100:.1f}%")

        tot = self._ep_action_hist.sum() or 1
        print(f"\n  Distribuzione azioni (ultimo episodio):")
        for i, name in enumerate(self.ACTION_NAMES):
            pct = self._ep_action_hist[i] / tot * 100
            bar = '█' * int(pct / 2)
            print(f"    {name:8s}: {pct:5.1f}% {bar}")
        print("═"*60 + "\n")

    def close(self):
        self.writer.flush()
        self.writer.close()
        print("[Monitor] 🔒 Monitor chiuso.")