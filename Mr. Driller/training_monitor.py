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

import os           # Operazioni sul filesystem: creare cartelle, costruire percorsi
import csv          # Lettura e scrittura di file CSV (Comma-Separated Values)
import time         # Usato per ottenere il timestamp corrente (time.strftime)
                    # con cui si nomina la cartella di ogni run di training
import numpy as np  # NumPy: libreria di calcolo numerico. Usata per medie,
                    # convoluzione (smoothing dei grafici) e array di azioni
from collections import deque
# deque (double-ended queue): lista circolare a dimensione massima fissa.
# Quando è piena e si aggiunge un elemento, quello più vecchio viene rimosso automaticamente.
# Usata per le "rolling window" (medie mobili degli ultimi N episodi).


# ─────────────────────────────────────────────────────────────────────
# CSV WRITER
# ─────────────────────────────────────────────────────────────────────

class CSVWriter:
    """
    Scrive metriche su CSV con flush periodico.
    Formato colonne: step, tag, value

    Ogni riga del CSV rappresenta un singolo dato registrato:
    es. (step=1000, tag='Episode/Total_Reward', value=45.3)
    """

    def __init__(self, path: str):
        # Costruttore: apre il file CSV e scrive l'intestazione.
        # path — percorso completo del file CSV da creare (es. "runs/driller_0101_1200/training_log.csv")

        self.path = path        # Salva il percorso per riferimento futuro (es. stampe debug)
        self._buf = []          # Buffer interno: lista di righe da scrivere sul disco.
                                # Le righe vengono accumulate qui invece di scriverle subito,
                                # per ridurre le operazioni I/O sul disco (ogni write è costosa).

        self._file = open(path, 'w', newline='', encoding='utf-8')
        # Apre il file in modalità scrittura ('w' = write, crea o sovrascrive).
        # newline='' evita che Python aggiunga automaticamente un '\r' extra su Windows
        # (il modulo csv gestisce i newline internamente).
        # encoding='utf-8' supporta caratteri speciali nelle etichette (es. emoji nei tag).

        self._csv = csv.writer(self._file)
        # Crea un oggetto csv.writer che formatta i dati come CSV e li scrive sul file.
        # Gestisce automaticamente i separatori di colonna (virgole) e le virgolette.

        self._csv.writerow(['step', 'tag', 'value'])
        # Scrive la riga di intestazione (header) del CSV.
        # step  = numero di step o episodio al momento della misurazione
        # tag   = nome della metrica (es. 'Episode/Total_Reward')
        # value = valore numerico della metrica

        self._file.flush()
        # Forza la scrittura dell'intestazione sul disco subito,
        # così il file è immediatamente apribile e leggibile da Excel/pandas anche durante il training.

        print(f"[Monitor] 📄 Log CSV: {path}")  # Conferma a console il percorso del file creato

    def add_scalar(self, tag: str, value: float, step: int):
        # Aggiunge una metrica scalare al buffer (non scrive subito sul disco).
        # tag   — nome della metrica (es. 'Step/Loss_Mean', 'Episode/Won')
        # value — valore numerico da registrare
        # step  — passo di training o numero episodio associato a questo valore

        self._buf.append((step, tag, float(value)))
        # Crea una tupla (step, tag, value) e la aggiunge alla lista buffer.
        # float(value) assicura che il valore sia sempre un numero decimale
        # (es. converte int 1 in 1.0 per uniformità nel CSV).

        if len(self._buf) >= 200:
            self._flush_buf()
        # Se il buffer ha raggiunto 200 voci, svuotalo sul disco.
        # 200 è un compromesso: abbastanza grande da ridurre le scritture disco,
        # abbastanza piccolo da non perdere dati se il programma crasha.

    def _flush_buf(self):
        # Metodo privato: scrive tutte le righe accumulate nel buffer sul file CSV
        # e poi svuota il buffer.

        for row in self._buf:
            self._csv.writerow(row)
            # Scrive ogni tupla (step, tag, value) come una riga CSV nel file.
            # csv.writer formatta automaticamente es. (1000, 'Episode/Reward', 45.3)
            # come la stringa "1000,Episode/Reward,45.3\n"

        self._buf.clear()       # Svuota la lista buffer (tutte le voci sono state scritte)
        self._file.flush()      # Forza il flush del buffer interno di Python verso il disco
                                # (Python mantiene un proprio buffer interno oltre al nostro)

    def add_histogram(self, *args, **kwargs):
        pass
        # Stub (funzione vuota) per compatibilità con l'API TensorBoard.
        # TensorBoard supporta istogrammi (es. distribuzione dei pesi della rete),
        # ma questa versione CSV non li supporta.
        # *args e **kwargs accettano qualsiasi parametro senza errori.
        # "pass" ignora silenziosamente la chiamata.

    def flush(self):
        self._flush_buf()
        # Metodo pubblico che espone _flush_buf() all'esterno.
        # Chiamato da TrainingMonitor.close() per assicurarsi che nessun dato rimanga nel buffer.

    def close(self):
        self._flush_buf()   # Scrivi tutto ciò che è ancora nel buffer
        self._file.close()  # Chiudi il file (libera il file descriptor del sistema operativo)
        # IMPORTANTE: chiamare sempre close() alla fine del training per evitare perdita di dati.


# ─────────────────────────────────────────────────────────────────────
# TRAINING MONITOR
# ─────────────────────────────────────────────────────────────────────

class TrainingMonitor:
    """
    Monitor centralizzato per training Rainbow DQN.
    Backend: CSV + matplotlib (no TensorBoard, no crash SDL/X11).

    Raccoglie metriche step-by-step e per episodio, le scrive su CSV
    e genera grafici PNG periodicamente o a richiesta.
    """

    ACTION_NAMES = ["IDLE", "LEFT", "RIGHT", "DRILL_L", "DRILL_R", "DRILL_D"]
    # Lista costante dei nomi delle 6 azioni possibili dell'agente.
    # Usata per etichettare le statistiche sulle azioni nei log e nei grafici.
    # L'ordine corrisponde agli indici 0-5 dell'array _ep_action_hist.

    def __init__(self, log_dir="runs/driller_dqn", log_every=50,
                 window=100, plot_every=100, **kwargs):
        # Costruttore del monitor.
        # log_dir    — prefisso del nome della cartella dove salvare i log
        # log_every  — ogni quanti step scrivere le metriche step-level sul CSV
        # window     — dimensione della finestra per le medie mobili (rolling average)
        # plot_every — ogni quanti episodi generare automaticamente i grafici PNG
        # **kwargs   — parametri extra ignorati (compatibilità con versioni precedenti)

        self.log_every  = log_every     # Salva la frequenza di log per gli step
        self.window     = window        # Salva la dimensione della rolling window
        self.plot_every = plot_every    # Salva la frequenza di generazione grafici

        # Crea il nome univoco della cartella di questo run di training.
        # time.strftime('%m%d_%H%M%S') produce es. "0115_143022" (mese-giorno_ora-min-sec)
        # Così ogni run ha la sua cartella distinta e non si sovrascrivono i log precedenti.
        self.run_name  = f"{log_dir}_{time.strftime('%m%d_%H%M%S')}"
        self.plots_dir = os.path.join(self.run_name, "plots")
        # Sottocartella "plots" dentro la cartella del run, dove salvare i PNG

        os.makedirs(self.run_name,  exist_ok=True)
        # Crea la cartella del run (e tutte le cartelle intermedie necessarie).
        # exist_ok=True: non lancia errore se la cartella esiste già.

        os.makedirs(self.plots_dir, exist_ok=True)
        # Crea la sottocartella plots/ dentro il run

        # Crea il file CSV e il writer nella cartella del run
        csv_path    = os.path.join(self.run_name, "training_log.csv")
        self.writer = CSVWriter(csv_path)   # Istanza del CSVWriter appena definito
        self.csv_path = csv_path            # Salva il percorso per riferimento (es. print finale)

        # ── Stato dell'episodio corrente ────────────────────────────────
        # Questi accumulatori vengono azzerati a ogni inizio di episodio
        # e popolati step-per-step durante l'episodio.

        self._ep_reward       = 0.0     # Somma dei reward ricevuti in questo episodio
        self._ep_steps        = 0       # Numero di step eseguiti in questo episodio
        self._ep_max_depth    = 0       # Profondità massima raggiunta (riga più in basso toccata)
        self._ep_min_oxy      = 100.0   # Ossigeno minimo raggiunto (il momento più critico)
        self._ep_wall_hits    = 0       # Numero di volte che l'agente ha colpito un muro X
        self._ep_delayed_hits = 0       # Numero di blocchi Delayed colpiti
        self._ep_action_hist  = np.zeros(6, dtype=np.int32)
        # Array NumPy di 6 interi inizializzati a 0: conta quante volte ogni azione è stata eseguita.
        # dtype=np.int32: usa interi a 32 bit per efficienza di memoria

        # ── Rolling windows (medie mobili degli ultimi N episodi) ──────
        # Ogni deque mantiene al massimo 'window' valori recenti.
        # Quando è piena, il valore più vecchio viene rimosso automaticamente.

        self._ep_rewards    = deque(maxlen=window)   # Reward totali degli ultimi N episodi
        self._ep_depths     = deque(maxlen=window)   # Profondità massime degli ultimi N episodi
        self._ep_lengths    = deque(maxlen=window)   # Lunghezze (in step) degli ultimi N episodi
        self._ep_won        = deque(maxlen=window)   # 1.0 se vinto, 0.0 se perso (ultimi N episodi)
        self._ep_wall_rates = deque(maxlen=window)   # Tasso di colpi a muro (ultimi N episodi)

        # ── Storico completo di tutti gli episodi (per i grafici) ──────
        # Queste liste crescono indefinitamente durante il training.
        # Contengono tutti i dati dall'inizio, non solo gli ultimi N.
        # Usate da plot_all() per disegnare le curve complete.

        self._all_ep_rewards = []   # Lista di tutti i reward totali per episodio
        self._all_ep_depths  = []   # Lista di tutte le profondità massime per episodio
        self._all_ep_lengths = []   # Lista di tutte le lunghezze per episodio
        self._all_ep_won     = []   # Lista di tutti i risultati (1.0/0.0) per episodio

        # ── Step buffers (medie mobili per le metriche step-level) ─────
        # Questi deque raccolgono i valori degli ultimi N step per calcolare
        # medie da loggare periodicamente (ogni log_every step).

        self._losses  = deque(maxlen=100)   # Ultimi 100 valori di loss (errore di training)
        self._q_means = deque(maxlen=100)   # Ultimi 100 valori medi dei Q-values
        self._rewards = deque(maxlen=100)   # Ultimi 100 reward step-level

        self._current_lr  = None    # Learning rate corrente (aggiornato quando ricevuto da log_step)
        self._current_mem = None    # Dimensione corrente del replay buffer
        self.episode_count = 0      # Contatore globale degli episodi completati

        # Messaggi di avvio a console per confermare la configurazione
        print(f"[Monitor] 🚀 TrainingMonitor avviato (CSV+matplotlib)")
        print(f"[Monitor] 📁 Run: {self.run_name}")
        print(f"[Monitor] 📊 Grafici PNG aggiornati ogni {plot_every} episodi")

    # ──────────────────────────────────────────────────────────────────
    # LOG STEP
    # ──────────────────────────────────────────────────────────────────

    def log_step(self, reward=0.0, action_idx=0, depth=0, oxy=100.0,
                 loss=None, q_mean=None, epsilon=None, steps_done=0,
                 was_hard_block=False, was_delayed_block=False,
                 lr=None, mem_size=None, player_x=None, player_y=None,
                 q_values=None, **kwargs):
        """Chiamare ogni step AI (FASE ACTION e FASE LEARNING)."""
        # Questo metodo è il "cuore" del monitor: viene chiamato ad ogni singolo step
        # dell'agente per accumulare dati sull'episodio corrente.
        # Tutti i parametri hanno valori di default per permettere chiamate parziali.

        # Alias retrocompatibilità q_values → q_mean
        if q_mean is None and q_values is not None:
            q_mean = q_values
        # Alcune versioni di main.py passano il parametro come 'q_values',
        # altre come 'q_mean'. Questo if normalizza i due nomi.

        # ── Accumulo metriche episodio ─────────────────────────────────
        self._ep_reward    += reward                                # Somma il reward di questo step
        self._ep_steps     += 1                                     # Conta un altro step nell'episodio
        self._ep_max_depth  = max(self._ep_max_depth, depth)        # Aggiorna la profondità massima
        self._ep_min_oxy    = min(self._ep_min_oxy, oxy)            # Aggiorna l'ossigeno minimo (il peggio)

        if was_hard_block:    self._ep_wall_hits    += 1   # Conta i colpi a blocchi indistruttibili X
        if was_delayed_block: self._ep_delayed_hits += 1   # Conta i colpi a blocchi ritardati

        if 0 <= action_idx < 6:
            self._ep_action_hist[action_idx] += 1
            # Incrementa il contatore dell'azione eseguita.
            # Il check 0 <= action_idx < 6 evita IndexError per indici fuori range.

        # ── Aggiornamento buffer step-level ───────────────────────────
        self._rewards.append(reward)    # Aggiunge il reward corrente al buffer degli ultimi 100

        if loss is not None:
            self._losses.append(float(loss))
            # Aggiunge la loss solo se disponibile (None durante i primi step
            # prima che il training inizi, o quando non c'è stato un update)

        if q_mean is not None:
            val = float(q_mean.mean()) if hasattr(q_mean, 'mean') else float(q_mean)
            # Se q_mean è un tensore o array NumPy (ha il metodo .mean()), calcola la media.
            # Altrimenti (è già uno scalare), lo converte direttamente in float.
            self._q_means.append(val)

        if lr is not None:       self._current_lr  = lr        # Aggiorna il learning rate corrente
        if mem_size is not None: self._current_mem = mem_size  # Aggiorna la dimensione del replay buffer

        if steps_done > 0 and steps_done % self.log_every == 0:
            self._flush_step(steps_done, epsilon)
            # Ogni log_every step (es. ogni 50 step), scrivi le medie accumulate sul CSV.
            # steps_done % log_every == 0: il modulo è 0 solo ai multipli esatti (50, 100, 150, ...)
            # steps_done > 0: evita di loggare allo step 0 (prima di qualsiasi azione)

    def _flush_step(self, step: int, epsilon):
        # Metodo privato: calcola le medie dei buffer step-level e le scrive sul CSV.
        # Chiamato da log_step() ogni log_every step.
        # step    — numero di step globale corrente
        # epsilon — valore corrente di epsilon (per la politica ε-greedy dell'agente)

        w = self.writer     # Alias locale per scrivere meno codice

        # Per ogni metrica, scrivi la media solo se il buffer non è vuoto
        if self._rewards: w.add_scalar('Step/Reward_Mean',  np.mean(self._rewards), step)
        if self._losses:  w.add_scalar('Step/Loss_Mean',    np.mean(self._losses),  step)
        if self._q_means: w.add_scalar('Step/Q_Mean',       np.mean(self._q_means), step)
        # np.mean() calcola la media aritmetica di tutti i valori nel deque

        if epsilon is not None:
            w.add_scalar('Step/Epsilon', epsilon, step)
            # Epsilon è il parametro di esplorazione: parte alto (es. 1.0 = esplora sempre casualmente)
            # e scende nel tempo (es. 0.01 = segue quasi sempre la policy appresa)

        if self._current_lr is not None:
            w.add_scalar('Step/Learning_Rate', self._current_lr, step)
            # Il learning rate può cambiare durante il training (es. con uno scheduler)

        if self._current_mem is not None:
            w.add_scalar('Step/Memory_Size', self._current_mem, step)
            # Traccia quanto si riempie il replay buffer nel tempo

        # ── Distribuzione delle azioni ─────────────────────────────────
        tot = self._ep_action_hist.sum() or 1
        # Somma totale di tutte le azioni dell'episodio.
        # "or 1" evita la divisione per zero se non c'è ancora nessuna azione registrata.

        for i, name in enumerate(self.ACTION_NAMES):
            w.add_scalar(f'Actions/{name}_pct', self._ep_action_hist[i] / tot * 100, step)
            # Calcola la percentuale di utilizzo di ogni azione (0-100%) e la logga.
            # Es. se IDLE è stata usata 30 volte su 100 totali → 30.0%

        # ── Metriche patologiche (comportamenti indesiderati) ──────────
        drill_d = int(self._ep_action_hist[5])
        # Conteggio azione DRILL_D (perforazione verso il basso, indice 5)

        drill_l = int(self._ep_action_hist[3]) + int(self._ep_action_hist[4])
        # Conteggio azioni DRILL_L + DRILL_R (perforazioni laterali, indici 3 e 4)

        w.add_scalar('Pathology/DrillDown_ratio', drill_d / max(drill_l + drill_d, 1), step)
        # Rapporto di perforazione verso il basso rispetto a tutte le perforazioni.
        # Alto = l'agente scende prevalentemente verso il basso (buono in Mr. Driller).
        # max(..., 1) evita divisione per zero se non c'è ancora alcuna perforazione.

        w.add_scalar('Pathology/IDLE_pct',    self._ep_action_hist[0] / tot * 100, step)
        # Percentuale di azioni IDLE. Alta = l'agente è pigro/bloccato (comportamento indesiderato).

        w.add_scalar('Pathology/WallHits',    self._ep_wall_hits,    step)
        w.add_scalar('Pathology/DelayedHits', self._ep_delayed_hits, step)
        # Conteggi assoluti delle collisioni con blocchi speciali

    # ──────────────────────────────────────────────────────────────────
    # LOG EPISODE
    # ──────────────────────────────────────────────────────────────────

    def log_episode(self, steps_done=0, level_id=0, won=False,
                    death_cause="unknown", depth=0, final_oxy=None,
                    score=0, lives_left=0, **kwargs):
        """Chiamare su morte o fine livello."""
        # Questo metodo viene chiamato una volta alla fine di ogni episodio
        # (quando il giocatore muore o completa il livello).
        # Finalizza le metriche dell'episodio e le scrive sul CSV.

        self.episode_count += 1     # Incrementa il contatore globale degli episodi
        ep = self.episode_count     # Alias locale per brevità
        w  = self.writer            # Alias locale per brevità

        ep_depth = max(self._ep_max_depth, depth)
        # Prende il massimo tra la profondità accumulata negli step e quella
        # passata come parametro (depth al momento della morte/vittoria).
        # I due possono differire se l'ultimo step non ha aggiornato _ep_max_depth.

        ep_min_oxy = self._ep_min_oxy
        if final_oxy is not None:
            ep_min_oxy = min(ep_min_oxy, final_oxy)
        # Considera anche l'ossigeno al momento della morte come possibile minimo

        # ── Aggiornamento rolling windows e storico ────────────────────
        self._ep_rewards.append(self._ep_reward)    # Aggiunge il reward dell'episodio alla rolling window
        self._ep_depths.append(ep_depth)            # Aggiunge la profondità massima alla rolling window
        self._ep_lengths.append(self._ep_steps)     # Aggiunge la lunghezza (in step) alla rolling window
        self._ep_won.append(float(won))             # Aggiunge 1.0 (vittoria) o 0.0 (sconfitta)

        wall_rate = self._ep_wall_hits / max(self._ep_steps, 1)
        # Tasso di colpi a muro: quanti colpi X per step.
        # max(..., 1) evita divisione per zero se l'episodio è durato 0 step.
        self._ep_wall_rates.append(wall_rate)

        # Aggiunge gli stessi dati allo storico completo (liste che crescono senza limite)
        self._all_ep_rewards.append(self._ep_reward)
        self._all_ep_depths.append(ep_depth)
        self._all_ep_lengths.append(self._ep_steps)
        self._all_ep_won.append(float(won))

        # ── Scrittura metriche episodio sul CSV ────────────────────────
        # Ogni add_scalar scrive una riga: (episodio, tag, valore)
        w.add_scalar('Episode/Total_Reward',  self._ep_reward, ep)   # Reward cumulativo episodio
        w.add_scalar('Episode/Length_Steps',  self._ep_steps,  ep)   # Durata in step
        w.add_scalar('Episode/Max_Depth',     ep_depth,        ep)   # Profondità massima raggiunta
        w.add_scalar('Episode/Min_Oxygen',    ep_min_oxy,      ep)   # Ossigeno minimo (momento peggiore)
        w.add_scalar('Episode/Level_ID',      level_id,        ep)   # Quale livello è stato giocato
        w.add_scalar('Episode/Won',           float(won),      ep)   # 1.0 = vittoria, 0.0 = sconfitta
        w.add_scalar('Episode/WallBash_Rate', wall_rate,       ep)   # Frequenza colpi a muro
        w.add_scalar('Episode/Score',         score,           ep)   # Punteggio finale del giocatore

        # ── Metriche sui pattern di perforazione ──────────────────────
        drill_d = int(self._ep_action_hist[5])
        drill_l = int(self._ep_action_hist[3]) + int(self._ep_action_hist[4])
        drill_ratio = drill_d / max(drill_l + drill_d, 1)
        # Rapporto drill_down / (drill_down + drill_laterale)
        # Valore 1.0 = l'agente perfora sempre verso il basso (ottimale)
        # Valore 0.0 = l'agente perfora solo lateralmente (subottimale)
        w.add_scalar('Episode/DrillDown_ratio', drill_ratio, ep)

        tot        = self._ep_action_hist.sum() or 1
        idle_ratio = self._ep_action_hist[0] / tot * 100
        # Percentuale di azioni IDLE sull'episodio (0-100%).
        # Alta = agente pigro o bloccato. Utile per rilevare comportamenti degeneri.
        w.add_scalar('Episode/IDLE_pct', idle_ratio, ep)

        # Distribuzione percentuale di tutte le azioni per questo episodio
        for i, name in enumerate(self.ACTION_NAMES):
            w.add_scalar(f'EpActions/{name}_pct', self._ep_action_hist[i] / tot * 100, ep)

        # ── Rolling averages (scritte sul CSV solo se la window è abbastanza piena) ──
        if len(self._ep_rewards) >= 5:
            # Scrivi le medie mobili solo dopo almeno 5 episodi per avere dati significativi
            w.add_scalar('Rolling/Reward_Avg',   np.mean(self._ep_rewards),    ep)  # Media reward ultimi N episodi
            w.add_scalar('Rolling/Depth_Avg',    np.mean(self._ep_depths),     ep)  # Media profondità
            w.add_scalar('Rolling/WinRate',      np.mean(self._ep_won),        ep)  # Win rate (0.0-1.0)
            w.add_scalar('Rolling/WallRate_Avg', np.mean(self._ep_wall_rates), ep)  # Media tasso colpi muro

        # ── Causa di morte codificata come numero ─────────────────────
        cause_map = {"oxy": 0, "block_fall": 1, "unknown": 2, "win": 3}
        # Mappa le cause di morte/fine episodio a valori numerici per il CSV
        # (il CSV non supporta stringhe come valori in modo efficiente per i grafici)
        w.add_scalar('Death/Cause_code', cause_map.get(death_cause, 2), ep)
        # .get(death_cause, 2): se la causa non è nella mappa, usa 2 ("unknown") come default

        # ── Stampa a console ──────────────────────────────────────────
        avg_r  = f"{np.mean(self._ep_rewards):+.1f}" if self._ep_rewards else "---"
        # Formatta la media reward con segno (+/-) e 1 decimale.
        # "---" se non ci sono ancora dati nella rolling window.

        status = "WIN" if won else f"DEAD({death_cause})"
        # Stringa di stato: "WIN" se ha vinto, "DEAD(oxy)" o "DEAD(block_fall)" se morto

        print(
            f"[EP {ep:5d}|Step {steps_done:7d}] "
            # ep:5d = numero episodio con padding a 5 cifre (es. "    1", "  100", "10000")
            # steps_done:7d = step globale con padding a 7 cifre
            f"R:{self._ep_reward:+8.2f}  Avg:{avg_r:>8}  "
            # :+8.2f = reward con segno, 8 caratteri totali, 2 decimali
            # :>8 = media allineata a destra in 8 caratteri
            f"Depth:{ep_depth:3d}  Oxy:{int(ep_min_oxy):3d}%  "
            f"IDLE:{idle_ratio:4.1f}%  DrillD:{drill_ratio*100:4.1f}%  "
            f"Lv{level_id} {status}"
        )

        # ── Grafici automatici periodici ──────────────────────────────
        if ep % self.plot_every == 0:
            self.plot_all(silent=True)
            # Ogni plot_every episodi (es. ogni 500), genera automaticamente i PNG.
            # silent=True: non stampa il percorso di ogni grafico (già tanti log in console).

        # ── Reset accumulatori per il prossimo episodio ───────────────
        # IMPORTANTE: questi reset devono avvenire DOPO aver usato i dati,
        # cioè dopo le scritture CSV e la stampa console.
        self._ep_reward       = 0.0
        self._ep_steps        = 0
        self._ep_max_depth    = 0
        self._ep_min_oxy      = 100.0
        self._ep_wall_hits    = 0
        self._ep_delayed_hits = 0
        self._ep_action_hist  = np.zeros(6, dtype=np.int32)
        # np.zeros(6) crea un nuovo array di 6 zeri: ogni contatore azione riparte da 0

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
            matplotlib.use('Agg')
            # Seleziona il backend "Agg" (Anti-Grain Geometry) di matplotlib.
            # Agg è un backend non-interattivo: renderizza in memoria senza aprire
            # finestre grafiche. FONDAMENTALE perché pygame ha già una finestra aperta
            # e aprirne un'altra con matplotlib causerebbe conflitti SDL/X11.
            # DEVE essere chiamato PRIMA di importare matplotlib.pyplot.

            import matplotlib.pyplot as plt
            # Importa l'interfaccia di alto livello per creare grafici.
            # Importato qui (non in cima al file) per due motivi:
            # 1. matplotlib è opzionale: se non è installato, il programma funziona lo stesso
            # 2. Il backend va impostato prima dell'import di pyplot

        except ImportError:
            if not silent:
                print("[Monitor] ⚠️  matplotlib non disponibile.")
                print("[Monitor] 💡 Installa con: pip install matplotlib")
            return  # Esce senza errori: i grafici sono opzionali

        out = output_dir or self.plots_dir
        # Usa la cartella specificata, oppure quella di default del run.
        # "or" in Python: se output_dir è None (falsy), usa self.plots_dir.

        os.makedirs(out, exist_ok=True)  # Crea la cartella se non esiste

        def smooth(data, k=20):
            # Funzione interna di smoothing con media mobile.
            # data — lista di valori da lisciare
            # k    — dimensione della finestra di smoothing (default 20)
            if len(data) < k:
                return list(data)
                # Se i dati sono meno di k, non ha senso fare smoothing: restituisce i dati originali

            return list(np.convolve(data, np.ones(k)/k, mode='valid'))
            # np.convolve(data, kernel, mode='valid') applica una convoluzione:
            # il kernel np.ones(k)/k è una finestra rettangolare normalizzata (media mobile).
            # mode='valid' restituisce solo i valori dove il kernel si sovrappone completamente ai dati
            # (quindi la lista risultante ha len(data)-k+1 elementi, cioè è più corta di k-1 elementi).

        # ── Grafico 1: Reward per episodio ─────────────────────────────
        if self._all_ep_rewards:
            fig, ax = plt.subplots(figsize=(12, 4))
            # Crea una figura di 12×4 pollici con un solo sottoplot (ax).
            # fig = oggetto figura (gestisce le dimensioni, il salvataggio)
            # ax  = oggetto assi (dove si disegnano le curve)

            r = self._all_ep_rewards
            ax.plot(r, alpha=0.25, color='steelblue', lw=0.8, label='Per episodio')
            # Disegna la curva grezza dei reward.
            # alpha=0.25 = 25% opacità (quasi trasparente, si vede ma non copre lo smoothing)
            # lw=0.8 = linewidth sottile (linea sottile per dati rumorosi)

            if len(r) >= 20:
                ax.plot(range(19, len(r)), smooth(r, 20), color='steelblue',
                        lw=2, label='Media (20 ep)')
                # Disegna la curva smoothed sopra quella grezza.
                # range(19, len(r)): l'asse x parte da 19 perché smooth() con k=20
                # produce k-1=19 elementi in meno (mode='valid').
                # lw=2 = linea più spessa per la curva smoothed (più visibile)

            ax.axhline(0, color='gray', lw=0.5, ls='--')
            # Disegna una linea orizzontale tratteggiata grigia a y=0
            # come riferimento visivo (reward positivo = sopra, negativo = sotto)

            ax.set_title('Reward per Episodio')
            ax.set_xlabel('Episodio'); ax.set_ylabel('Reward')
            ax.legend(); ax.grid(True, alpha=0.3)
            # alpha=0.3 per la griglia: visibile ma non invadente

            plt.tight_layout()          # Ottimizza automaticamente i margini del grafico
            p = os.path.join(out, 'reward_curve.png')
            plt.savefig(p, dpi=120)     # Salva il PNG con risoluzione 120 DPI (buona qualità)
            plt.close()                 # IMPORTANTE: chiude la figura per liberare memoria
                                        # (senza close(), le figure si accumulano in RAM)
            if not silent: print(f"[Monitor] 📈 {p}")

        # ── Grafico 2: Profondità massima per episodio ─────────────────
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

        # ── Grafico 3: Win rate rolling ────────────────────────────────
        if self._all_ep_won:
            fig, ax = plt.subplots(figsize=(12, 4))
            wn = self._all_ep_won; k = 20
            wr = [np.mean(wn[max(0,i-k):i+1]) * 100 for i in range(len(wn))]
            # Calcola il win rate rolling per ogni episodio i:
            # prende la finestra [max(0, i-k) : i+1] (ultimi k episodi fino a i)
            # calcola la media (0.0-1.0) e moltiplica per 100 per avere la percentuale.
            # max(0, i-k) evita indici negativi all'inizio (prima di avere k episodi).

            ax.plot(wr, color='gold', lw=2)
            ax.fill_between(range(len(wr)), wr, alpha=0.25, color='gold')
            # fill_between riempie l'area sotto la curva con il colore gold semitrasparente.
            # range(len(wr)) = asse x (0, 1, 2, ..., numero episodi)
            # wr = asse y (valori win rate)
            # alpha=0.25 = 25% opacità per l'area riempita

            ax.set_title(f'Win Rate Rolling (finestra {k} ep)')
            ax.set_xlabel('Episodio'); ax.set_ylabel('Win Rate (%)')
            ax.set_ylim(0, 100)         # Fissa l'asse Y tra 0% e 100%
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            p = os.path.join(out, 'win_rate.png')
            plt.savefig(p, dpi=120); plt.close()
            if not silent: print(f"[Monitor] 📈 {p}")

        # ── Grafico 4: Distribuzione azioni (torta) ────────────────────
        if self._ep_action_hist.sum() > 0:
            # Genera il grafico solo se c'è almeno una azione registrata
            fig, ax = plt.subplots(figsize=(7, 5))   # Figura quadrata per il pie chart
            sizes  = self._ep_action_hist.astype(float)
            # Converte l'array di interi in float per le percentuali

            labels = [n for n, s in zip(self.ACTION_NAMES, sizes) if s > 0]
            # Prende solo le etichette delle azioni che sono state usate almeno una volta
            # (zip accoppia nomi e conteggi, il filtro if s > 0 esclude le azioni con 0 usi)

            vals   = [s for s in sizes if s > 0]
            # Analogamente, solo i valori > 0

            colors = ['#e74c3c','#3498db','#2ecc71','#9b59b6','#f39c12','#1abc9c']
            # 6 colori distinti per le 6 possibili azioni (rosso, blu, verde, viola, arancione, turchese)

            ax.pie(vals, labels=labels, autopct='%1.1f%%', colors=colors[:len(vals)])
            # Disegna il grafico a torta.
            # autopct='%1.1f%%' mostra la percentuale su ogni fetta con 1 decimale (es. "35.2%")
            # colors[:len(vals)] usa solo tanti colori quante sono le azioni presenti

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
        # Stampa un riepilogo testuale del training a console.
        # Chiamato tipicamente alla fine del training per una panoramica rapida.

        print("\n" + "═"*60)      # Linea di separazione decorativa
        print("  🏁  RIEPILOGO FINALE TRAINING")
        print("═"*60)
        print(f"  Episodi totali       : {self.episode_count}")

        if self._all_ep_depths:
            print(f"  Profondità massima   : {max(self._all_ep_depths)}")       # Record assoluto
            print(f"  Profondità media     : {np.mean(self._all_ep_depths):.1f}") # Media su tutti gli episodi

        if self._all_ep_rewards:
            print(f"  Reward media totale  : {np.mean(self._all_ep_rewards):.1f}")

        if self._ep_rewards:
            print(f"  Reward ultima window : {np.mean(self._ep_rewards):.1f}")
            # np.mean sull'ultima rolling window (ultimi N episodi):
            # indica se recentemente l'agente sta migliorando

        if self._all_ep_won:
            print(f"  Win rate totale      : {np.mean(self._all_ep_won)*100:.1f}%")
            # Media di tutti i float 0.0/1.0 → percentuale vittorie sull'intera storia

        tot = self._ep_action_hist.sum() or 1
        print(f"\n  Distribuzione azioni (ultimo episodio):")
        for i, name in enumerate(self.ACTION_NAMES):
            pct = self._ep_action_hist[i] / tot * 100
            bar = '█' * int(pct / 2)
            # Genera una barra ASCII: ogni █ rappresenta il 2% di utilizzo.
            # Es. 40% → 20 blocchi █. int() tronca (non arrotonda) per semplicità.
            print(f"    {name:8s}: {pct:5.1f}% {bar}")
            # :8s = nome azione con padding a 8 caratteri (allineamento colonne)
            # :5.1f = percentuale con 1 decimale e 5 caratteri totali

        print("═"*60 + "\n")

    def close(self):
        # Chiude correttamente il monitor: svuota il buffer CSV e chiude il file.
        # DEVE essere chiamato alla fine del training per non perdere dati.
        self.writer.flush()     # Scrive le ultime metriche nel buffer sul disco
        self.writer.close()     # Chiude il file CSV
        print("[Monitor] 🔒 Monitor chiuso.")