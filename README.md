# Mr. Driller AI — Rainbow DQN + LLM Agents

Progetto di Reinforcement Learning applicato al gioco **Mr. Driller**: un agente Rainbow DQN (con Dueling, NoisyNets, Prioritized Replay, N-step return) impara a giocare partendo da zero. Il repository include anche due agenti alternativi basati su LLM (OpenRouter) e un sistema di benchmark/valutazione comparativa.

---

## Indice

- [Struttura del progetto](#struttura-del-progetto)
- [Requisiti di sistema](#requisiti-di-sistema)
- [Installazione](#installazione)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
- [Configurazione](#configurazione)
- [Utilizzo](#utilizzo)
  - [Modalità PLAY (demo visiva)](#modalità-play-demo-visiva)
  - [Modalità TRAINING (addestramento)](#modalità-training-addestramento)
  - [Benchmark LLM](#benchmark-llm)
  - [Valutazione comparativa agenti](#valutazione-comparativa-agenti)
- [File principali](#file-principali)
- [Parametri configurabili](#parametri-configurabili)
- [Output e salvataggi](#output-e-salvataggi)
- [Risoluzione problemi](#risoluzione-problemi)

---

## Struttura del progetto

```
.
├── main.py                  # Entry point: loop di gioco e training Rainbow DQN
├── ai_agent.py              # Architettura rete, replay buffer, reward function
├── llm_agent.py             # Agente LLM (selezione azioni via OpenRouter)
├── llm_reward_model.py      # Reward model LLM (stima reward via OpenRouter)
├── benchmark.py             # Benchmark comparativo tra modelli LLM
├── evaluate_agents.py       # Valutazione comparativa DQN vs LLM (headless)
├── training_monitor.py      # Monitor CSV + grafici matplotlib
├── character.py             # Classe Character (giocatore)
├── block.py                 # Classi blocchi (Classic, Pill, Solo, Delayed, ...)
├── menu.py                  # Menu, restart, cambio livello
├── level.py                 # Generazione livelli
├── eventHandling.py         # Gestione input (movimento, trivellamento)
├── Assets/                  # Sprite, font, musica
│   ├── Misc/
│   │   ├── oxyAnim/         # 101 frame animazione ossigeno (0.png … 100.png)
│   │   ├── police/          # Font personalizzato
│   │   └── userinterface.png
│   └── Music/               # Tracce audio (Level1.wav … Level10.wav, menu.wav)
└── RenfLearningAgent/       # Checkpoint e buffer (creati automaticamente)
    ├── driller_ai_v31.pth   # Pesi della rete
    └── driller_buffer_v31/  # Buffer di replay persistente
```

---

## Requisiti di sistema

| Componente | Minimo | Consigliato |
|---|---|---|
| Python | 3.10 | 3.11 o 3.12 |
| RAM | 4 GB | 8 GB |
| GPU | — (CPU ok) | NVIDIA con CUDA 11.8+ |
| Spazio disco | 500 MB | 2 GB (buffer replay) |

### Dipendenze Python

```
pygame>=2.5
torch>=2.1          # CPU o CUDA — vedi note installazione
numpy>=1.24
httpx[http2]>=0.25  # Solo per agenti LLM
matplotlib>=3.7     # Solo per grafici monitor
```

---

## Installazione

### Windows

#### 1. Installa Python 3.11

Scarica da [python.org](https://www.python.org/downloads/). Durante l'installazione spunta **"Add Python to PATH"**.

Verifica l'installazione:
```cmd
python --version
pip --version
```

#### 2. Crea e attiva un ambiente virtuale

```cmd
python -m venv venv
venv\Scripts\activate
```

Il prompt cambierà in `(venv) C:\...>`.

#### 3. Installa PyTorch

**Solo CPU:**
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Con GPU NVIDIA (CUDA 12.1):**
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verifica:
```cmd
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

#### 4. Installa le restanti dipendenze

```cmd
pip install pygame numpy httpx[http2] matplotlib
```

#### 5. Note specifiche Windows

- Se `pygame` non trova le librerie audio, installa il redistributable Visual C++: [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe).
- Se il gioco si avvia in modalità headless per errore, verifica che `PLAY_MODE = True` in `main.py`.
- Il percorso del buffer usa `/` come separatore ma Python lo gestisce correttamente su Windows. Non modificare manualmente i percorsi nei file `.npy`.

---

### macOS

#### 1. Installa Python 3.11 tramite Homebrew

```bash
# Installa Homebrew se non presente
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python@3.11
```

Oppure scarica il pkg da [python.org](https://www.python.org/downloads/macos/).

Verifica:
```bash
python3.11 --version
```

#### 2. Crea e attiva un ambiente virtuale

```bash
python3.11 -m venv venv
source venv/bin/activate
```

#### 3. Installa PyTorch

**Apple Silicon (M1/M2/M3) — con accelerazione MPS:**
```bash
pip install torch torchvision
```
PyTorch rileva automaticamente MPS. Verifica:
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

**Intel Mac — solo CPU:**
```bash
pip install torch torchvision
```

#### 4. Installa le restanti dipendenze

```bash
pip install pygame numpy "httpx[http2]" matplotlib
```

#### 5. Note specifiche macOS

- Su macOS Ventura e successivi, la prima apertura di una finestra Pygame può richiedere di concedere i permessi di accessibilità: vai in **Impostazioni di sistema → Privacy e sicurezza → Accessibilità** e aggiungi il terminale.
- Se ricevi `OMP: Error #15`, la variabile `KMP_DUPLICATE_LIB_OK=True` è già impostata in `main.py`.
- Il training su MPS è supportato ma alcune operazioni di PyTorch ricadono su CPU. Usa `device = torch.device("mps")` solo se la versione di PyTorch è ≥ 2.1.
- Su Apple Silicon, se ricevi errori audio da SDL, aggiungi `export SDL_AUDIODRIVER=dummy` nel tuo `.zshrc` o lancia il programma con:
  ```bash
  SDL_AUDIODRIVER=dummy python main.py
  ```

---

### Linux

#### 1. Installa Python 3.11 e dipendenze di sistema

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip \
     libsdl2-dev libsdl2-mixer-dev libsdl2-image-dev \
     libportmidi-dev libjpeg-dev libfreetype6-dev git
```

**Fedora/RHEL:**
```bash
sudo dnf install python3.11 python3.11-devel SDL2-devel SDL2_mixer-devel \
     SDL2_image-devel portmidi-devel libjpeg-devel freetype-devel
```

**Arch Linux:**
```bash
sudo pacman -S python sdl2 sdl2_mixer sdl2_image portmidi freetype2
```

Verifica:
```bash
python3.11 --version
```

#### 2. Crea e attiva un ambiente virtuale

```bash
python3.11 -m venv venv
source venv/bin/activate
```

#### 3. Installa PyTorch

**Solo CPU:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Con GPU NVIDIA (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Con GPU AMD (ROCm 6.0, solo Linux):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

Verifica CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

#### 4. Installa le restanti dipendenze

```bash
pip install pygame numpy "httpx[http2]" matplotlib
```

#### 5. Training headless su server senza display

Su server senza schermo (cloud, Colab, WSL senza GUI), il training headless funziona già grazie alle variabili `SDL_VIDEODRIVER=dummy` e `SDL_AUDIODRIVER=dummy` impostate automaticamente in `main.py` quando `PLAY_MODE = False`.

Non è necessario installare Xvfb o VirtualGL per il training.

Se vuoi eseguire la modalità PLAY su una macchina remota tramite SSH con X11 forwarding:
```bash
ssh -X utente@server
python main.py   # PLAY_MODE = True in main.py
```

---

## Configurazione

### Variabili d'ambiente (per agenti LLM)

Gli agenti LLM richiedono una chiave API di [OpenRouter](https://openrouter.ai).

**Windows (Command Prompt):**
```cmd
set OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxx
```

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxxxxxxxxxx"
```

Per renderla permanente in Windows: cerca "Variabili d'ambiente" nel menu Start e aggiungila nelle variabili utente.

**macOS/Linux (sessione corrente):**
```bash
export OPENROUTER_API_KEY="sk-or-xxxxxxxxxxxxxxxxxxxxxxxx"
```

**macOS/Linux (permanente):**
```bash
# Aggiungi a ~/.zshrc (macOS) o ~/.bashrc (Linux)
echo 'export OPENROUTER_API_KEY="sk-or-xxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.zshrc
source ~/.zshrc
```

### Parametri principali in `main.py`

| Variabile | Default | Descrizione |
|---|---|---|
| `PLAY_MODE` | `True` | `True` = demo visiva; `False` = training |
| `MODEL_PATH` | `RenfLearningAgent/driller_ai_v31.pth` | Percorso checkpoint |
| `BATCH_SIZE` | `128` | Dimensione batch per la backpropagation |
| `GAMMA` | `0.99` | Fattore di sconto |
| `LEARNING_RATE` | `2e-5` | Tasso di apprendimento AdamW |
| `N_STEP` | `8` | Passi per il N-step return |
| `MEMORY_SIZE` | `50000` | Capienza replay buffer |
| `WARMUP_STEPS` | `2000` | Step prima di iniziare il training |

---

## Utilizzo

### Modalità PLAY (demo visiva)

Imposta `PLAY_MODE = True` in `main.py` (valore di default), poi:

```bash
python main.py
```

La finestra 800×600 si apre e l'AI gioca autonomamente usando i pesi in `RenfLearningAgent/driller_ai_v31.pth`. Se il file non esiste, la rete parte con pesi casuali.

### Modalità TRAINING (addestramento)

Imposta `PLAY_MODE = False` in `main.py`:

```python
PLAY_MODE = False
```

Poi lancia:
```bash
python main.py
```

Il training è headless (nessuna finestra). L'output è testuale nel terminale. I checkpoint vengono salvati ogni 1000 step in `RenfLearningAgent/driller_ai_v31.pth`. Il replay buffer viene salvato ogni 5000 step in `RenfLearningAgent/driller_buffer_v31/`.

Per riprendere un training interrotto, rilancia semplicemente il comando: il checkpoint viene caricato automaticamente.

Per interrompere il training in modo sicuro (salvataggio automatico):
```
CTRL+C
```

### Benchmark LLM

Confronta le performance dei modelli `llama` e `deepseek` sui task azione e reward:

```bash
# Assicurati che OPENROUTER_API_KEY sia impostata
python benchmark.py --episodes 30
```

Opzioni disponibili:
```bash
python benchmark.py --episodes 50 --models llama deepseek --task both
python benchmark.py --episodes 20 --task action   # solo task azione
python benchmark.py --episodes 20 --task reward   # solo task reward
python benchmark.py --episodes 10 --consistency-repeats 5
```

L'output è una tabella ASCII con latenza, cache hit rate, error rate, consistenza e score composito.

### Valutazione comparativa agenti

Esegui partite reali con i tre agenti (`dqn`, `llm_act`, `llm_rew`) e confronta le loro performance:

```bash
python evaluate_agents.py --episodes 20 --agents dqn llm_act llm_rew
```

Opzioni disponibili:
```bash
# Solo DQN, 50 episodi, output verboso
python evaluate_agents.py --episodes 50 --agents dqn --verbose

# Solo LLM action, con modello deepseek
python evaluate_agents.py --agents llm_act --llm_model deepseek --episodes 10

# Checkpoint DQN personalizzato
python evaluate_agents.py --dqn_path mio_modello.pth --episodes 30

# Controlla il timeout per episodi bloccati (default 80 iterazioni)
python evaluate_agents.py --stuck_steps 120 --episodes 20
```

I risultati vengono salvati in `eval_results/` come file CSV, JSON e PNG.

---

## File principali

| File | Scopo |
|---|---|
| `main.py` | Entry point. Contiene il loop principale, CurriculumManager, SmartFrameStacker e tutta la logica di training. |
| `ai_agent.py` | Definisce la rete `DrillerDuelingDQN`, `NoisyLinear`, `PrioritizedReplayMemory` (SumTree), `NStepBuffer`, `RewardShaper`, `DrillTracker`, `calculate_reward`, `optimize_model`. |
| `llm_agent.py` | `LLMAgent`: chiama un LLM via OpenRouter per scegliere le azioni. Cache LRU 512 entry, streaming SSE con early exit, HTTP/2 keepalive. |
| `llm_reward_model.py` | `LLMRewardModel`: chiama un LLM per stimare la reward di una transizione. Stessa architettura di `llm_agent.py`. |
| `benchmark.py` | Confronto offline tra modelli LLM su stati di gioco mock. Non richiede il gioco installato. |
| `evaluate_agents.py` | Valutazione headless con il motore di gioco reale. Produce CSV, JSON e grafici PNG. |
| `training_monitor.py` | Logger CSV + grafici matplotlib. Non usa TensorBoard (incompatibile con SDL/X11). |

---

## Output e salvataggi

### Checkpoint modello (`*.pth`)

Salvato ogni 1000 step in `RenfLearningAgent/driller_ai_v31.pth`. Contiene:
- Pesi della rete (`model_state_dict`)
- Stato ottimizzatore (`optimizer_state_dict`)
- Step corrente
- Epsilon corrente
- Stato del curriculum (livello massimo sbloccato, storia episodi)

### Replay buffer (`driller_buffer_v31/`)

Salvato ogni 5000 step come cartella di file `.npy`. Contiene fino a 50.000 transizioni (stato, azione, reward, stato successivo, done, priorità). Al riavvio vengono caricate le ultime 20.000 transizioni.

### Log training (`runs/`)

Il `TrainingMonitor` crea una cartella `runs/driller_v6_MMDD_HHMMSS/` con:
- `training_log.csv`: tutte le metriche step-by-step e per episodio
- `plots/reward_curve.png`: curva della reward
- `plots/depth_curve.png`: profondità massima per episodio
- `plots/win_rate.png`: win rate rolling su 20 episodi
- `plots/action_dist.png`: distribuzione delle azioni (grafico a torta)

I grafici PNG vengono aggiornati automaticamente ogni 100 episodi durante il training.

### Risultati benchmark (`eval_results/`)

Il file `evaluate_agents.py` salva in `eval_results/`:
- `results_TIMESTAMP.csv`: un record per episodio per ogni agente
- `summary_TIMESTAMP.json`: statistiche aggregate con intervalli di confidenza bootstrap
- `comparison_TIMESTAMP.png`: grafico comparativo con 6 subplot

---

## Risoluzione problemi

### `ImportError: No module named 'pygame'`
L'ambiente virtuale non è attivato o la dipendenza non è installata.
```bash
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
pip install pygame
```

### `RuntimeError: CUDA out of memory`
Riduci `BATCH_SIZE` in `main.py` (es. da 128 a 64) o abbassa `MEMORY_SIZE`.

### Il training è lento su CPU
È normale: con CPU il training è circa 5-10× più lento che con GPU. Per accelerare, imposta `TRAIN_EVERY = 5` (meno backpropagation per azione) o riduci `N_FRAMES = 2`.

### `EnvironmentError: OPENROUTER_API_KEY non impostata`
La variabile d'ambiente manca. Vedi la sezione [Configurazione](#configurazione).

### Il gioco si avvia senza finestra (solo testo nel terminale)
`PLAY_MODE` è impostato a `False` in `main.py`. Cambialo in `True` per vedere la finestra grafica.

### Errore audio su Linux (`ALSA lib pcm.c: unable to open slave`)
SDL non trova il dispositivo audio. Il messaggio è innocuo (il gioco funziona ugualmente). Per sopprimerlo:
```bash
SDL_AUDIODRIVER=dummy python main.py
```

### `OMP: Error #15: Initializing libiomp5.dylib` (macOS)
Già gestito in `main.py` con `KMP_DUPLICATE_LIB_OK=True`. Se l'errore persiste:
```bash
pip uninstall numpy
pip install numpy
```

### Il checkpoint non viene caricato (`Architettura cambiata, riparto da zero`)
L'architettura della rete è cambiata dall'ultima sessione. Il training riparte da zero con pesi casuali. Se vuoi preservare il training precedente, ripristina la versione precedente di `ai_agent.py`.

### `pygame.error: No video mode has been set` in headless training
Assicurati che `PLAY_MODE = False` sia impostato PRIMA dell'avvio di pygame. La variabile `SDL_VIDEODRIVER=dummy` viene impostata automaticamente in `main.py` solo se `HEADLESS_TRAINING = True`.

---

## Licenza

Distribuito a scopo educativo e di ricerca. Le dipendenze esterne (PyTorch, Pygame, httpx) sono soggette alle rispettive licenze.
