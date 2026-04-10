"""
llm_agent.py — LLM agent ottimizzato per latenza (sincrono)
Provider: OpenRouter (unica API per tutti i modelli)

Modelli supportati:
  - llama    → meta-llama/llama-3.1-8b-instruct
  - deepseek → deepseek/deepseek-coder-v2

OTTIMIZZAZIONI:
  ① LRU Cache (512 entry, chiave MD5 su stato discretizzato, finestra 9×11)
  ② Prompt compresso (~250 token, griglia 9×11)
  ③ Streaming con early JSON exit (chiude stream appena {"action":X} è pronto)
  ④ httpx HTTP/2 keepalive (elimina TCP+TLS handshake ripetuti)
  ⑤ System prompt statico separato dal user message e salvato a livello modulo
     (viene definito UNA SOLA VOLTA all'import e riusato senza riallocazioni)
"""

# ── Librerie standard ──────────────────────────────────────────────────────────
import re        # Espressioni regolari: cerca pattern nel testo (es. {"action":5})
import json      # Serializzazione/deserializzazione JSON: dict ↔ stringa
import time      # Funzioni temporali: sleep(), perf_counter(), monotonic()
import hashlib   # Algoritmi di hashing: MD5 per generare chiavi di cache compatte
import os        # Accesso al sistema operativo: legge variabili d'ambiente
from typing import Optional  # Tipo generico: Optional[X] = X oppure None

import httpx     # Client HTTP moderno con supporto HTTP/2 e streaming SSE
                 # Installazione: pip install httpx[http2]

# =============================================================================
# CONFIGURAZIONE GLOBALE
# =============================================================================

# Legge la chiave API da variabile d'ambiente; stringa vuota se non impostata
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")

# URL base dell'API OpenRouter (stile OpenAI-compatible)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Dizionario che mappa nomi brevi → identificatori completi dei modelli su OpenRouter
AVAILABLE_MODELS = {
    "llama":    "meta-llama/llama-3.1-8b-instruct",  # LLM leggero, veloce
    "deepseek": "deepseek/deepseek-chat"              # LLM alternativo
}

# Modello usato di default se non viene specificato nulla al costruttore
DEFAULT_MODEL_KEY = "llama"

# Numero massimo di token che il modello può generare nella risposta.
# 150 token sono sufficienti per {"action": 5, "reason": "testo breve"} anche
# con il prompt leggermente più ricco dato dalla griglia 9×11.
MAX_TOKENS       = 150

# Quante volte riprovare la chiamata HTTP in caso di errore
MAX_RETRIES      = 2

# Secondi di attesa tra un retry e l'altro per non sovraccaricare l'API
RETRY_DELAY_SEC  = 0.3

# Azione di default se tutti i retry falliscono (5 = DRILL_D = forare verso il basso)
FALLBACK_ACTION  = 5  # DRILL_D

# Dimensioni della finestra di gioco visibile nel prompt (righe × colonne).
# 11 righe × 9 colonne → stessa dimensione della finestra locale dell'agente RL
# usato come baseline, così i due agenti operano sulla stessa quantità di contesto.
GRID_ROWS        = 11   # hr = 5 righe sopra e sotto il giocatore
GRID_COLS        = 9    # hc = 4 colonne a sinistra e destra del giocatore

# Numero massimo di entry nella cache LRU (ogni entry occupa ~200 byte con chiave 9×11)
CACHE_SIZE        = 512

# Granularità di discretizzazione dell'ossigeno per la chiave di cache.
# Es: 78% e 82% → entrambi diventano 80 → stessa chiave → riusa la cache
CACHE_OXY_BUCKET  = 10

# Mappa da indice numerico al nome leggibile dell'azione
ACTION_NAMES = {
    0: "IDLE",     # Non fare nulla
    1: "LEFT",     # Muoviti a sinistra
    2: "RIGHT",    # Muoviti a destra
    3: "DRILL_L",  # Forare a sinistra
    4: "DRILL_R",  # Forare a destra
    5: "DRILL_D",  # Forare verso il basso
}

# =============================================================================
# SYSTEM PROMPT STATICO
# =============================================================================
# Definito UNA SOLA VOLTA a livello di modulo e riusato in ogni chiamata.
# Non metterlo dentro la funzione call() evita di ricreare la stringa ad ogni step.
# Descrive all'LLM: il gioco, i simboli della griglia, le azioni disponibili,
# e le regole di priorità da seguire per scegliere l'azione migliore.
#
# NOTA: questo è il "caching" del system prompt — essendo una costante di modulo
# Python la alloca una volta sola in memoria e la stringa viene riusata per
# riferimento in tutte le chiamate successive, senza riallocazioni.

_SYSTEM_PROMPT = (
    "You are a Mr. Driller game agent. Select the best action.\n"
    # Legenda dei simboli usati nella griglia ASCII 9×11.
    # I blocchi classic hanno sempre 1 HP; il numero mostrato è sempre "1".
    # Il blocco X NON è indistruttibile: ogni colpo su X toglie il 20% di ossigeno → evitare.
    # Il blocco solo (S) è perforabile e può cadere.
    # Il blocco delayed (D) si rompe lentamente: un colpo avvia un timer di 3 secondi
    # dopodiché il blocco sparisce senza causare danni; "!" indica che il timer è già attivo.
    # I blocchi classic dello stesso colore si aggregano: cadono insieme come gruppo.
    "SYMBOLS: @=you .=empty 1=classic(1hp,falls_as_color_group) *=combo_group "
    "O=oxygen X=costly(each_hit=-20%_oxy,avoid!) S=solo(drillable,falls) "
    "D=delayed(3s_to_break,no_dmg,dormant) !=delayed(timer_active,breaking) "
    "#=wall v=falling_block\n"
    # Mappa indice → nome azione
    "ACTIONS: 0=IDLE 1=LEFT 2=RIGHT 3=DRILL_L 4=DRILL_R 5=DRILL_D\n"
    # Regole in ordine STRETTO di priorità decrescente.
    # L'LLM deve rispettare questo ordine senza eccezioni.
    "PRIORITY RULES (strict order, highest first):\n"
    "  1. v in window (any column ±4, rows -5..+5) → move sideways to avoid crush\n"
    # Blocchi in caduta ovunque nella finestra: spostarsi lateralmente subito.
    # I classic cadono per gruppi cromatici: se un gruppo perde supporto, tutti i membri cadono.
    "  2. oxy<20% → reach nearest O capsule immediately (critical)\n"
    # Ossigeno quasi esaurito: priorità assoluta dopo la sicurezza fisica.
    "  3. oxy<40% → plan direct path to nearest O capsule\n"
    # Ossigeno basso: pianificare rotta verso la capsula più vicina.
    "  4. DRILL_D to go deeper (main objective)\n"
    # Obiettivo principale: scendere il più in profondità possibile.
    "  5. X blocks: NEVER drill into X — each hit costs 20% oxy; go around them\n"
    # I blocchi X non sono indistruttibili ma ogni colpo è molto costoso in ossigeno.
    # È quasi sempre meglio aggirare un X che tentare di sfondarlo.
    "  6. Drill classic/solo blocks to clear path downward\n"
    # Blocchi standard: fora per aprire la strada verso il basso.
    "  7. Avoid drilling delayed (D/!) blocks: they take 3s → wasted time\n"
    # I delayed rallentano: se c'è un percorso alternativo, preferirlo.
    "  8. * combo: destroying one block chain-destroys the whole color group\n"
    # Combo vantaggiosi: un colpo solo per eliminare molti blocchi adiacenti.
    "  9. IDLE only if moving would cause a falling block to hit you; otherwise move\n"
    # Non stare mai fermo senza motivo: si consuma ossigeno senza avanzare.
    # Fermarsi è accettabile SOLO per schivare un blocco che sta per cadere su di te.
    # Formato obbligatorio della risposta: JSON compatto su una sola riga
    'Reply ONLY with compact JSON: {{"action":<0-5>,"reason":"<Italian, max 8 words>"}}'
)

# =============================================================================
# LRU CACHE (implementazione manuale con dict ordinato)
# =============================================================================
# Python 3.7+ mantiene l'ordine di inserimento nei dict.
# L'LRU (Least Recently Used) si implementa così:
#   - GET: rimuovi e re-inserisci la chiave (la sposta in fondo = "usata di recente")
#   - SET: se la cache è piena, elimina il PRIMO elemento (il meno recente)

# Dict globale che funge da cache: chiave MD5 → indice azione (0-5)
_cache: dict[str, int] = {}


def _cache_key(player, level) -> str:
    """
    Genera una chiave MD5 che rappresenta lo stato corrente del gioco
    in modo discretizzato, così stati "quasi identici" condividono la stessa cache.

    La chiave include:
    - Posizione del giocatore (pX, pY) — griglia intera
    - Ossigeno arrotondato al bucket (es. ogni 10%)
    - Vite rimaste
    - Simbolo+HP di TUTTE le celle nella finestra 9×11 attorno al giocatore
      (range dx ∈ [-4,+4], dy ∈ [-5,+5] → 99 celle totali)

    NOTA: la finestra è stata estesa da 5×5 a 9×11 per corrispondere alla
    stessa finestra locale usata dall'agente RL baseline, evitando così
    collisioni di cache su stati che differiscono in celle periferiche
    (ad es. un blocco in caduta a ±4 colonne che prima era invisibile alla chiave).
    """
    # Legge posizione corrente del giocatore dalla sua struttura dati
    pY, pX = player.posAcc()

    # Arrotonda l'ossigeno al bucket più vicino (es: 78 → 70, 82 → 80)
    oxy    = int(player.oxyAcc() // CACHE_OXY_BUCKET) * CACHE_OXY_BUCKET

    # Vite rimaste (influenza la strategia: con 1 vita si gioca più cauti)
    lives  = player.livesAcc()

    # Dimensioni della mappa di gioco
    rows   = len(level)
    cols   = len(level[0]) if rows else 0  # Evita divisione per 0 su mappa vuota

    # hc e hr corrispondono a metà delle dimensioni della finestra locale 9×11,
    # allineati con GRID_COLS=9 e GRID_ROWS=11 usati nella griglia del prompt.
    hc = GRID_COLS // 2   # 4 colonne a sinistra/destra
    hr = GRID_ROWS // 2   # 5 righe sopra/sotto

    # Costruisce la "impronta" di tutte le 99 celle nella finestra 9×11
    # attorno al giocatore (range dx ∈ [-4,+4], dy ∈ [-5,+5])
    cells = []
    for dy in range(-hr, hr + 1):       # Da -5 a +5 (11 righe)
        for dx in range(-hc, hc + 1):   # Da -4 a +4 (9 colonne)
            ny, nx = pY + dy, pX + dx   # Coordinata assoluta della cella
            if 0 <= ny < rows and 0 <= nx < cols:
                # Cella valida: legge tipo e HP del blocco
                b  = level[ny][nx]
                bt = b.typeAccess()          # Tipo: "classic", "pill", "unbreakable", ecc.
                hp = min(b.hpAccess(), 3)    # HP capped a 3 per la discretizzazione
                cells.append(f"{bt[0]}{hp}") # Es: "c1" (classic 1HP), "s1" (solo), "x2" (X)
            else:
                # Fuori dai bordi della mappa → trattato come muro
                cells.append("##")

    # Assembla la stringa grezza: posizione + ossigeno + vite + celle finestra 9×11
    raw = f"{pX},{pY},{oxy},{lives}|{''.join(cells)}"

    # Restituisce l'hash MD5 della stringa (128 bit → 32 caratteri esadecimali)
    # MD5 è abbastanza veloce e le collisioni sono rare per questo uso
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[int]:
    """
    Legge dalla cache (LRU). Restituisce l'azione salvata oppure None.
    L'operazione pop+reassign sposta la chiave in fondo al dict
    (= "usata di recente"), implementando la politica LRU.
    """
    if key in _cache:
        val = _cache.pop(key)   # Rimuove la chiave dalla posizione attuale
        _cache[key] = val       # Re-inserisce in fondo (= più recente)
        return val              # Restituisce il valore trovato
    return None                 # Cache miss


def _cache_set(key: str, action: int) -> None:
    """
    Scrive nella cache (LRU). Se la cache è piena, elimina l'elemento
    meno recentemente usato (il primo nel dict, grazie all'ordine di inserimento).
    """
    if len(_cache) >= CACHE_SIZE:
        # next(iter(...)) restituisce la PRIMA chiave del dict (la meno recente)
        del _cache[next(iter(_cache))]  # Eviction LRU
    _cache[key] = action  # Inserisce la nuova coppia chiave→azione


# =============================================================================
# COSTRUZIONE PROMPT COMPRESSO
# =============================================================================

# Tipi di blocco che possono cadere quando non supportati dal basso.
# NOTA: "unbreakable" è escluso — è fisso nella mappa e non cade mai.
#       "solo" è incluso — cade esattamente come i delayed.
#       "classic" è incluso ma richiede logica speciale di gruppo cromatico
#       (vedere _compute_falling_set): un classic cade solo se l'intero suo
#       gruppo di colore perde il supporto.
_FALLING_TYPES = {"classic", "delayed", "solo"}


def _compute_falling_set(level, rows, cols, yr_start, yr_end, xr_start, xr_end) -> set:
    """
    Calcola il set di coordinate (ny, nx) dei blocchi in caduta all'interno
    della finestra rettangolare [yr_start..yr_end) × [xr_start..xr_end).

    Regole per tipo:
    - solo, delayed : cade se HP > 0 e cella direttamente sotto è vuota (HP ≤ 0)
    - classic       : cade solo se l'INTERO gruppo cromatico connesso è privo
                      di supporto esterno. I classic dello stesso colore si
                      "aggrappano" tra loro e cadono come corpo rigido.
                      Algoritmo:
                        1. BFS per raccogliere tutti i classic adiacenti dello
                           stesso colore (gruppo cromatico).
                        2. Il gruppo è "supportato" se ALMENO UN membro ha
                           sotto di sé un blocco con HP > 0 che NON appartiene
                           al gruppo stesso, oppure se si trova sul bordo
                           inferiore della mappa.
                        3. Se nessun membro è supportato → tutto il gruppo cade.
    - unbreakable   : non è in _FALLING_TYPES → ignorato (non cade mai)
    """
    falling  = set()
    # Mappa (ny, nx) → bool indicante se il blocco fa parte di un gruppo già analizzato.
    # Evita di ricalcolare il BFS per ogni membro dello stesso gruppo cromatico.
    group_result: dict[tuple, bool] = {}

    for ny in range(max(yr_start, 0), min(yr_end, rows)):
        for nx in range(max(xr_start, 0), min(xr_end, cols)):
            b  = level[ny][nx]
            if b.hpAccess() <= 0:
                continue                         # Cella vuota: salta
            bt = b.typeAccess()
            if bt not in _FALLING_TYPES:
                continue                         # Tipo non soggetto a caduta: salta

            # ── Caso 1: solo e delayed — regola semplice ────────────────────
            if bt in ("solo", "delayed"):
                # Cade se la cella direttamente sotto è vuota
                if ny + 1 < rows and level[ny + 1][nx].hpAccess() <= 0:
                    falling.add((ny, nx))
                continue

            # ── Caso 2: classic — logica gruppo cromatico ───────────────────
            # Se questo membro è già stato analizzato come parte di un gruppo,
            # usa il risultato precalcolato senza rieseguire il BFS.
            if (ny, nx) in group_result:
                if group_result[(ny, nx)]:
                    falling.add((ny, nx))
                continue

            # Legge il colore del blocco per individuare il gruppo cromatico.
            # Se il metodo colorAccess() non esiste, ogni blocco è trattato
            # come gruppo singolo (comportamento conservativo).
            try:
                color = b.colorAccess()
            except AttributeError:
                color = None

            # BFS per raccogliere l'intero gruppo cromatico connesso.
            # Considera solo i blocchi classic adiacenti (4 direzioni ortogonali)
            # dello stesso colore.
            group: set[tuple] = {(ny, nx)}
            queue             = [(ny, nx)]
            while queue:
                cy, cx = queue.pop(0)
                for ddy, ddx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nny, nnx = cy + ddy, cx + ddx
                    if (nny, nnx) in group:
                        continue
                    if not (0 <= nny < rows and 0 <= nnx < cols):
                        continue
                    nb = level[nny][nnx]
                    if nb.typeAccess() != "classic" or nb.hpAccess() <= 0:
                        continue
                    # Controlla compatibilità di colore:
                    # se uno dei due blocchi non ha colore → li tratta come stesso gruppo
                    try:
                        nc = nb.colorAccess()
                    except AttributeError:
                        nc = None
                    if color is None or nc is None or nc == color:
                        group.add((nny, nnx))
                        queue.append((nny, nnx))

            # Verifica se il gruppo ha almeno un punto di supporto esterno.
            # "Supporto" = cella sotto con HP > 0 che NON è nel gruppo stesso,
            # oppure essere sul bordo inferiore della mappa.
            group_supported = False
            for (gy, gx) in group:
                if gy + 1 >= rows:
                    # Fondo mappa: il blocco poggia sulla terra → supportato
                    group_supported = True
                    break
                nb_below = level[gy + 1][gx]
                if nb_below.hpAccess() > 0 and (gy + 1, gx) not in group:
                    # Blocco sotto appartiene a un tipo diverso o colore diverso
                    # → fornisce supporto esterno al gruppo
                    group_supported = True
                    break

            # Memorizza il risultato per tutti i membri del gruppo
            is_falling = not group_supported
            for member in group:
                group_result[member] = is_falling
            if is_falling:
                falling.update(group)   # Tutto il gruppo cade insieme

    return falling


def _build_compact_grid(player, level) -> str:
    """
    Costruisce una griglia ASCII 9×11 centrata sul giocatore.
    Ogni cella è rappresentata da 1-3 caratteri secondo la legenda del system prompt.
    I blocchi in caduta vengono prefissati con 'v'.

    La griglia è 9 colonne × 11 righe (hc=4, hr=5), allineata con la finestra
    locale dell'agente RL baseline per garantire la stessa quantità di contesto.

    Tipi di blocco:
      classic     → "1" (HP sempre 1) o "1*" se in combo cromatico; cade per gruppo
      solo        → "S"; cade come blocco singolo (a differenza di unbreakable)
      pill        → "O" (capsula ossigeno)
      unbreakable → "X" (ogni colpo costa 20% ossigeno, non indistruttibile!)
      delayed     → "D" (dormiente, timer non avviato)
                    "!" (timer attivo, si romperà tra ~3s; nessuna esplosione)
    """
    # Posizione assoluta del giocatore nella mappa completa
    pY, pX = player.posAcc()
    rows   = len(level)
    cols   = len(level[0]) if rows else 0

    # Metà righe/colonne per centrare la finestra sul giocatore (9×11)
    hc = GRID_COLS // 2   # 4 colonne a sinistra/destra
    hr = GRID_ROWS // 2   # 5 righe sopra/sotto

    # ── Pre-calcolo blocchi in caduta per l'intera finestra 9×11 ─────────────
    # Usa _compute_falling_set che gestisce la logica di gruppo cromatico
    # per i classic e la regola semplice per solo/delayed.
    # L'intervallo copre tutta la finestra visibile + 1 riga extra in basso
    # per verificare il supporto dell'ultima riga della finestra.
    falling = _compute_falling_set(
        level, rows, cols,
        yr_start = pY - hr,          # Prima riga della finestra
        yr_end   = pY + hr + 2,      # Ultima riga + 1 extra per il controllo supporto
        xr_start = pX - hc,          # Prima colonna della finestra
        xr_end   = pX + hc + 1       # Ultima colonna della finestra
    )

    # ── Costruisce la griglia riga per riga ──────────────────────────────────
    grid_lines = []
    for dy in range(-hr, hr + 1):      # Da -5 a +5 righe rispetto al giocatore (11)
        row_cells = []
        for dx in range(-hc, hc + 1):  # Da -4 a +4 colonne rispetto al giocatore (9)
            # Cella centrale = posizione del giocatore
            if dy == 0 and dx == 0:
                row_cells.append("@")  # @ = simbolo del giocatore
                continue

            # Coordinata assoluta della cella da visualizzare
            ny, nx = pY + dy, pX + dx

            # Fuori dai bordi della mappa → muro invalicabile
            if ny < 0 or ny >= rows or nx < 0 or nx >= cols:
                row_cells.append("#")
                continue

            # Legge i dati del blocco nella cella (ny, nx)
            b  = level[ny][nx]
            hp = b.hpAccess()   # HP del blocco (0 = vuoto/distrutto)
            bt = b.typeAccess() # Tipo stringa del blocco

            # ── Selezione simbolo per ogni tipo di blocco ────────────────────
            if hp <= 0:
                sym = "."           # Cella vuota (già forata o mai occupata)

            elif bt == "pill":
                sym = "O"           # Capsula di ossigeno: priorità alta se oxy basso

            elif bt == "unbreakable":
                # Blocco costoso: NON indistruttibile, ma ogni colpo toglie il 20%
                # di ossigeno al giocatore. Da evitare assolutamente se possibile.
                sym = "X"

            elif bt == "solo":
                # Blocco solo: perforabile e soggetto a caduta singola (non di gruppo)
                # Prefisso "v" se sta cadendo (cella sotto vuota)
                sym = "vS" if (ny, nx) in falling else "S"

            elif bt == "delayed":
                # Blocco a tempo: richiede UN SOLO colpo per avviare il timer (3s),
                # poi sparisce senza causare danni. Non esplode, non danneggia.
                # "!" = timer già attivo (il blocco sta per sparire da solo),
                # "D" = dormiente (non ancora colpito, impiegherebbe 3s a rompersi).
                # Conviene evitarli: occupano tempo senza beneficio tattico immediato.
                try:
                    sym = "!" if b.isActiveAccess() else "D"
                except AttributeError:
                    sym = "D"       # Se il metodo non esiste, assume non attiva

                # Anche i delayed possono cadere (sono in _FALLING_TYPES)
                if (ny, nx) in falling:
                    sym = "v" + sym

            elif bt == "classic":
                # Blocco standard: HP sempre 1, cade per gruppo cromatico connesso.
                # "v" = prefisso se l'intero gruppo cromatico sta cadendo.
                h = min(hp, 1)  # Classic ha sempre 1 HP; capped a 1 per sicurezza

                # Controlla se questo blocco fa parte di un gruppo combo (stesso colore
                # adiacente): distruggere un membro causa la distruzione a catena dell'intero
                # gruppo → vantaggio tattico.
                in_combo = False
                try:
                    color = b.colorAccess()  # Colore del blocco corrente
                    # Controlla i 4 vicini ortogonali (su, giù, sinistra, destra)
                    for ddy, ddx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nny, nnx = ny + ddy, nx + ddx
                        if 0 <= nny < rows and 0 <= nnx < cols:
                            nb = level[nny][nnx]
                            if nb.typeAccess() == "classic" and nb.hpAccess() > 0:
                                try:
                                    if nb.colorAccess() == color:
                                        in_combo = True  # Trovato almeno un vicino stesso colore
                                        break
                                except AttributeError:
                                    pass
                except AttributeError:
                    pass

                # Aggiunge "*" se è in combo (informazione tattica per il chain-destroy)
                sym = f"{h}*" if in_combo else str(h)

                # Prefisso "v" se il blocco (e il suo intero gruppo cromatico) sta cadendo
                if (ny, nx) in falling:
                    sym = "v" + sym

            else:
                sym = "?"  # Tipo sconosciuto (futuro/debug)

            row_cells.append(sym)

        # Unisce le celle della riga con spazi e aggiunge alla lista di righe
        grid_lines.append(" ".join(row_cells))

    # Unisce tutte le righe con newline → stringa griglia multi-riga 9×11
    return "\n".join(grid_lines)


def _nearest_pill_info(player, level) -> str:
    """
    Scansiona un raggio 12×12 attorno al giocatore e trova la capsula di ossigeno
    più vicina (distanza di Manhattan: |Δx| + |Δy|).
    Restituisce una stringa descrittiva con direzione e distanza, o "" se non trovata.
    """
    pY, pX = player.posAcc()
    rows   = len(level)
    cols   = len(level[0]) if rows else 0
    best_dist, best_desc = 999, ""  # Distanza inizialmente "infinita"

    for dy in range(-12, 13):       # Scansiona 25 righe attorno al giocatore
        for dx in range(-12, 13):   # Scansiona 25 colonne attorno al giocatore
            ny, nx = pY + dy, pX + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                b = level[ny][nx]
                # Cerca solo capsule di ossigeno intatte (hp > 0)
                if b.typeAccess() == "pill" and b.hpAccess() > 0:
                    dist = abs(dy) + abs(dx)  # Distanza di Manhattan
                    if dist < best_dist:      # Tiene solo la più vicina
                        best_dist = dist
                        # Formatta: offset relativo e distanza totale
                        best_desc = f"O at ({dx:+d},{dy:+d}) dist={dist}"

    return best_desc  # Stringa vuota se nessuna capsula trovata nel raggio


def _immediate_threats(player, level) -> str:
    """
    Scansiona l'intera finestra locale 9×11 (dx ∈ [-4,+4], dy ∈ [-5,+5])
    per rilevare minacce immediate:

    1. Blocchi in caduta nell'intera finestra — le 5 righe SOPRA il giocatore
       sono considerate "sopra la testa"; le righe SOTTO rilevano blocchi
       che potrebbero schiacciarlo quando fora verso il basso. La finestra
       completa 9×11 replica la stessa visibilità dell'agente RL baseline.

    2. I blocchi in caduta vengono calcolati con _compute_falling_set che
       include la logica di gruppo cromatico per i classic: un gruppo di
       classic dello stesso colore cade insieme quando perde il supporto.

    NOTA IMPORTANTE — blocchi delayed (D/!):
       I blocchi delayed NON esplodono e NON danneggiano il giocatore per
       prossimità. Il loro unico impatto è il tempo: se colpiti avviano un
       timer di 3 secondi. Non vengono quindi segnalati come minacce
       attive, ma solo come "blocchi lenti" nel prompt della griglia.

    Restituisce una stringa con tutte le minacce trovate, o "none".
    """
    pY, pX = player.posAcc()
    rows   = len(level)
    cols   = len(level[0]) if rows else 0

    hc = GRID_COLS // 2   # 4 colonne ← stessa ampiezza della griglia 9×11
    hr = GRID_ROWS // 2   # 5 righe   ← stessa altezza della griglia 9×11

    threats = []

    # ── Calcola i blocchi in caduta nell'intera finestra 9×11 ───────────────
    # Usa _compute_falling_set per gestire correttamente i gruppi cromatici
    # dei classic, i solo e i delayed. Gli unbreakable non cadono mai.
    falling = _compute_falling_set(
        level, rows, cols,
        yr_start = pY - hr,         # 5 righe sopra il giocatore
        yr_end   = pY + hr + 2,     # 5 righe sotto + 1 extra per il controllo supporto
        xr_start = pX - hc,         # 4 colonne a sinistra
        xr_end   = pX + hc + 1      # 4 colonne a destra
    )

    # ── Segnala blocchi in caduta nella finestra completa ───────────────────
    # Riporta la posizione relativa di ogni blocco in caduta rispetto al giocatore.
    # Includiamo sia le righe superiori (blocchi che cadono verso il giocatore)
    # sia le righe inferiori (blocchi già in caduta che il giocatore potrebbe
    # incontrare forando verso il basso — DRILL_D pericoloso).
    for (ny, nx) in sorted(falling):  # Ordina per posizione stabile nel messaggio
        dy = ny - pY   # Offset verticale rispetto al giocatore (negativo = sopra)
        dx = nx - pX   # Offset orizzontale

        # Filtra le coordinate all'interno della finestra 9×11
        if -hr <= dy <= hr and -hc <= dx <= hc:
            bt = level[ny][nx].typeAccess()
            threats.append(f"fall({dx:+d},{dy:+d},{bt[0]})")

    # Unisce tutte le minacce con spazio, o "none" se nessuna trovata
    return " ".join(threats) if threats else "none"


def build_compact_prompt(player, level, last_action_idx: int) -> str:
    """
    Costruisce il messaggio utente da inviare all'LLM.
    È ottimizzato per usare ~250 token invece degli ~800 del prompt originale:
    - Intestazione compatta su una riga (posizione, ossigeno, vite, score, ultima azione)
    - Riga minacce immediate (blocchi in caduta nell'intera finestra 9×11)
    - Riga capsula ossigeno più vicina (solo se esiste nel raggio)
    - Griglia ASCII 9×11 centrata sul giocatore

    SEQUENZIALITÀ: questo prompt viene costruito DOPO che il gioco ha eseguito
    l'azione precedente e aggiornato lo stato. Ogni chiamata all'LLM riceve
    quindi lo stato s(t) aggiornato → il loop è s(t) → a(t) → s(t+1) → a(t+1).
    """
    # Dati di stato del giocatore
    pY, pX = player.posAcc()     # Posizione griglia (colonna, riga)
    oxy    = player.oxyAcc()     # Ossigeno residuo in percentuale (0-100)
    lives  = player.livesAcc()  # Vite rimaste
    score  = player.scoreAcc()  # Punteggio corrente
    rows   = len(level)          # Altezza totale della mappa

    # Profondità relativa del giocatore nella mappa (0% = cima, 100% = fondo)
    depth  = round(pY / max(rows, 1) * 100, 1)

    # Flag testuale di allerta ossigeno (aggiunto inline nell'intestazione)
    oxy_flag = ""
    if oxy < 20:
        oxy_flag = "!!CRITICAL"   # Ossigeno quasi esaurito: priorità assoluta
    elif oxy < 40:
        oxy_flag = "!LOW"         # Ossigeno basso: pianificare subito percorso

    # Costruisce le singole sezioni del prompt
    grid     = _build_compact_grid(player, level)    # Griglia ASCII 9×11 centrata
    threats  = _immediate_threats(player, level)     # Stringa minacce finestra 9×11
    pill_str = _nearest_pill_info(player, level)     # Info capsula più vicina

    # Assembla il prompt finale con f-string multi-parte
    return (
        # Riga 1: stato generale compatto
        f"pos=({pX},{pY}) depth={depth}% oxy={oxy:.0f}%{oxy_flag} "
        f"lives={lives} score={score} last={ACTION_NAMES.get(last_action_idx,'?')}\n"
        # Riga 2: minacce + capsula ossigeno (aggiunta solo se presente)
        f"threats:{threats}"
        + (f" | nearest_O:{pill_str}" if pill_str else "")
        # Righe finali: griglia 9×11 con intestazione
        + f"\ngrid (9x11, rows=-5..+5 from @, cols=-4..+4, v=falling):\n{grid}"
    )


# =============================================================================
# HELPER: PARSING RETRY-AFTER (per gestione rate limit 429)
# =============================================================================

def _parse_retry_after(resp: httpx.Response) -> float:
    """
    Determina quanti secondi attendere dopo un errore HTTP 429 (Too Many Requests).
    Controlla le intestazioni HTTP in ordine di priorità:
    1. retry-after (standard HTTP)
    2. x-ratelimit-reset-requests (header proprietario OpenAI/OpenRouter)
    3. Messaggio di errore JSON nel corpo della risposta
    4. Default conservativo di 30 secondi

    Aggiunge sempre MARGIN=1.0s come buffer di sicurezza.
    """
    MARGIN = 1.0  # Buffer aggiuntivo per evitare di riprovare troppo presto

    # ── Prova l'header standard HTTP "retry-after" ───────────────────────────
    raw = resp.headers.get("retry-after", "")
    if raw:
        try:
            return float(raw) + MARGIN  # Può essere un numero di secondi
        except ValueError:
            pass  # Potrebbe essere una data HTTP → ignora e prova il successivo

    # ── Prova l'header proprietario di OpenRouter/OpenAI ────────────────────
    raw = resp.headers.get("x-ratelimit-reset-requests", "")
    if raw:
        try:
            # Il valore può avere il suffisso "s" (es. "30s") → va rimosso
            return float(raw.rstrip("s")) + MARGIN
        except ValueError:
            pass

    # ── Prova a estrarre il tempo dal corpo JSON dell'errore ─────────────────
    try:
        body = resp.json()  # Deserializza il corpo della risposta come JSON
        msg  = body.get("error", {}).get("message", "")  # Cerca il campo "message"
        # Cerca pattern "try again in X.Xs" con regex
        m    = re.search(r'try again in ([\d.]+)s', msg)
        if m:
            return float(m.group(1)) + MARGIN  # Estrae il numero di secondi
    except Exception:
        pass  # Se il corpo non è JSON valido o il campo manca → ignora

    # ── Fallback conservativo: attendi 30 secondi ────────────────────────────
    return 30.0


# =============================================================================
# CLIENT HTTP OTTIMIZZATO (OpenRouter) — classe privata (prefisso _)
# =============================================================================

class _FastHTTPClient:
    """
    Client HTTP riutilizzabile che mantiene la connessione aperta tra le chiamate.
    Vantaggi rispetto a un client "usa e getta":
    - Elimina il costo del TCP handshake (~50ms) e TLS handshake (~100ms) per ogni call
    - HTTP/2 multiplexing: più richieste sulla stessa connessione
    - Keepalive: la connessione rimane aperta fino a 120 secondi di inattività

    Implementa streaming SSE con "early exit": non aspetta la fine dello stream
    ma chiude la connessione non appena trova {"action": X} nel buffer.
    """

    def __init__(self, model_key: str = DEFAULT_MODEL_KEY) -> None:
        """
        Inizializza il client HTTP con un modello specifico.
        Valida che il modello esista nel registro AVAILABLE_MODELS.
        """
        # Valida il modello richiesto prima di qualsiasi altra operazione
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(
                f"Modello '{model_key}' non disponibile. "
                f"Scegli tra: {list(AVAILABLE_MODELS.keys())}"
            )
        self.model_key  = model_key                    # Nome breve (es: "llama")
        self._model_id  = AVAILABLE_MODELS[model_key]  # ID completo per l'API

        # Crea il client httpx con configurazione ottimizzata per bassa latenza
        self._client = httpx.Client(
            base_url = OPENROUTER_BASE_URL,  # Tutte le richieste partono da qui
            headers  = {
                # Token di autenticazione Bearer (richiesto da OpenRouter)
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                # Indica che il corpo della richiesta è JSON
                "Content-Type":  "application/json",
                # Obbligatorio da OpenRouter: identifica il progetto/app
                "HTTP-Referer":  "https://github.com/mr-driller-rl",
                # Nome opzionale dell'app (visibile nella dashboard OpenRouter)
                "X-Title":       "Mr. Driller RL Agent",
            },
            http2   = True,                 # Abilita HTTP/2 (multiplexing, header compression)
            timeout = httpx.Timeout(
                30.0,       # Timeout totale della richiesta in secondi
                connect=5.0 # Timeout solo per l'apertura della connessione TCP
            ),
            limits  = httpx.Limits(
                max_keepalive_connections = 5,    # Pool: max 5 connessioni in idle
                keepalive_expiry          = 120,  # Chiudi connessione dopo 120s di inattività
            ),
        )
        self._url               = "/chat/completions"  # Endpoint relativo all'API
        self._retry_after_until = 0.0  # Timestamp fino a cui evitare nuove chiamate (rate limit)

        print(f"[FAST_LLM:{model_key}] Pronto → {self._model_id}")

    def call(self, system: str, user: str) -> Optional[dict]:
        """
        Punto di ingresso pubblico per una chiamata al modello.
        Verifica prima se c'è un cooldown attivo per rate limit,
        poi delega alla funzione di streaming.
        """
        now = time.monotonic()  # Orologio monotono (non risente di cambio ora sistema)
        if self._retry_after_until > now:
            # Siamo ancora nel periodo di cooldown post-429
            wait = self._retry_after_until - now
            print(f"[FAST_LLM:{self.model_key}] Rate-limit cooldown: {wait:.1f}s")
            time.sleep(wait)  # Blocca il thread per il tempo necessario
        return self._call_stream(system, user)  # Esegui la chiamata in streaming

    def _call_stream(self, system: str, user: str) -> Optional[dict]:
        """
        Esegue la chiamata HTTP POST con streaming SSE e implementa l'early exit.

        Protocollo SSE (Server-Sent Events):
        - Il server invia righe nel formato: "data: {JSON}\n\n"
        - L'ultima riga è: "data: [DONE]\n\n"
        - Ogni chunk JSON contiene il delta del testo generato

        Early exit: non aspetta [DONE] ma chiude lo stream non appena
        trova {"action": X} nel buffer accumulato.
        """
        # Payload della richiesta: specifiche del modello e messaggi
        payload = {
            "model":      self._model_id,  # ID completo del modello su OpenRouter
            "max_tokens": MAX_TOKENS,       # Limita la lunghezza della risposta (150)
            "stream":     True,             # Abilita lo streaming SSE
            "messages": [
                # Il system prompt viene inviato come contesto fisso.
                # _SYSTEM_PROMPT è una costante di modulo: nessuna riallocazione.
                {"role": "system", "content": system},
                # Il user message contiene lo stato del gioco (griglia 9×11 + metadati)
                {"role": "user",   "content": user},
            ],
        }
        buffer = ""  # Accumula i delta di testo ricevuti fino all'early exit

        try:
            # Context manager: apre lo stream e lo chiude automaticamente all'uscita
            with self._client.stream("POST", self._url, json=payload) as resp:

                # ── Gestione errori HTTP ─────────────────────────────────────
                if resp.status_code != 200:
                    resp.read()  # Legge il corpo completo per poter accedere a resp.text/json()
                    if resp.status_code == 429:
                        # Rate limit: calcola quanto attendere e registra il cooldown
                        retry_after = _parse_retry_after(resp)
                        self._retry_after_until = time.monotonic() + retry_after
                        print(f"[FAST_LLM:{self.model_key}] 429 → attesa {retry_after:.1f}s")
                    else:
                        # Altro errore: logga le prime 200 caratteri del corpo
                        preview = resp.text[:200] if resp.text else "(no body)"
                        print(f"[FAST_LLM:{self.model_key}] HTTP {resp.status_code}: {preview}")
                    resp.raise_for_status()  # Solleva httpx.HTTPStatusError

                # ── Lettura dello stream SSE riga per riga ───────────────────
                for line in resp.iter_lines():
                    # Salta righe vuote (separatori SSE) e righe senza prefisso dati
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Rimuove il prefisso "data: " (6 caratteri)

                    # Il server ha finito di trasmettere → esci dal loop
                    if data_str == "[DONE]":
                        break

                    # Decodifica il chunk JSON e estrae il delta di testo
                    try:
                        chunk  = json.loads(data_str)
                        # Naviga nella struttura: choices[0].delta.content
                        # Usa "or ''" per gestire il caso in cui content sia None
                        delta  = chunk["choices"][0]["delta"].get("content") or ""
                        buffer += delta  # Aggiunge il delta al buffer accumulato
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue  # Chunk malformato → salta (non è critico)

                    # ── Early exit ───────────────────────────────────────────
                    # Cerca il pattern {"action": <cifra singola>} nel buffer.
                    # [^}]* permette la presenza del campo "reason" tra "action" e "}"
                    m = re.search(r'\{\s*"action"\s*:\s*(\d)[^}]*\}', buffer)
                    if m:
                        try:
                            # Tenta di parsare il JSON trovato
                            result = json.loads(m.group())
                            if "action" in result:
                                return result  # ← chiude lo stream e restituisce subito
                        except json.JSONDecodeError:
                            pass  # JSON ancora incompleto (es: "reason" troncato) → continua

        except httpx.HTTPStatusError:
            pass  # Già gestito sopra con raise_for_status, solo per uscire pulito
        except httpx.RequestError as e:
            # Errori di rete: timeout, connessione rifiutata, DNS, ecc.
            print(f"[FAST_LLM:{self.model_key}] Request error: {e}")
        except Exception as e:
            # Cattura qualsiasi altro errore imprevisto (logging + prosegui)
            print(f"[FAST_LLM:{self.model_key}] Unexpected: {type(e).__name__}: {e}")

        # ── Fallback post-stream: estrazione parziale dal buffer ─────────────
        # Lo stream è finito o si è interrotto, ma il buffer potrebbe contenere
        # un JSON parziale con almeno il campo "action"
        m = re.search(r'\{\s*"action"\s*:\s*(\d)', buffer)
        if m:
            # Crea un dict minimale con solo l'action (il reason è perso)
            return {"action": int(m.group(1)), "reason": "partial"}
        return None  # Nessun dato utilizzabile → il chiamante userà il fallback

    def close(self) -> None:
        """Chiude esplicitamente il client HTTP e rilascia le connessioni in pool."""
        try:
            self._client.close()
        except Exception:
            pass  # Ignora errori nella chiusura (es: client già chiuso)


# =============================================================================
# CLASSE PUBBLICA: LLMAgent
# =============================================================================

class LLMAgent:
    """
    Agente LLM ottimizzato per bassa latenza, sincrono (nessun async/await).
    Combina cache LRU + HTTP/2 keepalive + streaming early exit per minimizzare
    il tempo di risposta per frame del gioco.

    SEQUENZIALITÀ GARANTITA:
      Questo agente opera in modo strettamente sequenziale:
        s(t) → select_action() → a(t) → [gioco esegue a(t)] → s(t+1) → select_action() → …
      Non viene mai avviata una nuova chiamata all'LLM prima che la precedente
      sia completata e che il gioco abbia aggiornato lo stato con gli effetti
      dell'azione scelta. Il thread chiamante si blocca in select_action() per
      tutta la durata della richiesta HTTP (o viene servito immediatamente dalla
      cache). Non c'è parallelismo, né buffering di azioni future.

    Flusso select_action():
      1. Genera chiave cache dallo stato corrente del gioco (finestra 9×11)
      2. Cache hit → restituisce immediatamente l'azione salvata
      3. Cache miss → costruisce il prompt compresso (griglia 9×11)
      4. Chiama l'LLM in streaming con retry
      5. Salva in cache e restituisce l'azione

    USO:
        from llm_agent import LLMAgent
        agent = LLMAgent(model_key="llama")  # o "deepseek"
        action_idx, reason = agent.select_action(player, level, last_action_idx)
        print(agent.stats())
    """

    def __init__(self, model_key: str = DEFAULT_MODEL_KEY) -> None:
        """
        Inizializza l'agente validando la chiave API e creando il client HTTP.
        Raise EnvironmentError se OPENROUTER_API_KEY non è impostata.
        """
        # Controllo obbligatorio: senza API key non possiamo fare nulla
        if not OPENROUTER_API_KEY:
            raise EnvironmentError(
                "[FAST_LLM] OPENROUTER_API_KEY non impostata.\n"
                "Ottieni la chiave su https://openrouter.ai e imposta:\n"
                "  export OPENROUTER_API_KEY='sk-or-...'"
            )
        self.model_key      = model_key           # Nome breve del modello attivo
        self._http          = _FastHTTPClient(model_key)  # Client HTTP con keepalive

        # ── Metriche di monitoraggio ─────────────────────────────────────────
        self.call_count     = 0    # Numero di chiamate effettive all'LLM (cache miss)
        self.cache_hits     = 0    # Numero di risposte servite dalla cache
        self.error_count    = 0    # Numero di fallimenti (timeout, errori API, ecc.)
        self.total_latency  = 0.0  # Latenza totale accumulata in secondi

        # ── Mappa per calcolo consistenza ────────────────────────────────────
        # Per ogni chiave di cache, lista di azioni che il modello ha scelto.
        # Se il modello è consistente, ogni lista ha un solo valore unico.
        self._consistency_map: dict[str, list[int]] = {}

        print(
            f"[FAST_LLM:{model_key}] Agente pronto | "
            f"cache={CACHE_SIZE} (finestra 9×11) | streaming=True | early_exit=True | "
            f"max_tokens={MAX_TOKENS} | sequenziale=True"
        )

    def select_action(
        self,
        player,
        level,
        last_action_idx: int,
        n_actions: int = 6,      # Numero totale di azioni valide (0-5)
    ) -> tuple[int, str]:
        """
        Seleziona l'azione migliore per il frame corrente.
        Restituisce una tupla (action_idx, reason).

        Pipeline:
          Cache hit  → O(1), restituisce subito
          Cache miss → prompt (griglia 9×11) + HTTP + parsing + cache write + return
        """
        # ── Passo 1: calcola la chiave e controlla la cache ──────────────────
        key    = _cache_key(player, level)  # Hash MD5 della finestra 9×11
        cached = _cache_get(key)             # Cerca nella cache LRU
        if cached is not None:
            # Cache hit: evita completamente la chiamata all'LLM
            self.cache_hits += 1
            # Registra per la metrica di consistenza
            self._consistency_map.setdefault(key, []).append(cached)
            return cached, "cached"  # Restituisce subito con reason="cached"

        # ── Passo 2: costruisci il prompt per l'LLM ──────────────────────────
        # Eseguito solo in caso di cache miss
        user_msg = build_compact_prompt(player, level, last_action_idx)

        # ── Passo 3: chiama l'LLM con retry ─────────────────────────────────
        for attempt in range(MAX_RETRIES):
            t0     = time.perf_counter()           # Timer ad alta risoluzione (ns)
            result = self._http.call(_SYSTEM_PROMPT, user_msg)  # Chiamata HTTP
            lat    = time.perf_counter() - t0      # Latenza di questa chiamata
            self.total_latency += lat              # Accumula per la media
            self.call_count    += 1                # Conta la chiamata

            if result is not None:
                # ── Risposta valida ricevuta ─────────────────────────────────
                action_idx = int(result.get("action", FALLBACK_ACTION))  # Estrae l'azione
                reason     = result.get("reason", "")                    # Estrae la motivazione

                # Verifica che l'indice sia nel range valido [0, n_actions)
                if 0 <= action_idx < n_actions:
                    # Salva in cache per i frame futuri con stato simile
                    _cache_set(key, action_idx)
                    # Registra per la metrica di consistenza
                    self._consistency_map.setdefault(key, []).append(action_idx)
                    print(
                        f"[FAST_LLM:{self.model_key}] {ACTION_NAMES[action_idx]:8s} | "
                        f"lat={lat*1000:.0f}ms | {reason}"
                    )
                    return action_idx, reason  # ← Percorso normale di successo

                # Azione fuori range (bug dell'LLM) → logga e riprova
                print(f"[FAST_LLM:{self.model_key}] action fuori range ({action_idx}), fallback")

            # ── Risposta None o azione non valida → conta come errore ────────
            self.error_count += 1
            # Aspetta prima del prossimo retry (ma non dopo l'ultimo)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SEC)

        # ── Passo 4: fallback dopo tutti i retry ─────────────────────────────
        # DRILL_D è il fallback più sicuro: scendere è quasi sempre utile
        print(f"[FAST_LLM:{self.model_key}] tutti i retry falliti → DRILL_D (fallback)")
        return FALLBACK_ACTION, "fallback"

    def consistency_rate(self) -> float:
        """
        Metrica di affidabilità del modello.
        Calcola la percentuale di stati per cui il modello ha SEMPRE scelto
        la stessa azione (valutata su chiamate multiple con cache svuotata).

        Un valore di 1.0 significa perfetta coerenza decisionale.
        Un valore basso suggerisce instabilità del modello.
        """
        if not self._consistency_map:
            return 1.0  # Nessun dato → assume consistenza perfetta

        # Conta gli stati per cui tutte le azioni nella lista sono identiche
        consistent = sum(
            1 for calls in self._consistency_map.values()
            if len(set(calls)) == 1   # set() di un solo elemento = sempre stessa azione
        )
        # Percentuale: stati consistenti / stati totali osservati
        return consistent / len(self._consistency_map)

    def stats(self) -> dict:
        """
        Restituisce un dizionario con le metriche di performance dell'agente.
        Utile per il monitoraggio durante il training e per confrontare modelli.
        """
        total_reqs = self.call_count + self.cache_hits  # Richieste totali (LLM + cache)
        avg_lat    = self.total_latency / max(self.call_count, 1)  # Evita divisione per 0
        return {
            "model"          : self.model_key,
            "call_count"     : self.call_count,                              # Chiamate LLM reali
            "cache_hits"     : self.cache_hits,                              # Risposte dalla cache
            "cache_hit_rate" : round(self.cache_hits / max(total_reqs, 1), 3),  # % cache hit
            "error_count"    : self.error_count,                             # Errori totali
            "error_rate"     : round(self.error_count / max(self.call_count, 1), 3),  # % errori
            "avg_latency_ms" : round(avg_lat * 1000, 1),                     # Latenza media in ms
            "consistency"    : round(self.consistency_rate(), 3),            # Coerenza decisionale
        }

    def clear_cache(self) -> None:
        """Svuota completamente la cache globale. Utile tra episodi di gioco."""
        _cache.clear()  # Svuota il dict globale (shared tra tutte le istanze)
        print(f"[FAST_LLM:{self.model_key}] Cache svuotata")

    def __del__(self) -> None:
        """
        Distruttore: chiude il client HTTP quando l'oggetto viene garbage collected.
        Garantisce il rilascio delle connessioni anche se close() non viene chiamato.
        """
        try:
            self._http.close()
        except Exception:
            pass  # Ignora errori nel cleanup (es: già distrutto)