"""
llm_reward_model.py — Reward model LLM ottimizzato per latenza

Provider: OpenRouter (unica API per tutti i modelli)

Modelli supportati:
  - llama    → meta-llama/llama-3.1-8b-instruct
  - deepseek → deepseek/deepseek-coder-v2

OTTIMIZZAZIONI RISPETTO ALL'ORIGINALE:
  ① LRU Cache già presente → mantenuta, leggermente migliorata
  ② Prompt compresso: ~120 token vs ~400 dell'originale
  ③ httpx con HTTP/2 keepalive (elimina TCP/TLS handshake per ogni call)
  ④ Streaming con early JSON exit: chiude lo stream appena {"reward": X} è pronto
  ⑤ System prompt statico separato dal user message dinamico
  ⑥ Multi-modello: accetta model_key="llama"|"deepseek" al costruttore
"""

# ── Librerie standard ──────────────────────────────────────────────────────────
import re        # Espressioni regolari: cerca pattern come {"reward": -12.5} nel testo
import json      # Parsing/serializzazione JSON: converte stringhe ↔ dict Python
import time      # Funzioni temporali: sleep(), perf_counter() per misurare latenze
import hashlib   # Hashing crittografico: MD5 per generare chiavi di cache compatte
import os        # Interfaccia OS: legge variabili d'ambiente come OPENROUTER_API_KEY
from typing import Optional  # Annotazione tipo: Optional[X] significa "X oppure None"

import httpx     # Client HTTP asincrono/sincrono con supporto HTTP/2 e streaming SSE

# ── Import opzionale di PyTorch ───────────────────────────────────────────────
# Il reward model può restituire un tensore PyTorch (per integrazione con reti RL)
# oppure un semplice float se torch non è installato.
try:
    import torch               # Libreria deep learning (tensor operations)
    _TORCH_AVAILABLE = True    # Flag: torch è disponibile in questo ambiente
except ImportError:
    _TORCH_AVAILABLE = False   # torch non installato: useremo float plain

# ── Import opzionale del reward model classico ────────────────────────────────
# Se l'LLM fallisce, si fa fallback al calcolo deterministico originale.
try:
    from ai_agent import calculate_reward as _classic_reward  # Funzione di reward classica
    from ai_agent import DrillTracker                         # Tracker stato drill per il classico
    _CLASSIC_AVAILABLE = True   # Flag: il fallback classico è disponibile
except ImportError:
    _CLASSIC_AVAILABLE = False  # ai_agent non trovato: fallback sarà FALLBACK_REWARD=0.0

# =============================================================================
# CONFIGURAZIONE GLOBALE
# =============================================================================

# Legge la chiave API OpenRouter dalla variabile d'ambiente; "" se non trovata
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")

# URL base dell'API OpenRouter (compatibile con lo schema OpenAI)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Registro dei modelli: nome breve → identificatore completo per l'API
AVAILABLE_MODELS = {
    "llama":    "meta-llama/llama-3.1-8b-instruct",  # Modello veloce e leggero
    "deepseek": "deepseek/deepseek-chat"              # Modello alternativo
}

# Modello attivo di default se non viene specificato nel costruttore
DEFAULT_MODEL_KEY = "llama"

# Numero massimo di token che il modello può generare nella risposta.
# 80 token sono più che sufficienti per {"reward": -12.5, "reason": "vita persa"}
MAX_TOKENS       = 80

# Numero massimo di tentativi HTTP in caso di errore prima di arrendersi
MAX_RETRIES      = 2

# Secondi di attesa tra un retry e l'altro (evita di saturare l'API)
RETRY_DELAY_SEC  = 0.3

# Dimensione massima della cache LRU (numero di transizioni memorizzate)
CACHE_SIZE        = 512

# Fattore moltiplicativo applicato al reward prima di restituirlo.
# 1.0 = nessuna modifica (identità). Utile per scalare i reward per il training.
REWARD_SCALE      = 1.0

# Reward restituito quando sia l'LLM che il fallback classico falliscono
FALLBACK_REWARD   = 0.0

# Mappa da indice numerico al nome dell'azione (usata nel prompt per leggibilità)
ACTION_NAMES = {
    0: "IDLE",     # Non fare nulla (spreca ossigeno)
    1: "LEFT",     # Muoviti a sinistra
    2: "RIGHT",    # Muoviti a destra
    3: "DRILL_L",  # Fora a sinistra
    4: "DRILL_R",  # Fora a destra
    5: "DRILL_D",  # Fora verso il basso (azione principale del gioco)
}

# =============================================================================
# SYSTEM PROMPT STATICO
# =============================================================================
# Creato UNA SOLA VOLTA al caricamento del modulo e riusato in ogni chiamata.
# Vantaggi:
# - Non occupa CPU per ricreare la stringa ad ogni frame
# - Token invarianti → il modello sviluppa "memoria" del contesto statico (KV cache server-side)
# - Separazione netta tra contesto fisso (system) e dati variabili (user)

_SYSTEM_PROMPT = (
    "You are a reward model for Mr. Driller RL agent. "
    "Evaluate the state transition and return a reward float.\n"
    # Criteri di reward con valori numerici concreti: guidano il modello verso scale coerenti
    "REWARD CRITERIA:\n"
    "  +5 per cell descended (dy>0 = good)\n"              # Ogni cella scesa vale +5
    "  +2 per % oxygen gained (especially if oxy<40%)\n"   # Recupero ossigeno premiato
    "  +10 per 100 score gained\n"                         # Punti = +10 ogni 100
    "  -100 life lost\n"                                   # Morte = penalità pesante
    "  +300 level complete\n"                               # Livello completato = bonus enorme
    "  -10 hit unbreakable block\n"                        # Tentare di forare X = penalità
    "  -0.3 IDLE without reason\n"                         # Stare fermi = lieve penalità
    "  Typical range: -100 to +300\n"                      # Ancora per la scala dei valori
    # Formato obbligatorio: JSON con reward float e reason in italiano
    'Reply ONLY with compact JSON: {"reward":<float>,"reason":"<Italian, max 8 words>"}'
)

# =============================================================================
# LRU CACHE per reward (implementazione manuale con dict ordinato Python 3.7+)
# =============================================================================
# Stessa logica della cache nell'agente, ma applicata alle transizioni di stato.
# Chiave = hash MD5 della tupla (stato_prima, stato_dopo, azione)
# Valore = reward float già calcolato dall'LLM

# Dict globale: funge da cache LRU condivisa tra tutte le istanze di LLMRewardModel
_reward_cache: dict[str, float] = {}


def _make_cache_key(
    prev_y, prev_x, new_y, new_x,        # Posizioni prima/dopo
    prev_oxy, new_oxy,                    # Ossigeno prima/dopo (percentuale)
    prev_score, new_score,                # Punteggio prima/dopo
    prev_lives, new_lives,                # Vite prima/dopo
    action_idx,                           # Azione eseguita (0-5)
    is_hard_block,                        # True se il giocatore ha colpito un blocco X
    is_delayed_block,                     # True se presente un blocco bomba
    is_level_complete,                    # True se il livello è completato
) -> str:
    """
    Genera una chiave MD5 per la cache della transizione di stato.

    Discretizzazione applicata per aumentare i cache hit:
    - Ossigeno arrotondato all'intero (es: 78.3% → 78) → ignora variazioni sub-percentuali
    - Score diviso per 100 e cappato a 999 → es: 1523 e 1587 → entrambi diventano 15
    - Bool → int (0 o 1) per serializzazione uniforme

    La discretizzazione è un trade-off: più granulare = meno hit ma più accurato.
    """
    parts = (
        prev_y, prev_x, new_y, new_x,              # Posizioni esatte (int)
        int(prev_oxy), int(new_oxy),                # Ossigeno arrotondato all'intero
        min(prev_score  // 100, 999),               # Punteggio in centinaia, max 999
        min(new_score   // 100, 999),               # Stesso per nuovo punteggio
        prev_lives, new_lives,                      # Vite (tipicamente 1-3)
        action_idx,                                 # Indice azione (0-5)
        int(is_hard_block),                         # bool → 0 o 1
        int(is_delayed_block),                      # bool → 0 o 1
        int(is_level_complete),                     # bool → 0 o 1
    )
    # Converte la tupla in stringa e calcola l'MD5 → 32 caratteri hex
    return hashlib.md5(str(parts).encode()).hexdigest()


def _cache_get(key: str) -> Optional[float]:
    """
    Legge dalla cache LRU.
    L'operazione pop + re-insert sposta la chiave in coda al dict
    (= "usata di recente"), implementando la politica di eviction LRU.
    Restituisce il reward float se trovato, None altrimenti.
    """
    if key in _reward_cache:
        val = _reward_cache.pop(key)   # Rimuove dalla posizione attuale
        _reward_cache[key] = val       # Re-inserisce in fondo = più recente
        return val                     # Cache hit
    return None                        # Cache miss


def _cache_set(key: str, value: float) -> None:
    """
    Scrive nella cache LRU.
    Se la cache è piena (>= CACHE_SIZE), rimuove l'elemento meno recente
    (il primo nel dict, grazie all'ordine di inserimento garantito da Python 3.7+).
    """
    if len(_reward_cache) >= CACHE_SIZE:
        # next(iter(dict)) = prima chiave = meno recentemente usata
        del _reward_cache[next(iter(_reward_cache))]  # Eviction LRU
    _reward_cache[key] = value  # Inserisce il nuovo valore


# =============================================================================
# COSTRUZIONE PROMPT COMPRESSO (~120 token vs ~400 originale)
# =============================================================================

def build_compact_reward_prompt(
    prev_y: int, prev_x: int, new_y: int, new_x: int,   # Coordinate posizione
    prev_oxy: float, new_oxy: float,                      # Ossigeno prima/dopo
    prev_score: int, new_score: int,                      # Punteggio prima/dopo
    prev_lives: int, new_lives: int,                      # Vite prima/dopo
    action_idx: int,                                      # Azione eseguita
    is_hard_block: bool,                                  # Tentativo su blocco X
    is_delayed_block: bool,                               # Presenza bomba
    total_rows: int,                                      # Altezza totale mappa
    is_level_complete: bool,                              # Livello finito?
) -> str:
    """
    Costruisce il messaggio utente compresso per la valutazione del reward.
    Obiettivo: massimizzare l'informazione rilevante per frame token usato.

    Rispetto al prompt originale (~400 token) questo usa ~120 token perché:
    - I criteri di reward sono nel system prompt (non ripetuti qui)
    - Le transizioni sono espresse come delta (Δ) invece che valori assoluti
    - Solo i flag rilevanti vengono inclusi (quelli False sono omessi)
    - Tutto su 3 righe compatte invece di sezioni verbose
    """

    # ── Calcola le variazioni (delta) tra stato precedente e nuovo ────────────
    dy      = new_y - prev_y        # Variazione riga: positivo = sceso (buono nel gioco)
    d_oxy   = new_oxy - prev_oxy    # Variazione ossigeno: positivo = ha recuperato O₂
    d_score = new_score - prev_score  # Variazione punteggio: positivo = ha guadagnato punti

    # Profondità attuale come percentuale dell'altezza totale della mappa
    # Es: riga 30 su mappa alta 100 righe → depth=30.0%
    depth   = round(new_y / max(total_rows, 1) * 100, 1)

    # True se il giocatore si è fisicamente spostato (posizione cambiata)
    moved   = new_y != prev_y or new_x != prev_x

    # Nome leggibile dell'azione (es: 5 → "DRILL_D")
    action  = ACTION_NAMES.get(action_idx, str(action_idx))

    # ── Raccoglie solo i flag degli eventi significativi ─────────────────────
    # Lista vuota se nessun evento speciale: il prompt rimane compatto
    flags = []
    if is_level_complete:            flags.append("LEVEL_COMPLETE")        # Livello completato
    if new_lives < prev_lives:       flags.append(f"LIFE_LOST({prev_lives}→{new_lives})")  # Vita persa
    if is_hard_block:                flags.append("HIT_X")                 # Colpito blocco indistruttibile
    if is_delayed_block:             flags.append("DELAYED_BLOCK")         # Vicino a una bomba
    if not moved:                    flags.append("NOT_MOVED")             # Non si è spostato

    # Unisce i flag con " | " oppure "none" se lista vuota
    flag_str = " | ".join(flags) if flags else "none"

    # ── Genera avviso livello ossigeno ────────────────────────────────────────
    oxy_warn = ""
    if new_oxy < 20:   oxy_warn = "!!OXY_CRITICAL"  # Ossigeno quasi esaurito: urgentissimo
    elif new_oxy < 40: oxy_warn = "!OXY_LOW"        # Ossigeno basso: attenzione

    # ── Assembla il prompt su 3 righe ────────────────────────────────────────
    return (
        # Riga 1: azione eseguita, spostamento verticale, traiettoria posizione
        f"action={action} dy={dy:+d} pos=({prev_x},{prev_y})→({new_x},{new_y})\n"
        # Riga 2: ossigeno (con delta e avviso), punteggio (con delta), profondità
        f"oxy={prev_oxy:.0f}%→{new_oxy:.0f}%(Δ{d_oxy:+.0f}%){oxy_warn} "
        f"score={prev_score}→{new_score}(Δ{d_score:+d}) depth={depth}%\n"
        # Riga 3: eventi speciali (o "none")
        f"events:{flag_str}"
    )


# =============================================================================
# HELPER: PARSING RETRY-AFTER
# =============================================================================

def _parse_retry_after(resp: httpx.Response) -> float:
    """
    Estrae il tempo di attesa da un errore HTTP 429 (Too Many Requests).
    Prova 3 fonti in ordine di priorità, con fallback a 30 secondi.
    Aggiunge sempre MARGIN=1.0s come buffer di sicurezza per evitare di
    riprovare un millisecondo prima che il rate limit scada.
    """
    MARGIN = 1.0  # Buffer di sicurezza in secondi

    # ── Fonte 1: header standard HTTP "retry-after" ──────────────────────────
    raw = resp.headers.get("retry-after", "")
    if raw:
        try:
            return float(raw) + MARGIN
        except ValueError:
            pass  # Potrebbe essere una data RFC 2822, non un numero → salta

    # ── Fonte 2: header proprietario OpenRouter/OpenAI ───────────────────────
    raw = resp.headers.get("x-ratelimit-reset-requests", "")
    if raw:
        try:
            return float(raw.rstrip("s")) + MARGIN  # Rimuove suffisso "s" se presente
        except ValueError:
            pass

    # ── Fonte 3: campo "message" nel corpo JSON dell'errore ──────────────────
    try:
        body = resp.json()  # Deserializza il corpo come JSON
        msg  = body.get("error", {}).get("message", "")
        # Cerca il pattern "try again in X.Xs" usando regex
        m    = re.search(r'try again in ([\d.]+)s', msg)
        if m:
            return float(m.group(1)) + MARGIN
    except Exception:
        pass  # JSON malformato o campo assente → ignora

    # ── Fallback conservativo ────────────────────────────────────────────────
    return 30.0  # Aspetta 30 secondi se non riusciamo a determinare il tempo esatto


# =============================================================================
# CLIENT HTTP OTTIMIZZATO — classe privata (prefisso _ = non fa parte dell'API pubblica)
# =============================================================================

class _FastRewardHTTPClient:
    """
    Client HTTP riutilizzabile ottimizzato per il reward model.
    Stessa architettura di _FastHTTPClient dell'agente:
    - HTTP/2 con keepalive: evita TCP+TLS handshake ripetuti (~150ms risparmiati/call)
    - Streaming SSE con early exit su {"reward": X}: non aspetta la fine del token
    - Gestione automatica del rate limit 429 con cooldown
    """

    def __init__(self, model_key: str = DEFAULT_MODEL_KEY) -> None:
        """
        Inizializza e valida il client con il modello specificato.
        """
        # Controllo che il modello richiesto sia nel registro
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(
                f"Modello '{model_key}' non disponibile. "
                f"Scegli tra: {list(AVAILABLE_MODELS.keys())}"
            )
        self.model_key = model_key                    # Nome breve del modello
        self._model_id = AVAILABLE_MODELS[model_key]  # ID completo per l'API

        # Crea il client HTTP persistente (mantiene le connessioni aperte nel pool)
        self._client = httpx.Client(
            base_url = OPENROUTER_BASE_URL,
            headers  = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",  # Auth token
                "Content-Type":  "application/json",               # Body in JSON
                "HTTP-Referer":  "https://github.com/mr-driller-rl",  # Richiesto da OpenRouter
                "X-Title":       "Mr. Driller RL Reward Model",    # Nome app nella dashboard
            },
            http2   = True,                 # Abilita HTTP/2 per multiplexing e header compression
            timeout = httpx.Timeout(
                30.0,        # Timeout totale della richiesta (include attesa risposta)
                connect=5.0  # Timeout solo per l'apertura della connessione TCP
            ),
            limits  = httpx.Limits(
                max_keepalive_connections = 5,    # Pool di connessioni idle massimo
                keepalive_expiry          = 120,  # Chiudi connessione dopo 120s di inattività
            ),
        )
        self._url               = "/chat/completions"  # Endpoint relativo all'URL base
        self._retry_after_until = 0.0   # Timestamp: non chiamare prima di questo istante

        print(f"[FAST_REWARD:{model_key}] Pronto → {self._model_id}")

    def call(self, system: str, user: str) -> Optional[float]:
        """
        Chiama il modello LLM con streaming e restituisce il reward come float,
        oppure None se la chiamata fallisce completamente.

        Gestisce il cooldown per rate limit prima di eseguire la chiamata.
        """
        now = time.monotonic()  # Orologio monotono: non risente del cambio orario
        if self._retry_after_until > now:
            # Siamo nel periodo di cooldown → attendi prima di riprovare
            wait = self._retry_after_until - now
            print(f"[FAST_REWARD:{self.model_key}] Rate-limit cooldown: {wait:.1f}s")
            time.sleep(wait)  # Blocca il thread (sincrono)

        # Costruisce e invia la richiesta in streaming SSE
        payload = {
            "model":      self._model_id,  # Modello da usare su OpenRouter
            "max_tokens": MAX_TOKENS,       # Limite token risposta (80 sufficienti)
            "stream":     True,             # Abilita lo streaming SSE
            "messages": [
                {"role": "system", "content": system},  # Contesto fisso (criteri reward)
                {"role": "user",   "content": user},     # Transizione di stato da valutare
            ],
        }
        buffer = ""  # Buffer che accumula i delta di testo dallo stream SSE

        try:
            # Apre lo stream HTTP POST. Il with-block chiude automaticamente la connessione.
            with self._client.stream("POST", self._url, json=payload) as resp:

                # ── Gestione errori HTTP non-200 ─────────────────────────────
                if resp.status_code != 200:
                    resp.read()  # Forza la lettura del corpo (necessario per accedere a resp.text)
                    if resp.status_code == 429:
                        # Rate limit: calcola il cooldown e imposta il timestamp
                        retry_after = _parse_retry_after(resp)
                        self._retry_after_until = time.monotonic() + retry_after
                        print(f"[FAST_REWARD:{self.model_key}] 429 → attesa {retry_after:.1f}s")
                    else:
                        # Altro errore (401=auth, 500=server error, ecc.): logga e solleva
                        preview = resp.text[:200] if resp.text else "(no body)"
                        print(f"[FAST_REWARD:{self.model_key}] HTTP {resp.status_code}: {preview}")
                    resp.raise_for_status()  # Solleva httpx.HTTPStatusError

                # ── Lettura dello stream SSE riga per riga ───────────────────
                for line in resp.iter_lines():
                    # Le righe vuote sono separatori SSE standard → ignora
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]  # Rimuove i 6 caratteri del prefisso "data: "

                    # Il server segnala la fine dello stream con il token speciale [DONE]
                    if data_str == "[DONE]":
                        break

                    # Decodifica il chunk JSON del delta
                    try:
                        chunk  = json.loads(data_str)
                        # Estrae il contenuto del delta (può essere None → usiamo "")
                        delta  = chunk["choices"][0]["delta"].get("content") or ""
                        buffer += delta  # Aggiunge al buffer accumulato
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue  # Chunk malformato → salta silenziosamente

                    # ── Early exit: cerca {"reward": <numero>} nel buffer ────
                    # Pattern: ammette interi, float e negativi (es: -100, 12.5, 5)
                    # [^}]* consente la presenza del campo "reason" dopo il reward
                    m = re.search(
                        r'\{\s*"reward"\s*:\s*(-?[\d.]+)[^}]*\}',
                        buffer
                    )
                    if m:
                        try:
                            # Tenta di parsare il JSON completo trovato nel buffer
                            result = json.loads(m.group())
                            # ← chiude lo stream non appena il reward è disponibile
                            return float(result["reward"])
                        except (json.JSONDecodeError, KeyError, ValueError):
                            pass  # JSON ancora troncato → continua a leggere

        except httpx.HTTPStatusError:
            pass  # Già gestito con raise_for_status sopra
        except httpx.RequestError as e:
            # Errori di rete: timeout connessione, DNS failure, reset TCP, ecc.
            print(f"[FAST_REWARD:{self.model_key}] Request error: {e}")
        except Exception as e:
            # Cattura qualsiasi altro errore imprevisto
            print(f"[FAST_REWARD:{self.model_key}] Unexpected: {type(e).__name__}: {e}")

        # ── Tentativo finale: estrazione parziale dal buffer ─────────────────
        # Lo stream è terminato (o crashato), ma il buffer potrebbe contenere
        # almeno il valore numerico del reward (senza il JSON completo)
        m = re.search(r'"reward"\s*:\s*(-?[\d.]+)', buffer)
        if m:
            try:
                return float(m.group(1))  # Estrae solo il numero float
            except ValueError:
                pass
        return None  # Nessun reward recuperabile → il chiamante userà il fallback

    def close(self) -> None:
        """Chiude esplicitamente il client e rilascia le connessioni nel pool."""
        try:
            self._client.close()
        except Exception:
            pass


# =============================================================================
# CLASSE PUBBLICA: LLMRewardModel
# =============================================================================

class LLMRewardModel:
    """
    Reward model LLM ottimizzato per latenza. Drop-in replacement dell'originale
    LLMRewardModel: stessa firma del metodo get_reward(), stessa interfaccia stats().

    Architettura a 4 livelli:
      ① Cache LRU   → risposta immediata se la transizione è già stata vista
      ② Prompt LLM  → costruisce descrizione compatta della transizione
      ③ HTTP stream  → invia all'LLM, legge in streaming, chiude early
      ④ Fallback     → usa calculate_reward() classico se l'LLM fallisce

    Restituisce torch.Tensor([reward]) se PyTorch è disponibile, altrimenti float.
    Questo garantisce la compatibilità con framework RL come Stable Baselines 3.

    USO:
        from llm_reward_model_fast import LLMRewardModel
        reward_model = LLMRewardModel(model_key="llama")
        reward = reward_model.get_reward(
            prev_y=5, prev_x=4, new_y=6, new_x=4,
            prev_oxy=80.0, new_oxy=78.0,
            prev_score=100, new_score=150,
            prev_lives=3, new_lives=3,
            action_idx=5,
        )
    """

    def __init__(self, model_key: str = DEFAULT_MODEL_KEY, use_cache: bool = True) -> None:
        """
        Inizializza il reward model.

        Args:
            model_key:  Nome breve del modello ("llama" o "deepseek")
            use_cache:  Se True, abilita la cache LRU (consigliato per il training)
        """
        # Verifica obbligatoria: senza chiave API non si può fare nulla
        if not OPENROUTER_API_KEY:
            raise EnvironmentError(
                "[FAST_REWARD] OPENROUTER_API_KEY non impostata.\n"
                "Ottieni la chiave su https://openrouter.ai e imposta:\n"
                "  export OPENROUTER_API_KEY='sk-or-...'"
            )
        self.model_key  = model_key             # Salva il nome del modello
        self.use_cache  = use_cache             # Flag: abilita/disabilita la cache
        self._http      = _FastRewardHTTPClient(model_key)  # Client HTTP con keepalive

        # ── Metriche di performance ──────────────────────────────────────────
        self.call_count    = 0    # Chiamate effettive all'LLM (esclude cache hit)
        self.cache_hits    = 0    # Risposte servite dalla cache LRU
        self.error_count   = 0    # Errori totali (timeout, parsing, out-of-range)
        self.total_latency = 0.0  # Latenza totale accumulata in secondi

        # ── Mappa per la metrica di consistenza del modello ──────────────────
        # Struttura: {cache_key: [reward_1, reward_2, ...]}
        # Usata per calcolare quanto è coerente il modello su transizioni identiche
        self._consistency_map: dict[str, list[float]] = {}

        print(
            f"[FAST_REWARD:{model_key}] Reward model pronto | "
            f"cache={CACHE_SIZE} | streaming=True | early_exit=True"
        )

    def get_reward(
        self,
        prev_y: int, prev_x: int,        # Posizione precedente (riga, colonna)
        new_y: int,  new_x: int,          # Posizione nuova (riga, colonna)
        prev_oxy: float, new_oxy: float,  # Ossigeno prima e dopo (percentuale 0-100)
        prev_score: int, new_score: int,  # Punteggio prima e dopo
        prev_lives: int, new_lives: int,  # Vite prima e dopo
        action_idx: int,                  # Azione eseguita (0-5)
        is_hard_block: bool    = False,   # True se il giocatore ha colpito un blocco X
        is_delayed_block: bool = False,   # True se c'è un blocco bomba in gioco
        total_rows: int        = 100,     # Altezza totale della mappa (per calcolo depth%)
        is_level_complete: bool = False,  # True se il livello è stato completato
        **kwargs,                         # Argomenti extra ignorati (backward compat)
    ):
        """
        Calcola il reward per la transizione di stato (prev → new).

        Pipeline a 4 stadi:
          ① Cache check   → se la transizione è già nota, restituisce subito il reward
          ② Prompt build  → costruisce il messaggio compresso per l'LLM
          ③ LLM call      → chiama l'LLM in streaming con retry e early exit
          ④ Fallback      → se l'LLM fallisce, usa calculate_reward() classico o 0.0

        Returns:
            torch.Tensor([reward * REWARD_SCALE]) se torch disponibile, altrimenti float
        """

        # ── ① Cache check ────────────────────────────────────────────────────
        # Genera la chiave MD5 della transizione discretizzata
        cache_key = _make_cache_key(
            prev_y, prev_x, new_y, new_x,
            prev_oxy, new_oxy, prev_score, new_score,
            prev_lives, new_lives, action_idx,
            is_hard_block, is_delayed_block, is_level_complete,
        )

        if self.use_cache:
            cached = _cache_get(cache_key)  # Cerca nella cache LRU
            if cached is not None:
                self.cache_hits += 1
                # Registra per la metrica di consistenza
                self._consistency_map.setdefault(cache_key, []).append(cached)
                # Restituisce il reward scalato e wrappato (tensor o float)
                return self._wrap(cached * REWARD_SCALE)

        # ── ② Costruzione prompt compresso ───────────────────────────────────
        # Eseguito solo in caso di cache miss
        user_msg = build_compact_reward_prompt(
            prev_y, prev_x, new_y, new_x,
            prev_oxy, new_oxy, prev_score, new_score,
            prev_lives, new_lives, action_idx,
            is_hard_block, is_delayed_block, total_rows, is_level_complete,
        )

        # ── ③ Chiamata LLM con retry ─────────────────────────────────────────
        reward_val: Optional[float] = None  # Inizialmente nessun reward valido

        for attempt in range(MAX_RETRIES):
            t0      = time.perf_counter()                      # Timer ad alta risoluzione
            result  = self._http.call(_SYSTEM_PROMPT, user_msg)  # Chiamata HTTP in streaming
            latency = time.perf_counter() - t0                 # Misura latenza
            self.total_latency += latency                      # Accumula per la media
            self.call_count    += 1                            # Conta la chiamata

            if result is not None:
                # ── Risposta valida: salva, logga e interrompi il loop retry ──
                reward_val = result  # float estratto dallo stream

                if self.use_cache:
                    _cache_set(cache_key, reward_val)  # Salva nella cache per riuso futuro

                # Registra per la metrica di consistenza
                self._consistency_map.setdefault(cache_key, []).append(reward_val)

                # Log periodico ogni 50 chiamate (non loggare ogni frame = troppo verbose)
                if self.call_count % 50 == 0:
                    print(
                        f"[FAST_REWARD:{self.model_key}] "
                        f"step={self.call_count} reward={reward_val:+.2f} "
                        f"lat={latency*1000:.0f}ms cache_hits={self.cache_hits}"
                    )
                break  # ← Uscita anticipata dal loop: reward trovato, non servono altri retry

            # ── Risposta None: conta l'errore e riprova ──────────────────────
            self.error_count += 1
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SEC)  # Attendi prima del prossimo tentativo

        # ── ④ Fallback ───────────────────────────────────────────────────────
        if reward_val is None:
            # reward_val è None se tutti i MAX_RETRIES tentativi hanno fallito
            if _CLASSIC_AVAILABLE:
                # Usa il reward model deterministico originale (nessuna chiamata di rete)
                _dt = DrillTracker()  # Istanza fresca del tracker (stateless per questa call)
                return _classic_reward(
                    prev_y=prev_y, prev_x=prev_x, new_y=new_y, new_x=new_x,
                    prev_oxy=prev_oxy, new_oxy=new_oxy,
                    prev_score=prev_score, new_score=new_score,
                    prev_lives=prev_lives, new_lives=new_lives,
                    action_idx=action_idx, is_hard_block=is_hard_block,
                    is_delayed_block=is_delayed_block,
                    drill_tracker=_dt, total_rows=total_rows,
                    is_level_complete=is_level_complete,
                )
            # Ultimo fallback: reward neutro 0.0 (nessuna informazione disponibile)
            return self._wrap(FALLBACK_REWARD)

        # ── Ritorno normale ───────────────────────────────────────────────────
        # Applica il fattore di scala e incapsula nel tipo corretto
        return self._wrap(reward_val * REWARD_SCALE)

    def _wrap(self, value: float):
        """
        Incapsula il valore float nel tipo atteso dal framework RL.

        Se PyTorch è disponibile → restituisce torch.Tensor([value])
        Questo è il formato atteso da Stable Baselines 3, RLlib, ecc.

        Se PyTorch non è installato → restituisce il float direttamente
        (utile per testing o ambienti leggeri senza deep learning).
        """
        if _TORCH_AVAILABLE:
            import torch  # Import locale: evita import a livello modulo se non usato
            return torch.tensor([value])  # Tensor 1D con un singolo elemento
        return value  # Plain float

    def consistency_rate(self) -> float:
        """
        Misura la coerenza decisionale del modello reward su transizioni identiche.

        Metrica: percentuale di transizioni (cache_key) per cui il modello
        ha sempre assegnato lo stesso reward (arrotondato all'intero).

        Arrotondamento all'intero (round(v, 0)): considera coerente una variazione
        di ±0.5 sul reward (tolleranza per il ruolo stocastico dell'LLM).

        Un valore vicino a 1.0 indica un modello stabile e affidabile.
        Un valore basso suggerisce alta varianza: il training RL potrebbe divergere.
        """
        if not self._consistency_map:
            return 1.0  # Nessun dato ancora → assume perfetta consistenza

        # Per ogni transizione osservata, controlla se tutti i reward sono uguali
        consistent = sum(
            1 for calls in self._consistency_map.values()
            # set() converte la lista in insieme: se ha 1 solo elemento → sempre uguale
            if len(set(round(v, 0) for v in calls)) == 1
        )
        # Percentuale: transizioni consistenti / transizioni totali
        return consistent / len(self._consistency_map)

    def stats(self) -> dict:
        """
        Restituisce le metriche di performance del reward model.
        Utile per confrontare modelli, monitorare la qualità durante il training,
        e identificare problemi (alta error_rate, bassa consistency, latenza elevata).
        """
        total_reqs = self.call_count + self.cache_hits  # Totale richieste (LLM + cache)
        avg_lat    = self.total_latency / max(self.call_count, 1)  # Evita divisione per 0
        return {
            "model"          : self.model_key,
            "call_count"     : self.call_count,                                   # Chiamate LLM reali
            "cache_hits"     : self.cache_hits,                                   # Risposte da cache
            "cache_hit_rate" : round(self.cache_hits / max(total_reqs, 1), 3),    # Tasso cache hit
            "error_count"    : self.error_count,                                  # Errori totali
            "error_rate"     : round(self.error_count / max(self.call_count, 1), 3),  # Tasso errori
            "avg_latency_ms" : round(avg_lat * 1000, 1),                          # Latenza media ms
            "consistency"    : round(self.consistency_rate(), 3),                 # Coerenza modello
        }

    def clear_cache(self) -> None:
        """
        Svuota completamente la cache globale dei reward.
        Da chiamare tipicamente tra un episodio e l'altro,
        o quando si cambia livello/scenario di gioco.
        """
        _reward_cache.clear()  # Svuota il dict globale (condiviso tra tutte le istanze)
        print(f"[FAST_REWARD:{self.model_key}] Cache svuotata")

    def __del__(self) -> None:
        """
        Distruttore Python: chiamato dal garbage collector quando l'oggetto
        non ha più riferimenti attivi. Chiude il client HTTP per rilasciare
        le connessioni nel pool e prevenire resource leak.
        Il try/except evita eccezioni durante lo shutdown dell'interprete.
        """
        try:
            self._http.close()
        except Exception:
            pass