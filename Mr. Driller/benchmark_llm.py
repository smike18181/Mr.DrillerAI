"""
benchmark.py — Confronto tra modelli LLM per Mr. Driller
=========================================================

Questo script confronta le prestazioni di due modelli linguistici (LLM),
`llama` e `deepseek`, su due compiti distinti del gioco Mr. Driller:

  - Task AZIONE  : dato uno stato di gioco, quale azione deve compiere il giocatore?
  - Task REWARD  : quanto vale (numericamente) una transizione da uno stato a un altro?

Le metriche misurate sono:
  1. Latenza media (ms)       → quanto tempo impiega il modello a rispondere; meno è meglio
  2. Cache hit rate           → quante risposte arrivano dalla cache (indice di stabilità); più è meglio
  3. Tasso di errori/fallback → quante chiamate falliscono o tornano una risposta di riserva; meno è meglio
  4. Consistenza              → con lo stesso input, il modello produce sempre lo stesso output? più è meglio

DEFINIZIONE DI EPISODIO:
  Un episodio corrisponde a una run completa di un livello, dalla comparsa del giocatore
  fino al termine della partita. La run termina in uno di due modi:
    - VITTORIA : il giocatore raggiunge il blocco End (uscita del livello).
    - GAME OVER: le vite scendono a -1 (ossigeno esaurito più volte di quante fossero le vite).
  Morire e rinascere (revive) NON termina l'episodio: si perde solo una vita.
  Esempio: 3 morti + vittoria finale = 1 episodio completato con successo.

USO DA TERMINALE:
  python benchmark.py                          # esegue 10 episodi di default
  python benchmark.py --episodes 30           # esegue 30 episodi
  python benchmark.py --episodes 5 --models llama deepseek  # sceglie i modelli

PREREQUISITI:
  pip install httpx[http2]          # libreria HTTP con supporto HTTP/2
  export OPENROUTER_API_KEY='sk-or-...'  # chiave API di OpenRouter
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORT STANDARD
# ──────────────────────────────────────────────────────────────────────────────

import argparse   # per leggere gli argomenti passati da riga di comando (es. --episodes 30)
import random     # per generare stati di gioco casuali nei test mock
import time       # per misurare il tempo trascorso (latenza)
import os         # per leggere variabili d'ambiente (es. OPENROUTER_API_KEY)
import sys        # per manipolare il path di importazione dei moduli Python
from typing import NamedTuple  # per definire tuple con campi nominati (tipo record immutabile)

# Aggiunge la directory corrente in cima al path di ricerca dei moduli Python.
# Questo assicura che i file locali (llm_agent.py, llm_reward_model.py, ecc.)
# vengano trovati anche se lo script viene lanciato da un'altra cartella.
sys.path.insert(0, os.path.dirname(__file__))

# Importa la classe LLMAgent e la sua cache interna (_cache) dal modulo llm_agent.
# LLMAgent è la classe che usa un LLM per scegliere le azioni nel gioco.
# _cache è il dizionario condiviso che memorizza risposte già ottenute (evita chiamate duplicate).
from llm_agent       import LLMAgent,       _cache as _agent_cache

# Importa la classe LLMRewardModel e la sua cache dal modulo llm_reward_model.
# LLMRewardModel usa un LLM per stimare il "valore" (reward) di una transizione di stato.
from llm_reward_model import LLMRewardModel, _reward_cache


# ──────────────────────────────────────────────────────────────────────────────
# OGGETTI MOCK
# Questi oggetti simulano le strutture del gioco reale (blocchi, giocatore, livello)
# senza dover avviare il gioco vero. Servono per il benchmark in isolamento.
#
# I mock replicano fedelmente l'interfaccia pubblica delle classi reali:
#   block.py     → Block, Classic, Solo, Unbreakable, Delayed, Pill, End
#   character.py → Character
#   level.py     → generateLvl
# ──────────────────────────────────────────────────────────────────────────────

# ── Probabilità di generazione blocchi ─────────────────────────────────────
# Queste costanti replicano i parametri di default di generateLvl() in level.py.
# Sono usate da _make_level() per generare le griglia mock con la stessa
# distribuzione statistica del gioco reale.
#
#   PILL_PROB       → 5%  : probabilità per cella di generare una Pill
#   SOLO_PROB       → 10% : probabilità per cella di generare un blocco Solo
#   UNBREAKABLE_PROB→ 5%  : probabilità per cella di generare un blocco Unbreakable (X)
#   DELAYED_PROB    → 10% : probabilità per cella di generare un blocco Delayed (cristallo)
#   Classic (~70%)  → default: tutti i casi rimanenti producono un blocco Classic colorato
#
# Nota: i controlli nel codice di generateLvl avvengono in questo ordine:
#   1. Pill (con vincoli aggiuntivi) → 2. Solo → 3. Unbreakable → 4. Delayed → 5. Classic
# I controlli successivi si applicano solo se quelli precedenti sono falliti,
# quindi le probabilità effettive si sovrappongono parzialmente.
# Per il mock, usiamo le soglie raw così come definite in generateLvl.
PILL_PROB        = 5    # soglia Pill (PillRn < pillP)
SOLO_PROB        = 10   # soglia Solo (SoloRn < SoloP)
UNBREAKABLE_PROB = 5    # soglia Unbreakable (UnbreakableRn < UnbreakableP)
DELAYED_PROB     = 10   # soglia Delayed (DelayedRn < DelayedP)

# ── Costanti Pill di livello ────────────────────────────────────────────────
# generateLvl() limita le Pill con due parametri aggiuntivi che il mock rispetta:
#   PILL_MAX_PER_LEVEL   → massimo 7 Pill per livello intero (PillPL=7)
#   PILL_MIN_ROW_SPACING → minimo 20 righe di distanza tra due Pill consecutive (PillMLE=20)
# Questi vincoli evitano che il livello sia pieno di ricariche di ossigeno,
# rendendo il gioco troppo facile. Il mock li replica per generare livelli realistici.
PILL_MAX_PER_LEVEL   = 7    # PillPL: slot massimo di Pill nell'intero livello
PILL_MIN_ROW_SPACING = 20   # PillMLE: righe minime tra due Pill (distanza minima)

# ── Costanti di struttura livello ───────────────────────────────────────────
# Righe "vuote" all'inizio del livello: le prime EMPTY_TOP_ROWS righe sono sempre
# Classic con HP=0 (aria pura). Corrispondono al buffer sopra lo spawn del giocatore.
# In generateLvl: "if i in range(5): → Classic(j, i, 1, 0)"
EMPTY_TOP_ROWS = 5    # righe 0..4: zona di spawn/buffer, sempre vuota


class MockBlock:
    """
    Simula un singolo blocco del livello di Mr. Driller.

    Replica l'interfaccia pubblica della classe Block (e delle sue sottoclassi)
    definita in block.py, così i modelli LLM ricevono input identici a quelli
    del gioco reale. I metodi pubblici disponibili sono:

        hpAccess()    → int   : punti vita del blocco (0 = blocco vuoto/aria)
        typeAccess()  → str   : tipo del blocco (vedi TIPI DISPONIBILI sotto)
        ColorAccess() → str   : colore del blocco (rilevante solo per i Classic)

    TIPI DISPONIBILI (allineati ai tipi reali di block.py):
      - "classic"     : blocco colorato standard, partecipa alle chain reaction (HP=1)
      - "solo"        : blocco senza chain reaction, 1 solo colpo per distruggerlo (HP=1)
      - "unbreakable" : blocco quasi indestruttibile, richiede 5 colpi (HP=5)
      - "delayed"     : blocco cristallo con countdown di 2s dopo il primo colpo (HP=5)
      - "pill"        : capsula ossigeno, si raccoglie camminandoci sopra (HP=1)
      - "end"         : uscita del livello (blocco speciale nelle ultime righe)

    NOTA: il tipo "empty" (blocco vuoto) NON esiste come tipo separato in block.py.
    Un blocco vuoto è semplicemente un Classic (o qualsiasi tipo) con HP=0.
    Questa distinzione è fondamentale: hpAccess()==0 indica aria, non il tipo.
    La classe MockBlock non prevede più "empty" come tipo autonomo.
    """

    # Colori disponibili per i blocchi Classic.
    # In generateLvl: randint(1, colors) con colors=4 → colori 1, 2, 3, 4.
    # Qui usiamo stringhe descrittive per chiarezza nei prompt LLM.
    COLORS = ["red", "blue", "green", "yellow"]

    def __init__(
        self,
        force_empty:       bool = False,
        force_pill:        bool = False,
        force_end:         bool = False,
        force_type:        str  = None,
    ):
        """
        Costruttore del blocco mock. Può essere guidato da flag o lasciato casuale.

        Parametri:
          force_empty : se True, crea un blocco vuoto (qualsiasi tipo con HP=0).
                        Usato per le prime 5 righe del livello (zona di spawn).
          force_pill  : se True, crea sempre una capsula ossigeno (tipo "pill", HP=1).
                        Usato da _make_level() quando la cella deve contenere una Pill.
          force_end   : se True, crea un blocco di fine livello (tipo "end", HP=1).
                        Usato da _make_level() per le righe terminali del livello.
          force_type  : se fornita come stringa ("solo", "unbreakable", ecc.),
                        impone esplicitamente quel tipo con i valori HP corretti.
                        Usato da _make_level() per rispettare le probabilità di generateLvl.
        """

        if force_empty:
            # ── Blocco vuoto (aria) ───────────────────────────────────────────
            # In generateLvl le prime 5 righe sono Classic(j, i, 1, 0):
            # tipo "classic" con forceHP=0 → HP=0 = cella vuota.
            # Qui usiamo "classic" come tipo per coerenza, HP=0 indica l'assenza di materia.
            self._type  = "classic"
            self._hp    = 0
            self._color = self.COLORS[0]  # colore arbitrario, non rilevante (hp=0 = aria)

        elif force_pill:
            # ── Capsula ossigeno ──────────────────────────────────────────────
            # In block.py: Pill(posX, posY) → hp=1, tipo "pill".
            # Si distrugge (hp → 0) al primo contatto con il giocatore.
            self._type  = "pill"
            self._hp    = 1
            self._color = self.COLORS[0]  # le Pill non hanno colore rilevante

        elif force_end:
            # ── Blocco di fine livello ────────────────────────────────────────
            # In block.py: End(posX, posY) → hp=1, tipo "end".
            # Toccarlo genera un evento pygame di cambio livello.
            self._type  = "end"
            self._hp    = 1
            self._color = self.COLORS[0]  # i blocchi End non hanno colore rilevante

        elif force_type is not None:
            # ── Tipo imposto esplicitamente ───────────────────────────────────
            # Usato da _make_level() per rispettare le probabilità di generateLvl.
            # Ogni tipo ha un HP iniziale fisso, uguale a block.py:
            #   classic     → HP=1 (1 colpo per distruggerlo)
            #   solo        → HP=1 (identico al classic ma senza chain reaction)
            #   unbreakable → HP=5 (5 colpi: Unbreakable.__init__ usa forceHP=5)
            #   delayed     → HP=5 (Delayed usa forceHP=5, ma la logica è diversa)
            #   pill        → HP=1 (si raccoglie camminandoci sopra)
            #   end         → HP=1 (uscita del livello)
            self._type = force_type
            hp_map = {
                "classic":     1,
                "solo":        1,
                "unbreakable": 5,
                "delayed":     5,
                "pill":        1,
                "end":         1,
            }
            self._hp    = hp_map.get(force_type, 1)  # default 1 per tipi sconosciuti
            # Il colore è rilevante solo per i Classic (chain reaction usa il colore per
            # determinare quali blocchi adiacenti propagano il colpo).
            self._color = random.choice(self.COLORS) if force_type == "classic" else self.COLORS[0]

        else:
            # ── Tipo completamente casuale ────────────────────────────────────
            # Usato solo quando si crea un MockBlock senza contesto di livello.
            # Le probabilità seguono approssimativamente quelle di generateLvl:
            #   ~70% Classic, ~10% Solo, ~5% Unbreakable, ~10% Delayed, ~5% Pill.
            rn = random.randint(0, 100)  # numero casuale per selezionare il tipo

            if rn < PILL_PROB:
                # 0-4 → Pill (5%)
                self._type = "pill"
                self._hp   = 1
            elif rn < PILL_PROB + SOLO_PROB:
                # 5-14 → Solo (10%)
                self._type = "solo"
                self._hp   = 1
            elif rn < PILL_PROB + SOLO_PROB + UNBREAKABLE_PROB:
                # 15-19 → Unbreakable (5%)
                self._type = "unbreakable"
                self._hp   = 5
            elif rn < PILL_PROB + SOLO_PROB + UNBREAKABLE_PROB + DELAYED_PROB:
                # 20-29 → Delayed (10%)
                self._type = "delayed"
                self._hp   = 5
            else:
                # 30-100 → Classic (~70%, tutti i casi rimanenti)
                self._type = "classic"
                self._hp   = 1

            # Il colore è significativo solo per i Classic (determina la chain reaction).
            # Per gli altri tipi, assegniamo comunque un colore per uniformità dell'interfaccia.
            self._color = random.choice(self.COLORS)

    # ── Metodi accessori (getter) ────────────────────────────────────────────
    # Replicano ESATTAMENTE l'interfaccia pubblica della classe Block reale in block.py.
    # I modelli LLM (e il codice di select_action / get_reward) usano solo questi metodi
    # per leggere lo stato di un blocco: mai gli attributi privati direttamente.

    def hpAccess(self) -> int:
        """
        Restituisce i punti vita correnti del blocco.
        Valore chiave: hp==0 significa che la cella è vuota (aria).
        Corrisponde a Block.hpAccess() in block.py.
        """
        return self._hp

    def typeAccess(self) -> str:
        """
        Restituisce il tipo del blocco come stringa.
        Valori possibili: "classic", "solo", "unbreakable", "delayed", "pill", "end".
        Corrisponde a Block.typeAccess() in block.py.
        """
        return self._type

    def ColorAccess(self) -> str:
        """
        Restituisce il colore del blocco come stringa (es. "red", "blue").
        Rilevante principalmente per i blocchi "classic": determina con quali
        blocchi adiacenti può avvenire una chain reaction (stessa catena di colore).
        Corrisponde a Block.ColorAccess() in block.py (maiuscola: 'C').
        """
        return self._color


class MockPlayer:
    """
    Simula il personaggio giocante di Mr. Driller (classe Character in character.py).

    Replica l'interfaccia pubblica di Character usata dai modelli LLM e dal codice
    di benchmark. I metodi disponibili sono:

        posAcc()   → (int, int) : posizione (riga, colonna) nella griglia
        oxyAcc()   → float      : ossigeno residuo in percentuale [0-100]
        livesAcc() → int        : vite rimaste
        scoreAcc() → int        : punteggio accumulato

    NOTA: imgIndAcc(), IdlingAcc(), blocksFallenAcc(), climbAcc(), fallAcc(),
    airTimerAcc() sono altri getter di Character; qui non vengono replicati.
    I modelli LLM usano solo posizione, ossigeno, vite e punteggio per decidere l'azione.

    DEFINIZIONE DI EPISODIO (richiamata qui per contesto):
      Un episodio = una run completa del livello. Le vite di MockPlayer partono
      da un valore casuale tra 1 e 3, simulando un punto diverso dentro l'episodio
      (vite piene all'inizio, una vita al limite del game over). Game over = lives < 0.
    """

    def __init__(
        self,
        y:     int   = None,   # riga nella griglia (None = casuale nel range valido)
        x:     int   = None,   # colonna nella griglia (None = casuale nel range valido)
        oxy:   float = None,   # ossigeno % (None = casuale tra 15 e 100)
        lives: int   = None,   # vite rimaste (None = casuale tra 1 e 3)
        score: int   = None,   # punteggio (None = casuale tra 0 e 10000)
        rows:  int   = 50,     # altezza della griglia, usata per calcolare i limiti Y
        cols:  int   = 9,      # larghezza della griglia, usata per calcolare i limiti X
    ):
        """
        Costruttore del giocatore mock.

        I valori None vengono rimpiazzati con valori casuali nei range plausibili del gioco:
          - Y: tra EMPTY_TOP_ROWS (zona di spawn) e rows-5 (evita il fondo del livello)
          - X: tra 1 e cols-2 (evita le colonne con possibili pareti ai bordi)
          - Ossigeno: tra 15% e 100% (15% è già zona critica; 0% = morte)
          - Vite: 1-3 (in Character.__init__ lives=2 di default; resetScore usa lives=2)
          - Punteggio: 0-10000 (range realistico durante il gioco)
        """
        # ── Posizione nella griglia ───────────────────────────────────────────
        # Y: il giocatore parte dalla riga EMPTY_TOP_ROWS-1 (riga 4) per simulare
        # la zona di spawn; non va oltre rows-5 per evitare il blocco End.
        self._y     = y     if y     is not None else random.randint(EMPTY_TOP_ROWS, rows - 5)

        # X: evita la colonna 0 e cols-1 in caso di implementazioni future con pareti.
        # In generateLvl non ci sono pareti esplicite ai bordi, ma i valori 1..cols-2
        # rappresentano comunque le colonne "interne" dove il gioco si svolge.
        self._x     = x     if x     is not None else random.randint(1, cols - 2)

        # ── Ossigeno ──────────────────────────────────────────────────────────
        # In Character: __oxygen parte da 100 e scende di 1% per tick (funct=1),
        # di 20% per blocco X colpito (funct=2). La pill aggiunge 20% (funct=3).
        # 15% è la soglia sotto cui il personaggio mostra la texture "low oxygen".
        self._oxy   = oxy   if oxy   is not None else random.uniform(15, 100)

        # ── Vite ──────────────────────────────────────────────────────────────
        # In Character: starts con il valore passato a __init__ (es. 2 di default).
        # Ogni morte (revive) consuma una vita. Game over quando scende sotto 0.
        # Il mock usa 1-3 per simulare stati diversi nell'arco di un episodio:
        #   lives=3 → inizio episodio, pieno di vite
        #   lives=1 → ultima vita, situazione critica
        self._lives = lives if lives is not None else random.randint(1, 3)

        # ── Punteggio ─────────────────────────────────────────────────────────
        # In Character: parte da 0, viene incrementato da AddScore() ad ogni
        # blocco distrutto (10 pt per classic/solo/unbreakable, 20 pt per pill).
        self._score = score if score is not None else random.randint(0, 10000)

    # ── Metodi accessori (getter) ────────────────────────────────────────────
    # Replicano l'interfaccia pubblica di Character usata dal codice LLM.

    def posAcc(self) -> tuple[int, int]:
        """
        Restituisce la posizione del giocatore come tupla (riga, colonna).
        CONVENZIONE: Y (riga) prima di X (colonna), uguale a Character.posAcc().
        Corrisponde a Character.posAcc() in character.py.
        """
        return (self._y, self._x)

    def oxyAcc(self) -> float:
        """
        Restituisce il livello di ossigeno corrente in percentuale [0.0–100.0].
        Valore critico: sotto 20% il personaggio mostra la texture "low oxygen".
        Corrisponde a Character.oxyAcc() in character.py.
        """
        return self._oxy

    def livesAcc(self) -> int:
        """
        Restituisce il numero di vite rimaste.
        Game over quando scende sotto 0 (dopo revive() con lives già a 0).
        Corrisponde a Character.livesAcc() in character.py.
        """
        return self._lives

    def scoreAcc(self) -> int:
        """
        Restituisce il punteggio accumulato dal giocatore nell'episodio corrente.
        Corrisponde a Character.scoreAcc() in character.py.
        """
        return self._score


def _make_level(rows: int = 50, cols: int = 9) -> list[list[MockBlock]]:
    """
    Genera una griglia 2D di MockBlock che simula un livello di Mr. Driller.

    La logica di generazione REPLICA FEDELMENTE generateLvl() in level.py,
    usando le stesse probabilità, gli stessi vincoli sulle Pill, e la stessa
    struttura a zone (vuoto in cima, gioco nel mezzo, fine in basso).

    STRUTTURA DEL LIVELLO (uguale a generateLvl):
      Righe 0..EMPTY_TOP_ROWS-1  (prime 5):
        → Classic con HP=0 (aria pura). Buffer sopra il punto di spawn del giocatore.
          In generateLvl: "if i in range(5): Classic(j, i, 1, 0)"

      Righe EMPTY_TOP_ROWS..rows-1 (zona di gioco):
        → Ogni cella viene generata con le probabilità di generateLvl (nell'ordine):
            1. Pill       : se PillRn < PILL_PROB e slot disponibili e distanza rispettata
            2. Solo       : se SoloRn < SOLO_PROB
            3. Unbreakable: se UnbreakableRn < UNBREAKABLE_PROB
            4. Delayed    : se DelayedRn < DELAYED_PROB
            5. Classic    : altrimenti (default, ~70% delle celle)

      Righe rows..rows+2 (zona di fine livello):
        → Blocchi End. In generateLvl: "else: → End(j, i)"
          Il giocatore raggiungendo questa zona completa il livello (episodio vinto).

    Parametri:
      rows : numero di righe della zona di gioco (esclude le righe End finali)
      cols : numero di colonne della griglia

    Ritorna:
      Una lista di liste di MockBlock: level[riga][colonna].
      Totale righe = rows + 3 (zona vuota + zona gioco + zona End).
    """
    level = []              # griglia 2D: lista di righe, ogni riga è una lista di MockBlock

    # Contatori Pill: replicano le variabili locali di generateLvl()
    pill_slots_left      = PILL_MAX_PER_LEVEL    # PillPL: quante Pill si possono ancora generare
    rows_since_last_pill = -1                    # come LineRemainBeforePill in generateLvl;
                                                  # -1 = già "pronto" dalla prima riga di gioco

    total_rows = EMPTY_TOP_ROWS + rows + 3
    # Righe totali della griglia:
    #   EMPTY_TOP_ROWS righe vuote (0..4)
    #   rows righe di gioco effettivo (5..rows+4)
    #   3 righe di blocchi End in fondo (rows+5..rows+7)

    for r in range(total_rows):
        row = []    # riga corrente: verrà riempita colonna per colonna

        # ── Zona 1: Righe vuote (buffer spawn) ───────────────────────────────
        # Righe 0..EMPTY_TOP_ROWS-1 (di default 0..4).
        # In generateLvl: "if i in range(5): Classic(j, i, 1, 0)"
        # HP=0 → cella vuota (aria). Il giocatore appare in (x=3, y=4).
        if r < EMPTY_TOP_ROWS:
            for c in range(cols):
                row.append(MockBlock(force_empty=True))
                # force_empty=True → tipo "classic" con HP=0 (aria pura)

        # ── Zona 2: Righe di gioco effettivo ─────────────────────────────────
        # Righe EMPTY_TOP_ROWS..EMPTY_TOP_ROWS+rows-1 (di default 5..54).
        # In generateLvl: "elif i in range(lines): → generazione casuale per ogni cella"
        elif r < EMPTY_TOP_ROWS + rows:
            # Aggiorna il contatore di distanza dalla Pill precedente.
            # In generateLvl: "LineRemainBeforePill -= 1" all'inizio di ogni riga di gioco.
            rows_since_last_pill -= 1

            for c in range(cols):
                # Genera numeri casuali per ogni tipo speciale, come in generateLvl:
                #   PillRn, UnbreakableRn, DelayedRn, SoloRn → randint(0, 100)
                pill_rn        = random.randint(0, 100)
                solo_rn        = random.randint(0, 100)
                unbreakable_rn = random.randint(0, 100)
                delayed_rn     = random.randint(0, 100)

                # ── Controllo Pill (priorità 1, con vincoli aggiuntivi) ───────
                # In generateLvl: "if PillRn < pillP and PillPL != 0 and LineRemainBeforePill < 0"
                # Tre condizioni devono essere tutte soddisfatte simultaneamente:
                if (pill_rn < PILL_PROB
                        and pill_slots_left > 0
                        and rows_since_last_pill < 0):
                    pill_slots_left -= 1               # consuma uno slot Pill (PillPL -= 1)
                    rows_since_last_pill = PILL_MIN_ROW_SPACING  # resetta la distanza minima
                    row.append(MockBlock(force_pill=True))

                # ── Controllo Solo (priorità 2) ───────────────────────────────
                # In generateLvl: "elif SoloRn < SoloP"
                # Verificato solo se Pill non è stata generata in questa cella.
                elif solo_rn < SOLO_PROB:
                    row.append(MockBlock(force_type="solo"))

                # ── Controllo Unbreakable (priorità 3) ───────────────────────
                # In generateLvl: "elif UnbreakableRn < UnbreakableP"
                elif unbreakable_rn < UNBREAKABLE_PROB:
                    row.append(MockBlock(force_type="unbreakable"))

                # ── Controllo Delayed (priorità 4) ────────────────────────────
                # In generateLvl: "elif DelayedRn < DelayedP"
                elif delayed_rn < DELAYED_PROB:
                    row.append(MockBlock(force_type="delayed"))

                # ── Classic (default, priorità 5 = tutti i casi rimanenti) ───
                # In generateLvl: "else: Classic(j, i, randint(1, colors), 1)"
                # È il tipo più comune (~70% delle celle).
                else:
                    row.append(MockBlock(force_type="classic"))

        # ── Zona 3: Righe di fine livello (End) ──────────────────────────────
        # Righe EMPTY_TOP_ROWS+rows e oltre.
        # In generateLvl: "else: End(j, i)"
        # Il giocatore raggiungendo questa zona vince il livello (episodio completato).
        else:
            for c in range(cols):
                row.append(MockBlock(force_end=True))

        level.append(row)   # aggiunge la riga completata alla griglia

    return level    # restituisce la griglia 2D completa


def _make_mock_states(n: int, rows: int = 50) -> list[tuple]:
    """
    Genera una lista di `n` stati di gioco casuali per il benchmark del task AZIONE.

    Ogni stato è una tupla (player, level, last_action_idx) dove:
      - player          : un MockPlayer con posizione e status casuali
      - level           : una griglia MockBlock generata con le probabilità di generateLvl
      - last_action_idx : indice dell'ultima azione eseguita (0–5)

    INDICI AZIONE (da llm_agent.py, allineati con le azioni di Character in character.py)

    DEFINIZIONE DI EPISODIO (richiamata per contesto):
      Ogni stato rappresenta un istante all'interno di un episodio di gioco.
      Un episodio termina quando il giocatore vince (raggiunge End) o va in game over.
      Gli stati mock rappresentano snapshot casuali durante l'episodio, non l'intero flusso.
      Per un benchmark end-to-end reale si dovrebbe simulare il loop completo;
      per semplicità computazionale il mock campiona stati indipendenti.

    Parametri:
      n    : quanti stati generare (= quante chiamate a select_action() verranno fatte)
      rows : altezza della zona di gioco del livello (passata a _make_level)

    Ritorna:
      Lista di n tuple (MockPlayer, list[list[MockBlock]], int).
    """
    states = []
    for _ in range(n):
        level  = _make_level(rows=rows)      # genera un livello casuale con le probabilità reali
        player = MockPlayer(rows=rows)       # crea un giocatore con posizione e status casuali
        last   = random.randint(0, 5)        # ultima azione: uno degli 6 indici possibili
        states.append((player, level, last))
    return states


def _make_mock_transitions(n: int, rows: int = 50) -> list[dict]:
    """
    Genera una lista di `n` transizioni di stato casuali per il benchmark del task REWARD.

    Una transizione descrive cosa è successo tra uno step e il successivo:
    posizione precedente e nuova, ossigeno prima e dopo, punteggio, vite, ecc.
    Ogni transizione è un dizionario con le chiavi attese da get_reward().

    CALCOLO DELLE TRANSIZIONI (basato su Character in character.py):
      - LEFT  (idx=1): posX-1
      - RIGHT (idx=2): posX+1
      - DRILL_D (idx=5): posY+1 (il personaggio scende di una riga)
      - IDLE/DRILL_L/DRILL_R: posizione invariata

      Ossigeno:
        - scende di 1% per tick (funct=1 in updateOxygen)
        - scende di 20% colpendo un Unbreakable (funct=2)
        - sale di 20% raccogliendo una Pill (funct=3)
        Il mock simula variazioni casuali nell'intervallo [-5, +20] per rappresentare
        tutti e tre i casi sopra.

      Punteggio:
        - +10 per Classic/Solo/Unbreakable (AddScore(10))
        - +20 per Pill (AddScore(20))
        Il mock usa +0..200 per rappresentare un intero step di gioco (catene incluse).

      Vite:
        - diminuiscono di 1 solo se l'ossigeno scende a 0 (revive).
        - il mock simula questo con una probabilità del 5% di perdita vita per transizione.

    DEFINIZIONE DI EPISODIO (richiamata per contesto):
      Come per gli stati, ogni transizione è un campione indipendente di un momento
      dell'episodio. In un benchmark reale si simulerebbero tutte le transizioni sequenziali
      dall'inizio alla fine dell'episodio (vittoria o game over).

    Parametri:
      n    : quanti scenari di transizione generare
      rows : altezza del livello (per i limiti di posizione)

    Ritorna:
      Lista di n dizionari pronti da passare come **kwargs a get_reward().
    """
    transitions = []
    for _ in range(n):
        # ── Posizione di partenza del giocatore ──────────────────────────────
        # Y: tra EMPTY_TOP_ROWS (zona di spawn) e rows-5 (evita End blocks).
        prev_y = random.randint(EMPTY_TOP_ROWS, rows - 5)
        # X: tra 1 e 7 (colonne interne, evita i bordi).
        prev_x = random.randint(1, 7)

        # ── Azione eseguita ───────────────────────────────────────────────────
        # 0=IDLE, 1=LEFT, 2=RIGHT, 3=DRILL_L, 4=DRILL_R, 5=DRILL_D
        action = random.randint(0, 5)

        # ── Calcolo posizione risultante dall'azione ──────────────────────────
        # Basato sulla logica di Character.move() e Character.fall() in character.py:
        #   DRILL_D (5) → il personaggio perfora verso il basso e poi cade: posY += 1
        #   LEFT (1)    → si sposta di una colonna a sinistra: posX -= 1
        #   RIGHT (2)   → si sposta di una colonna a destra: posX += 1
        #   tutti gli altri (0, 3, 4) → posizione invariata
        new_y = prev_y + (1 if action == 5 else 0)
        new_x = prev_x + (-1 if action == 1 else (1 if action == 2 else 0))
        # Clamp: X deve restare nel range giocabile [1, 7] (evita uscita dai bordi)
        new_x = max(1, min(7, new_x))

        # ── Ossigeno ──────────────────────────────────────────────────────────
        # prev_oxy: tra 15% e 100% (15% è già zona critica in Character.Anim)
        prev_oxy = random.uniform(15, 100)
        # Variazione: negativa per consumo, positiva per Pill raccolta.
        # Range [-5, +20] copre:
        #   -1% per tick normale (funct=1)
        #   -20% per blocco X colpito (funct=2) → non raggiunto con -5, approssimazione
        #   +20% per Pill raccolta (funct=3) → coperto con +20
        d_oxy    = random.uniform(-5, 20)
        new_oxy  = max(0.0, min(100.0, prev_oxy + d_oxy))  # clamp in [0, 100]

        # ── Punteggio ─────────────────────────────────────────────────────────
        # prev_score: tra 0 e 5000 (metà del range massimo del mock)
        prev_score = random.randint(0, 5000)
        # Incremento: 0-200 per simulare:
        #   0 pt → nessun blocco distrutto in questo step (IDLE o movimento su vuoto)
        #   10 pt → blocco Classic/Solo/Unbreakable distrutto (AddScore(10))
        #   20 pt → Pill raccolta (AddScore(20))
        #   >20 pt → chain reaction (più blocchi Classic dello stesso colore distrutti)
        new_score = prev_score + random.randint(0, 200)

        # ── Vite ──────────────────────────────────────────────────────────────
        # prev_lives: tra 1 e 3 (il giocatore ha sempre almeno 1 vita in gioco attivo).
        prev_lives = random.randint(1, 3)
        # Perdita vita: 5% di probabilità per transizione (simula morte per ossigeno).
        # In Character.revive(), la vita viene consumata solo se ossigeno scende a 0.
        # Il mock non calcola l'ossigeno in modo sequenziale, quindi approssima con 5%.
        # Vincolo: non scende sotto 1 (game over solo quando va sotto 0, ma nel mock
        # non gestiamo game over completo per ogni transizione).
        new_lives = (prev_lives - 1
                     if random.random() < 0.05 and prev_lives > 1
                     else prev_lives)

        # ── Aggiunge la transizione come dizionario ───────────────────────────
        transitions.append(dict(
            prev_y  = prev_y,
            prev_x  = prev_x,
            new_y   = new_y,
            new_x   = new_x,
            prev_oxy     = prev_oxy,
            new_oxy      = new_oxy,
            prev_score   = prev_score,
            new_score    = new_score,
            prev_lives   = prev_lives,
            new_lives    = new_lives,
            action_idx   = action,
            # is_hard_block: True se il blocco colpito era Unbreakable.
            # In block.py: Unbreakable.hit() scala solo HP (5→4→...→0).
            # 5% di probabilità nel mock (raro, come UNBREAKABLE_PROB).
            is_hard_block    = random.random() < 0.05,
            # is_delayed_block: True se il blocco era Delayed (cristallo).
            # In block.py: Delayed.hit() avvia il countdown (isDisappearing=True).
            # 10% di probabilità nel mock (come DELAYED_PROB).
            is_delayed_block = random.random() < 0.10,
            total_rows       = rows,
            # is_level_complete: True se il giocatore ha raggiunto un blocco End.
            # In block.py: End.hit() → changeLvl() → evento pygame USEREVENT.
            # 2% di probabilità nel mock (raro: accade solo alla fine del livello).
            is_level_complete = random.random() < 0.02,
        ))
    return transitions


# ──────────────────────────────────────────────────────────────────────────────
# STRUTTURA RISULTATI
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkResult(NamedTuple):
    """
    Contenitore immutabile con tutti i risultati di un benchmark per un modello.
    Usa NamedTuple così i campi sono accessibili per nome (es. result.model)
    e l'oggetto è hashable e leggero.
    """
    model:           str    # nome del modello ("llama" o "deepseek")
    task:            str    # tipo di task: "action" (azione) o "reward" (valore)
    n_episodes:      int    # numero totale di episodi eseguiti
    avg_latency_ms:  float  # latenza media in millisecondi per risposta
    cache_hit_rate:  float  # frazione di risposte servite dalla cache [0.0 – 1.0]
    error_rate:      float  # frazione di chiamate che hanno generato un errore [0.0 – 1.0]
    consistency:     float  # frazione di risposte identiche per lo stesso input [0.0 – 1.0]
    score:           float  # punteggio composito su scala 0–100 (più alto = meglio)


def _composite_score(
    latency_ms:     float,
    cache_hit_rate: float,
    error_rate:     float,
    consistency:    float,
) -> float:
    """
    Calcola un punteggio composito da 0 a 100 che riassume le prestazioni
    complessive di un modello su tutte le metriche. Più alto è il punteggio,
    migliore è il modello.

    Formula con pesi:
      - Latenza    (40%): normalizzata su 2000 ms. 0 ms → 40 pt; 2000+ ms → 0 pt
      - Error rate (30%): 0 errori → 30 pt; 100% errori → 0 pt
      - Consistenza(20%): 100% consistente → 20 pt; 0% → 0 pt
      - Cache hit  (10%): 100% cache hit → 10 pt; 0% → 0 pt

    La latenza ha peso maggiore perché in un loop RL real-time
    (~10–30 step/secondo) è la metrica più critica.

    Parametri:
      latency_ms     : latenza media in ms
      cache_hit_rate : tasso di cache hit [0–1]
      error_rate     : tasso di errori [0–1]
      consistency    : tasso di consistenza [0–1]

    Ritorna:
      float arrotondato a 2 decimali nell'intervallo [0.0, 100.0]
    """
    # Punteggio latenza: più è bassa, più punti ottieni (min clamp a 0 per sicurezza)
    lat_score   = max(0.0, 40.0 * (1.0 - min(latency_ms / 2000.0, 1.0)))

    # Punteggio errori: inversamente proporzionale al tasso di errore
    err_score   = 30.0 * (1.0 - min(error_rate, 1.0))

    # Punteggio consistenza: direttamente proporzionale
    cons_score  = 20.0 * consistency

    # Punteggio cache: direttamente proporzionale
    cache_score = 10.0 * cache_hit_rate

    # Somma tutti i contributi e arrotonda a 2 decimali
    return round(lat_score + err_score + cons_score + cache_score, 2)


# ──────────────────────────────────────────────────────────────────────────────
# RUNNER DEL BENCHMARK
# ──────────────────────────────────────────────────────────────────────────────

def run_action_benchmark(
    model_keys:          list[str],
    n_episodes:          int,
    consistency_repeats: int = 3,
) -> list[BenchmarkResult]:
    """
    Esegue il benchmark del task AZIONE per tutti i modelli indicati.

    Funzionamento:
      1. Genera `n_episodes` stati mock casuali (ognuno = snapshot di un momento
         all'interno di un episodio di gioco, dove episodio = run completa del livello)
      2. Per ogni modello:
         a. Svuota la cache (ogni modello riparte da zero)
         b. Chiama select_action() su tutti gli stati
         c. Esegue un test di consistenza: prende i primi n_consistency stati,
            svuota la cache e li esegue `consistency_repeats` volte per vedere
            se il modello risponde sempre allo stesso modo allo stesso input
         d. Raccoglie le statistiche e calcola il punteggio composito

    Parametri:
      model_keys          : lista di chiavi modello (es. ["llama", "deepseek"])
      n_episodes          : numero di stati da testare per ogni modello
      consistency_repeats : quante volte ripetere gli stati di consistenza

    Ritorna:
      Lista di BenchmarkResult, uno per ogni modello.
    """
    # Intestazione visiva del benchmark
    print("\n" + "═" * 60)
    print(f"  BENCHMARK AZIONE  |  {n_episodes} episodi  |  modelli: {model_keys}")
    print("═" * 60)

    # Genera gli stati di gioco mock una sola volta (condivisi tra i modelli)
    states           = _make_mock_states(n_episodes)

    # Numero di stati usati per il test di consistenza: 10% degli episodi, minimo 3
    n_consistency    = max(3, n_episodes // 10)

    # Prende i primi n_consistency stati per il test di consistenza
    consistency_states = states[:n_consistency]

    # Lista che conterrà i risultati di ogni modello
    results          = []

    for model_key in model_keys:
        print(f"\n▶ Modello: {model_key.upper()}")
        print("  " + "─" * 40)

        # Svuota la cache condivisa prima di ogni modello per un confronto equo
        _agent_cache.clear()

        # Crea l'agente LLM con il modello specificato
        agent = LLMAgent(model_key=model_key)

        # ── Fase episodi normali ─────────────────────────────────────────────
        print(f"  Esecuzione {n_episodes} stati...")
        for i, (player, level, last_action) in enumerate(states):
            # Chiede all'agente di scegliere un'azione dato lo stato corrente
            action, reason = agent.select_action(player, level, last_action)

            # Stampa un aggiornamento ogni ~20% degli episodi
            if (i + 1) % max(1, n_episodes // 5) == 0:
                # Mostra azione scelta e i primi 30 caratteri della motivazione
                print(f"  [{i+1}/{n_episodes}] → {action} ({reason[:30]})")

        # ── Fase test di consistenza ─────────────────────────────────────────
        # Ripete gli stessi stati con cache azzerata per misurare la riproducibilità.
        # Un buon modello dovrebbe restituire la stessa azione per lo stesso stato
        # anche in chiamate separate (senza risultati in cache).
        print(f"  Test consistenza ({n_consistency} stati × {consistency_repeats} volte)...")
        for repeat in range(consistency_repeats):
            _agent_cache.clear()  # forza ogni ripetizione a fare nuove chiamate API
            for player, level, last_action in consistency_states:
                agent.select_action(player, level, last_action)

        # ── Raccolta statistiche finali ──────────────────────────────────────
        s = agent.stats()  # dizionario con avg_latency_ms, cache_hit_rate, error_rate, consistency

        # Calcola il punteggio composito con i quattro valori
        score = _composite_score(
            s["avg_latency_ms"],
            s["cache_hit_rate"],
            s["error_rate"],
            s["consistency"],
        )

        # Crea il record di risultato e lo aggiunge alla lista
        results.append(BenchmarkResult(
            model          = model_key,
            task           = "action",
            n_episodes     = n_episodes,
            avg_latency_ms = s["avg_latency_ms"],
            cache_hit_rate = s["cache_hit_rate"],
            error_rate     = s["error_rate"],
            consistency    = s["consistency"],
            score          = score,
        ))

        # Stampa riassunto del modello appena testato
        print(
            f"  ✓ lat={s['avg_latency_ms']}ms  cache={s['cache_hit_rate']:.1%}  "
            f"err={s['error_rate']:.1%}  consist={s['consistency']:.1%}  "
            f"SCORE={score:.1f}/100"
        )

    return results  # lista di BenchmarkResult per ogni modello


def run_reward_benchmark(
    model_keys:          list[str],
    n_episodes:          int,
    consistency_repeats: int = 3,
) -> list[BenchmarkResult]:
    """
    Esegue il benchmark del task REWARD per tutti i modelli indicati.

    Struttura identica a run_action_benchmark() ma usa transizioni di stato
    (non stati singoli) e chiama get_reward() invece di select_action().

    Una transizione descrive: dove era il giocatore, dove si trova adesso,
    quanti punti ha guadagnato, quanto ossigeno ha, ecc.
    Il modello deve stimare un valore numerico (reward) per quella transizione.

    DEFINIZIONE DI EPISODIO (richiamata per contesto):
      Ogni transizione è uno step di un episodio. Un episodio = run completa
      del livello. La transizione di fine episodio avrà is_level_complete=True
      (vittoria) o new_lives < 0 (game over). Il mock genera transizioni
      indipendenti e non sequenziali: approssimazione ragionevole per il benchmark.

    Parametri:
      model_keys          : lista di chiavi modello
      n_episodes          : numero di transizioni da testare
      consistency_repeats : quante volte ripetere le transizioni di consistenza

    Ritorna:
      Lista di BenchmarkResult, uno per ogni modello.
    """
    print("\n" + "═" * 60)
    print(f"  BENCHMARK REWARD  |  {n_episodes} episodi  |  modelli: {model_keys}")
    print("═" * 60)

    # Genera le transizioni mock una sola volta
    transitions          = _make_mock_transitions(n_episodes)

    # Numero di transizioni per il test di consistenza (10% degli episodi, min 3)
    n_consistency        = max(3, n_episodes // 10)

    # Prende le prime n_consistency transizioni per il test di consistenza
    consistency_trans    = transitions[:n_consistency]

    results              = []

    for model_key in model_keys:
        print(f"\n▶ Modello: {model_key.upper()}")
        print("  " + "─" * 40)

        # Svuota la cache del reward model prima di ogni modello
        _reward_cache.clear()

        # Crea il reward model LLM con il modello specificato
        reward_model = LLMRewardModel(model_key=model_key)

        # ── Fase episodi normali ─────────────────────────────────────────────
        print(f"  Esecuzione {n_episodes} transizioni...")
        for i, trans in enumerate(transitions):
            # Spacchetta il dizionario come argomenti keyword per get_reward()
            reward = reward_model.get_reward(**trans)

            # Stampa un aggiornamento ogni ~20% degli episodi
            if (i + 1) % max(1, n_episodes // 5) == 0:
                # Converte il reward in float (potrebbe essere un tensore PyTorch)
                val = float(reward) if not hasattr(reward, 'item') else reward.item()
                print(f"  [{i+1}/{n_episodes}] → reward={val:+.2f}")  # +/- indica il segno

        # ── Fase test di consistenza ─────────────────────────────────────────
        print(f"  Test consistenza ({n_consistency} transizioni × {consistency_repeats} volte)...")
        for repeat in range(consistency_repeats):
            _reward_cache.clear()   # forza nuove chiamate API per ogni ripetizione
            for trans in consistency_trans:
                reward_model.get_reward(**trans)

        # ── Raccolta statistiche finali ──────────────────────────────────────
        s = reward_model.stats()
        score = _composite_score(
            s["avg_latency_ms"],
            s["cache_hit_rate"],
            s["error_rate"],
            s["consistency"],
        )
        results.append(BenchmarkResult(
            model          = model_key,
            task           = "reward",
            n_episodes     = n_episodes,
            avg_latency_ms = s["avg_latency_ms"],
            cache_hit_rate = s["cache_hit_rate"],
            error_rate     = s["error_rate"],
            consistency    = s["consistency"],
            score          = score,
        ))
        print(
            f"  ✓ lat={s['avg_latency_ms']}ms  cache={s['cache_hit_rate']:.1%}  "
            f"err={s['error_rate']:.1%}  consist={s['consistency']:.1%}  "
            f"SCORE={score:.1f}/100"
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# STAMPA RISULTATI
# ──────────────────────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float = 100.0, width: int = 20) -> str:
    """
    Genera una barra ASCII proporzionale al valore dato rispetto al massimo.

    Esempi:
      _bar(50)  → "██████████░░░░░░░░░░"  (50% piena)
      _bar(100) → "████████████████████"  (100% piena)
      _bar(0)   → "░░░░░░░░░░░░░░░░░░░░"  (vuota)

    Parametri:
      value   : valore da rappresentare
      max_val : valore massimo della scala (default 100)
      width   : larghezza totale della barra in caratteri (default 20)

    Ritorna:
      Stringa di `width` caratteri (mix di █ e ░).
    """
    # Calcola quante celle riempire: proporzione clamped a [0, width]
    filled = int(width * min(value / max_val, 1.0))
    # Concatena celle piene e celle vuote
    return "█" * filled + "░" * (width - filled)


def print_results_table(
    action_results: list[BenchmarkResult],
    reward_results: list[BenchmarkResult],
) -> None:
    """
    Stampa sul terminale la tabella completa dei risultati del benchmark,
    inclusi i vincitori per ciascun task e il vincitore assoluto.

    Struttura della tabella:
      ① Sezione TASK AZIONE: tutti i modelli ordinati per score decrescente
      ② Sezione TASK REWARD: tutti i modelli ordinati per score decrescente
      ③ Sezione VINCITORI: miglior modello per azione, per reward, in assoluto
      ④ Dettaglio metriche del vincitore assoluto
      ⑤ Raccomandazione finale su quale modello usare per quale file

    Parametri:
      action_results : lista di BenchmarkResult per il task azione
      reward_results : lista di BenchmarkResult per il task reward
    """

    def _row(r: BenchmarkResult) -> str:
        """
        Formatta una riga della tabella per un singolo risultato.
        Aggiunge una medaglia 🥇 se il modello ha il punteggio più alto nel suo task.
        """
        # Controlla se questo risultato è il migliore nel suo task
        medal = "🥇" if r.score == max(
            rr.score for rr in (action_results if r.task == "action" else reward_results)
        ) else "  "

        # Costruisce la stringa formattata con tutti i valori allineati
        return (
            f"  {medal} {r.model:<10} "           # nome modello, allineato a sinistra su 10 char
            f"lat={r.avg_latency_ms:>7.1f}ms  "   # latenza con 1 decimale, allineata a destra
            f"cache={r.cache_hit_rate:>5.1%}  "   # cache hit in percentuale
            f"err={r.error_rate:>5.1%}  "         # error rate in percentuale
            f"consist={r.consistency:>5.1%}  "    # consistenza in percentuale
            f"│ SCORE {r.score:>5.1f}/100  {_bar(r.score)}"  # punteggio + barra visiva
        )

    width = 90  # larghezza totale del box della tabella in caratteri

    print("\n")
    # ── Bordo superiore della tabella ────────────────────────────────────────
    print("╔" + "═" * width + "╗")
    # ── Titolo centrato ───────────────────────────────────────────────────────
    print("║" + "  📊 RISULTATI BENCHMARK — Mr. Driller LLM Models".center(width) + "║")
    print("╠" + "═" * width + "╣")

    # ── Sezione Task Azione ───────────────────────────────────────────────────
    print("║" + "  🎮 TASK: SELEZIONE AZIONE".ljust(width) + "║")
    print("║" + ("  " + "─" * (width - 4)).ljust(width) + "║")
    # Ordina i risultati per score decrescente (il migliore in cima)
    for r in sorted(action_results, key=lambda x: x.score, reverse=True):
        print("║" + _row(r).ljust(width) + "║")

    print("╠" + "═" * width + "╣")

    # ── Sezione Task Reward ───────────────────────────────────────────────────
    print("║" + "  💰 TASK: CALCOLO REWARD".ljust(width) + "║")
    print("║" + ("  " + "─" * (width - 4)).ljust(width) + "║")
    for r in sorted(reward_results, key=lambda x: x.score, reverse=True):
        print("║" + _row(r).ljust(width) + "║")

    print("╠" + "═" * width + "╣")

    # ── Calcolo vincitori ─────────────────────────────────────────────────────

    # Migliore per il task azione: il BenchmarkResult con score massimo
    best_action = max(action_results, key=lambda r: r.score)

    # Migliore per il task reward: il BenchmarkResult con score massimo
    best_reward = max(reward_results, key=lambda r: r.score)

    # Migliore assoluto: somma degli score di azione + reward per ogni modello
    all_results = action_results + reward_results

    # Dizionario: {nome_modello: somma_score_totale}
    best_model_by_model = {}
    for r in all_results:
        key = r.model
        # Accumula lo score di ogni task per questo modello
        best_model_by_model[key] = best_model_by_model.get(key, 0) + r.score

    # Il modello con la somma di score più alta è il vincitore assoluto
    best_overall = max(best_model_by_model, key=best_model_by_model.get)

    # ── Stampa sezione vincitori ──────────────────────────────────────────────
    print("║" + "  🏆 VINCITORI".ljust(width) + "║")
    print("║" + "".ljust(width) + "║")
    print("║" + f"  🎮 Miglior modello per AZIONE : {best_action.model.upper():<12} (score={best_action.score:.1f})".ljust(width) + "║")
    print("║" + f"  💰 Miglior modello per REWARD : {best_reward.model.upper():<12} (score={best_reward.score:.1f})".ljust(width) + "║")
    # Score medio = somma_totale / 2 (due task)
    print("║" + f"  ⭐ Miglior modello in assoluto: {best_overall.upper():<12} (score medio={best_model_by_model[best_overall]/2:.1f})".ljust(width) + "║")
    print("║" + "".ljust(width) + "║")

    # ── Dettaglio metriche del vincitore assoluto ─────────────────────────────
    print("║" + "  📌 Dettaglio metriche vincitore assoluto:".ljust(width) + "║")
    for r in all_results:
        if r.model == best_overall:
            # Etichetta del task: "AZIONE" o "REWARD"
            tag = "AZIONE" if r.task == "action" else "REWARD"
            print("║" + f"     {tag}: lat={r.avg_latency_ms:.0f}ms  err={r.error_rate:.1%}  consist={r.consistency:.1%}".ljust(width) + "║")

    # ── Bordo inferiore della tabella ─────────────────────────────────────────
    print("╚" + "═" * width + "╝")
    print()

    # ── Raccomandazione finale ────────────────────────────────────────────────
    print("📝 RACCOMANDAZIONE:")
    if best_action.model == best_reward.model:
        # Se lo stesso modello vince entrambi i task, si può usare ovunque
        print(f"   Usa {best_action.model.upper()} per entrambi i task (azione + reward).")
    else:
        # Altrimenti, suggerisce di usare modelli diversi per i due file
        print(f"   Usa {best_action.model.upper()} in llm_agent_fast.py  (DEFAULT_MODEL_KEY = \"{best_action.model}\")")
        print(f"   Usa {best_reward.model.upper()} in llm_reward_model_fast.py  (DEFAULT_MODEL_KEY = \"{best_reward.model}\")")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — viene eseguito solo se si lancia direttamente questo file
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Funzione principale: legge gli argomenti da riga di comando,
    valida la chiave API, ed esegue i benchmark richiesti.
    """
    # Crea il parser degli argomenti da riga di comando
    parser = argparse.ArgumentParser(
        description="Benchmark LLM per Mr. Driller (azione + reward)",
        formatter_class=argparse.RawDescriptionHelpFormatter,  # preserva la formattazione di __doc__
        epilog=__doc__,   # mostra il docstring del modulo come epilogo nell'help
    )

    # Argomento --episodes / -n : numero di episodi da eseguire (default 10)
    parser.add_argument(
        "--episodes", "-n",
        type    = int,
        default = 30,
        help    = "Numero di episodi mock per modello (default: 10)",
    )

    # Argomento --models / -m : quali modelli confrontare (default: entrambi)
    parser.add_argument(
        "--models", "-m",
        nargs   = "+",                        # accetta uno o più valori
        default = ["llama", "deepseek"],       # default: tutti e due i modelli
        choices = ["llama", "deepseek"],       # solo questi due valori sono validi
        help    = "Modelli da confrontare (default: tutti e 2)",
    )

    # Argomento --task : quale task benchmarkare (default: entrambi)
    parser.add_argument(
        "--task",
        choices = ["action", "reward", "both"],
        default = "both",
        help    = "Quale task benchmarkare (default: both)",
    )

    # Argomento --consistency-repeats : quante volte ripetere il test di consistenza
    parser.add_argument(
        "--consistency-repeats",
        type    = int,
        default = 3,
        help    = "Quante volte ripetere gli stati di consistenza (default: 3)",
    )

    # Analizza gli argomenti passati da riga di comando
    args = parser.parse_args()

    # ── Verifica che la chiave API sia impostata ──────────────────────────────
    if not os.environ.get("OPENROUTER_API_KEY"):
        # Senza chiave API non si può chiamare OpenRouter → errore fatale
        print("❌ OPENROUTER_API_KEY non impostata.")
        print("   Ottieni la chiave su https://openrouter.ai")
        print("   Poi esegui: export OPENROUTER_API_KEY='sk-or-...'")
        sys.exit(1)   # esce con codice di errore 1

    # ── Banner di avvio ───────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════╗
║     Mr. Driller — LLM Model Benchmark                   ║
║     Modelli : {str(args.models):<40}║
║     Episodi : {args.episodes:<43}║
║     Task    : {args.task:<43}║
╚══════════════════════════════════════════════════════════╝
    """)

    # Fissa il seed casuale per riproducibilità: stesso seed → stessi stati mock
    random.seed(42)

    # Liste che raccoglieranno i risultati dei due task
    action_results: list[BenchmarkResult] = []
    reward_results: list[BenchmarkResult] = []

    # ── Esegue il benchmark del task AZIONE se richiesto ──────────────────────
    if args.task in ("action", "both"):
        action_results = run_action_benchmark(
            model_keys          = args.models,
            n_episodes          = args.episodes,
            consistency_repeats = args.consistency_repeats,
        )

    # ── Esegue il benchmark del task REWARD se richiesto ──────────────────────
    if args.task in ("reward", "both"):
        reward_results = run_reward_benchmark(
            model_keys          = args.models,
            n_episodes          = args.episodes,
            consistency_repeats = args.consistency_repeats,
        )

    # ── Gestione task singolo: duplica i risultati per la stampa comparativa ──
    # Se è stato eseguito solo un task, usa quegli stessi risultati per entrambe
    # le sezioni della tabella (evita una tabella a metà vuota)
    if not action_results:
        action_results = reward_results  # solo reward → mostra reward in entrambe le sezioni
    if not reward_results:
        reward_results = action_results  # solo action → mostra action in entrambe le sezioni

    # ── Stampa la tabella finale con tutti i risultati ────────────────────────
    print_results_table(action_results, reward_results)


# Punto di ingresso dello script: main() viene chiamata solo se si esegue
# direttamente questo file (non se viene importato come modulo)
if __name__ == "__main__":
    main()