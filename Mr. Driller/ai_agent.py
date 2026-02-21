"""
ai_agent.py — Rainbow DQN per Mr. Driller (v5.4 — Reward Rebalancing)

AGGIUNTO v5.4-PER-PERSIST:
  PrioritizedReplayMemory.save_async(path)
      Snapshot numpy nel main thread (GPU->CPU), poi scrittura npz compresso
      in thread daemon. Non blocca mai il training loop.
  PrioritizedReplayMemory.load(path)
      Ricarica transizioni + priorità originali + frame + _max_p.
      Il beta-annealing e le priorità riprendono esattamente da dove erano.
"""

import os
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
from collections import deque, namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_STRATEGIC_VALUES = {
    "end":         1.0,
    "pill":        0.9,
    "classic":     0.7,
    "delayed":     0.6,
    "solo":        0.5,
    "unbreakable": 0.1,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def get_local_window_tensor(level, pX, pY, window_h=11, window_w=9):
    dy, dx     = window_h // 2, window_w // 2
    rows, cols = len(level), len(level[0])
    pX, pY     = int(pX), int(pY)
    grid       = np.zeros((6, window_h, window_w), dtype=np.float32)

    i_idx = np.arange(window_h, dtype=np.float32)
    j_idx = np.arange(window_w, dtype=np.float32)
    rel_y = (i_idx - dy).reshape(-1, 1)
    rel_x = np.abs(j_idx - dx).reshape(1, -1)
    grid[4] = 1.0 / (1.0 + np.sqrt((rel_y * 0.8) ** 2 + (rel_x * 2.5) ** 2))

    abs_y_arr = (pY - dy + np.arange(window_h)).astype(np.int32)
    abs_x_arr = (pX - dx + np.arange(window_w)).astype(np.int32)
    out_y     = (abs_y_arr < 0) | (abs_y_arr >= rows)
    out_x     = (abs_x_arr < 0) | (abs_x_arr >= cols)

    for i in range(window_h):
        ay = abs_y_arr[i]
        if out_y[i]:
            grid[0, i, :] = 0.1
            grid[2, i, :] = 1.0
            continue
        level_row = level[ay]
        for j in range(window_w):
            ax = abs_x_arr[j]
            if out_x[j]:
                grid[0, i, j] = 0.1
                grid[2, i, j] = 1.0
                continue
            block  = level_row[ax]
            hp     = block.hpAccess()
            b_type = block.typeAccess()
            if hp <= 0 and b_type != "delayed":
                grid[5, i, j] = 1.0
                continue
            grid[0, i, j] = _STRATEGIC_VALUES.get(b_type, 0.5)
            if b_type == "classic":
                my_color      = block.ColorAccess()
                grid[1, i, j] = my_color / 5.0
                combo = 0.0
                for ny, nx in ((ay-1,ax),(ay+1,ax),(ay,ax-1),(ay,ax+1)):
                    if 0<=ny<rows and 0<=nx<cols:
                        nb = level[ny][nx]
                        if nb.typeAccess()=="classic" and nb.hpAccess()>0 and nb.ColorAccess()==my_color:
                            combo += 0.25
                grid[1, i, j] += combo
            if b_type == "delayed":
                is_active = getattr(block, 'idAcc', lambda: False)()
                if is_active:
                    timer_val     = getattr(block, 'getTimer', lambda: 2)()
                    grid[2, i, j] = float(timer_val) / 2.0
                else:
                    grid[2, i, j] = 1.0
            elif b_type == "unbreakable":
                grid[2, i, j] = float(hp) / 5.0
            else:
                grid[2, i, j] = float(hp)
            is_falling = getattr(block, 'isFalling', lambda: False)()
            is_shaking = getattr(block, 'isShaking', lambda: False)()
            urgency = 0.0
            if is_falling:
                ry = i - dy
                urgency = 0.9 if -1<=ry<=1 else (0.3 if ry<-1 else 0.6)
            elif is_shaking:
                ry = i - dy
                urgency = 0.7 if ry<0 else (0.4 if ry==0 else 0.2)
            grid[3, i, j] = urgency

    return torch.from_numpy(grid).float()


def get_internal_state_vector(player, total_level_rows, total_level_cols,
                               level_data, last_action_idx, n_actions):
    state = []
    pY, pX = player.posAcc()
    oxy    = player.oxyAcc()
    state += [pX/max(total_level_cols,1), pY/max(total_level_rows,1),
              oxy/100.0, max(0,player.livesAcc())/3.0,
              min(player.scoreAcc()/10000.0,1.0), min(pY/100.0,1.0)]
    state += [1.0 if oxy<20 else 0.0, 1.0 if oxy<50 else 0.0]
    d=[0.0]*4; di=player.imgIndAcc()
    if 0<=di<4: d[di]=1.0
    state += d
    a=[0.0]*n_actions
    if 0<=last_action_idx<n_actions: a[last_action_idx]=1.0
    state += a
    sur=[0.0]*4
    if isinstance(level_data,list) and level_data:
        try:
            if pX>0:                    sur[0]=min(level_data[pY][pX-1].hpAccess(),1.0)
            if pX<total_level_cols-1:   sur[1]=min(level_data[pY][pX+1].hpAccess(),1.0)
            if pY<total_level_rows-1:   sur[2]=min(level_data[pY+1][pX].hpAccess(),1.0)
            if pY>0:                    sur[3]=min(level_data[pY-1][pX].hpAccess(),1.0)
        except Exception: pass
    state += sur
    pill=[0.0]*4
    if isinstance(level_data,list) and level_data:
        try:
            if pX>0 and level_data[pY][pX-1].typeAccess()=="pill":                  pill[0]=1.0
            if pX<total_level_cols-1 and level_data[pY][pX+1].typeAccess()=="pill": pill[1]=1.0
            if pY<total_level_rows-1 and level_data[pY+1][pX].typeAccess()=="pill": pill[2]=1.0
            if pY>0 and level_data[pY-1][pX].typeAccess()=="pill":                  pill[3]=1.0
        except Exception: pass
    state += pill
    state += [1.0 if player.fallAcc() else 0.0,
              player.airTimerAcc()/60.0,
              1.0 if player.climbAcc()>0 else 0.0, 0.0]
    while len(state)<32: state.append(0.0)
    return torch.tensor(state[:32], dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# 2. NOISY LINEAR
# ─────────────────────────────────────────────────────────────────────────────
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.zeros(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))
        mu_r = 1.0/math.sqrt(in_features)
        self.weight_mu.data.uniform_(-mu_r, mu_r)
        self.weight_sigma.data.fill_(std_init/math.sqrt(in_features))
        self.bias_mu.data.uniform_(-mu_r, mu_r)
        self.bias_sigma.data.fill_(std_init/math.sqrt(out_features))
        self.sample_noise()

    @staticmethod
    def _f(x): return x.sign()*x.abs().sqrt()

    def sample_noise(self):
        dev = self.weight_mu.device
        ei  = self._f(torch.randn(self.in_features,  device=dev))
        ej  = self._f(torch.randn(self.out_features, device=dev))
        self.weight_epsilon.set_(ej.outer(ei))
        self.bias_epsilon.set_(ej)

    def forward(self, x):
        if self.training:
            return F.linear(x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu   + self.bias_sigma   * self.bias_epsilon)
        return F.linear(x, self.weight_mu, self.bias_mu)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ARCHITETTURA
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False),nn.BatchNorm2d(ch),nn.ReLU(True),
            nn.Conv2d(ch,ch,3,padding=1,bias=False),nn.BatchNorm2d(ch))
        self.act = nn.ReLU(True)
    def forward(self, x): return self.act(self.net(x)+x)


class SpatialAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch,1,1)
    def forward(self, x): return x*torch.sigmoid(self.conv(x))


class DrillerDuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions, state_dim):
        super().__init__()
        c,h,w = input_shape
        self.cnn  = nn.Sequential(
            nn.Conv2d(c,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU(True))
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.attn = SpatialAttention(64)
        flat      = 64*h*w
        self.fc_vars = nn.Sequential(
            nn.Linear(state_dim,128),nn.ReLU(True),
            nn.Linear(128,256),nn.ReLU(True))
        combined = flat+256
        self.val = nn.Sequential(NoisyLinear(combined,512),nn.ReLU(True),NoisyLinear(512,1))
        self.adv = nn.Sequential(NoisyLinear(combined,512),nn.ReLU(True),NoisyLinear(512,n_actions))
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, grid, vars):
        x = self.attn(self.res2(self.res1(self.cnn(grid)))).flatten(1)
        c = torch.cat([x, self.fc_vars(vars)], 1)
        val,adv = self.val(c),self.adv(c)
        return val+(adv-adv.mean(1,keepdim=True))

    def sample_noise(self):
        for m in self.modules():
            if isinstance(m,NoisyLinear): m.sample_noise()


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRIORITIZED REPLAY MEMORY  +  PERSISTENZA BUFFER
# ─────────────────────────────────────────────────────────────────────────────
class SumTree:
    def __init__(self, cap):
        self.cap  = cap
        self.tree = np.zeros(2*cap-1, dtype=np.float64)
        self.data = np.empty(cap, dtype=object)
        self.wr   = 0
        self.n    = 0

    def _prop(self, idx, delta):
        while idx>0:
            p=(idx-1)>>1; self.tree[p]+=delta; idx=p

    def _get(self, idx, s):
        while True:
            l=2*idx+1
            if l>=len(self.tree): return idx
            if s<=self.tree[l]: idx=l
            else: s-=self.tree[l]; idx=l+1

    @property
    def total(self): return float(self.tree[0])

    def add(self, p, data):
        i=self.wr+self.cap-1; self.data[self.wr]=data
        self.update(i,p)
        self.wr=(self.wr+1)%self.cap; self.n=min(self.n+1,self.cap)

    def update(self, idx, p):
        d=p-self.tree[idx]; self.tree[idx]=p; self._prop(idx,d)

    def get(self, s):
        s=max(0.0,min(float(s),self.total-1e-8))
        idx=self._get(0,s)
        return idx,self.tree[idx],self.data[idx-self.cap+1]


Transition = namedtuple('Transition',
    ('state_grid','state_vars','action',
     'next_state_grid','next_state_vars','reward','done'))


class PrioritizedReplayMemory:
    """
    PER con IS-weights.
    v5.4: TD error clamped a 10.0.
    v5.4-PER-PERSIST: save_async / load — il buffer sopravvive ai riavvii.
    """
    def __init__(self, cap, alpha=0.4, beta_start=0.4, beta_frames=200_000, eps=1e-5):
        self.alpha       = alpha
        self.eps         = eps
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.frame       = 1
        self.tree        = SumTree(cap)
        self._max_p      = 1.0

    def _beta(self):
        return min(1.0, self.beta_start+(1-self.beta_start)*self.frame/self.beta_frames)

    def push(self, *args):
        self.tree.add(self._max_p**self.alpha, Transition(*args))
        self.frame += 1

    def sample(self, bs):
        if self.tree.total<=0:
            raise RuntimeError("SumTree vuota.")
        idxs,prios,trans = [],[],[]
        seg = self.tree.total/bs
        for i in range(bs):
            s=random.uniform(seg*i, seg*(i+1))
            idx,p,t=self.tree.get(s)
            if t is None: idx,p,t=self.tree.get(random.uniform(0,self.tree.total))
            idxs.append(idx); prios.append(p); trans.append(t)
        probs   = np.array(prios,dtype=np.float64)/(self.tree.total+1e-8)
        weights = (self.tree.n*probs)**(-self._beta())
        weights /= weights.max()
        return trans, idxs, torch.tensor(weights,dtype=torch.float32,device=device)

    def update_priorities(self, idxs, errs):
        for i,e in zip(idxs,errs):
            e_clamped = min(abs(float(e)), 10.0)
            p = (e_clamped+self.eps)**self.alpha
            self._max_p = max(self._max_p, p)
            self.tree.update(i, p)

    def __len__(self): return self.tree.n

    # ── PERSISTENZA ───────────────────────────────────────────────────────────

    def _snapshot_to_numpy(self):
        """
        Converte il buffer valido in dict di array numpy.
        Eseguire SEMPRE nel main thread (i tensori possono essere su GPU).
        """
        n   = self.tree.n
        cap = self.tree.cap
        if n == 0:
            return None

        # Indici circolari in ordine cronologico (vecchio → recente)
        valid_idx = [(self.tree.wr - n + i) % cap for i in range(n)]

        grids, vars_, actions           = [], [], []
        next_grids, next_vars           = [], []
        rewards, dones, prios           = [], [], []

        for idx in valid_idx:
            t = self.tree.data[idx]
            if t is None:
                continue
            grids.append(t.state_grid.cpu().numpy())
            vars_.append(t.state_vars.cpu().numpy())
            actions.append(t.action.cpu().numpy())
            next_grids.append(t.next_state_grid.cpu().numpy())
            next_vars.append(t.next_state_vars.cpu().numpy())
            rewards.append(t.reward.cpu().numpy())
            dones.append(t.done.cpu().numpy())
            prios.append(float(self.tree.tree[idx + cap - 1]))

        if not grids:
            return None

        return dict(
            grids      = np.array(grids,      dtype=np.float32),
            vars       = np.array(vars_,      dtype=np.float32),
            actions    = np.array(actions,    dtype=np.int64),
            next_grids = np.array(next_grids, dtype=np.float32),
            next_vars  = np.array(next_vars,  dtype=np.float32),
            rewards    = np.array(rewards,    dtype=np.float32),
            dones      = np.array(dones,      dtype=bool),
            priorities = np.array(prios,      dtype=np.float64),
            # [wr, n, frame, _max_p]
            meta       = np.array([self.tree.wr, n, self.frame, self._max_p]),
        )

    def save_async(self, path):
        """
        Salva il buffer su disco senza bloccare il training.
        Passo 1 (main thread): GPU→CPU snapshot (~1-2s per 50k transizioni).
        Passo 2 (thread daemon): scrittura npz compresso su disco (~150-300MB).
        """
        arrays = self._snapshot_to_numpy()
        if arrays is None:
            print("[PER] Buffer vuoto — skip salvataggio.")
            return
        n = len(arrays['grids'])

        def _write():
            try:
                np.savez_compressed(path, **arrays)
                mb = os.path.getsize(path) / 1e6
                print(f"[PER] Buffer salvato: {n} transizioni → {path} ({mb:.1f} MB)")
            except Exception as e:
                print(f"[PER] Errore salvataggio buffer: {e}")

        threading.Thread(target=_write, daemon=True).start()

    def load(self, path):
        """
        Ricarica il buffer da file npz compresso.
        Ripristina: transizioni, priorità originali, frame (beta-annealing), _max_p.
        Se il file non esiste o è corrotto, parte da buffer vuoto senza crash.
        """
        if not os.path.exists(path):
            print(f"[PER] Nessun buffer in {path} — parto vuoto.")
            return
        try:
            print(f"[PER] Caricamento buffer da {path}...")
            d = np.load(path, allow_pickle=False)

            n = len(d['grids'])
            for i in range(n):
                t = Transition(
                    torch.from_numpy(d['grids'][i]).float().to(device),
                    torch.from_numpy(d['vars'][i]).float().to(device),
                    torch.from_numpy(d['actions'][i]).long().to(device),
                    torch.from_numpy(d['next_grids'][i]).float().to(device),
                    torch.from_numpy(d['next_vars'][i]).float().to(device),
                    torch.from_numpy(d['rewards'][i]).float().to(device),
                    torch.from_numpy(d['dones'][i]).to(device),
                )
                # Usa la priorità originale — non _max_p default
                self.tree.add(float(d['priorities'][i]), t)

            # Ripristina metadati: beta-annealing e max_p continuano da dove erano
            self.frame  = int(d['meta'][2])
            self._max_p = float(d['meta'][3])

            print(f"[PER] ✓ {n} transizioni caricate | "
                  f"frame={self.frame} | max_p={self._max_p:.4f} | "
                  f"beta={self._beta():.4f}")
        except Exception as e:
            print(f"[PER] Errore caricamento: {e} — parto vuoto.")


ReplayMemory = PrioritizedReplayMemory


# ─────────────────────────────────────────────────────────────────────────────
# 5. N-STEP BUFFER
# ─────────────────────────────────────────────────────────────────────────────
class NStepBuffer:
    def __init__(self, n_step=5, gamma=0.99):
        self.n_step = n_step; self.gamma = gamma
        self.buf    = deque(maxlen=n_step)

    def push(self, sg,sv,a,sg_next,sv_next,r,done):
        self.buf.append((sg,sv,a,sg_next,sv_next,r,done))

    def is_ready(self): return len(self.buf)==self.n_step

    def _calc_return(self, sub):
        R=0.0; terminal_idx=len(sub)-1
        for k,t in enumerate(sub):
            r    = t[5].item() if isinstance(t[5],torch.Tensor) else float(t[5])
            done = bool(t[6].item() if isinstance(t[6],torch.Tensor) else t[6])
            R   += self.gamma**k * r
            if done: terminal_idx=k; break
        return R, terminal_idx

    def get(self):
        if not self.is_ready(): return None
        buf_list=list(self.buf); R,ti=self._calc_return(buf_list)
        return (buf_list[0][0],buf_list[0][1],buf_list[0][2],
                buf_list[ti][3],buf_list[ti][4],
                torch.tensor([R],device=device), buf_list[ti][6])

    def drain(self):
        buf_list=list(self.buf); n=len(buf_list)
        start_offset = 1 if n==self.n_step else 0
        results=[]
        for start in range(start_offset,n):
            sub=buf_list[start:]; R,ti=self._calc_return(sub)
            results.append((sub[0][0],sub[0][1],sub[0][2],
                            sub[ti][3],sub[ti][4],
                            torch.tensor([R],device=device), sub[ti][6]))
        self.buf.clear(); return results

    def flush(self): self.buf.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 6. REWARD SHAPER
# ─────────────────────────────────────────────────────────────────────────────
class RewardShaper:
    def __init__(self): self.prev_phi=0.0
    def _phi(self,pY,oxy,rows):
        return (pY/max(rows,1))*5.0 + (oxy/100.0)*4.0
    def reset(self,pY,oxy,rows): self.prev_phi=self._phi(pY,oxy,rows)
    def step(self,pY,oxy,rows,gamma=0.99):
        phi=self._phi(pY,oxy,rows); b=gamma*phi-self.prev_phi
        self.prev_phi=phi; return b


# ─────────────────────────────────────────────────────────────────────────────
# 6b. DRILL TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class DrillTracker:
    def __init__(self, osc_window=8, col_window=12):
        self.recent_pos     = deque(maxlen=osc_window)
        self.recent_history = deque(maxlen=col_window)
    def reset(self): self.recent_pos.clear(); self.recent_history.clear()
    def update(self,x,y): self.recent_pos.append((x,y)); self.recent_history.append((x,y))
    def oscillation_penalty(self,x,y):
        hits=sum(1 for p in self.recent_pos if p==(x,y))
        if len(self.recent_history)>1 and y>self.recent_history[0][1]: return 0.0
        return hits*1.5
    def column_lock_penalty(self):
        if len(self.recent_history)<12: return 0.0
        history=list(self.recent_history)
        cols=[p[0] for p in history]; depths=[p[1] for p in history]
        if len(set(cols))==1:
            return 3.0 if depths[-1]<=depths[0]+2 else 0.0
        rc=cols[-6:]; rd=depths[-6:]
        if len(set(rc))==1 and rd[-1]<=rd[0]+1: return 1.0
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. SELECT ACTION
# ─────────────────────────────────────────────────────────────────────────────
def select_action(policy_net, state_grid, state_vars, epsilon, n_actions):
    policy_net.sample_noise()
    policy_net.eval()
    if random.random() < epsilon:
        return (torch.tensor([[random.randrange(n_actions)]],device=device,dtype=torch.long), None)
    with torch.no_grad():
        q = policy_net(state_grid.unsqueeze(0), state_vars.unsqueeze(0))
    return q.max(1)[1].view(1,1), q.squeeze().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 8. REWARD FUNCTION (v8.0)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_reward(prev_y,prev_x,new_y,new_x,prev_oxy,new_oxy,
                     prev_score,new_score,prev_lives,new_lives,
                     action_idx,is_hard_block,is_delayed_block,
                     drill_tracker,total_rows,shaping_bonus=0.0,
                     is_level_complete=False,blocks_around=None):
    if is_level_complete:
        return torch.tensor([300.0+(new_lives*50.0)], device=device)
    if new_lives < prev_lives:
        drill_tracker.reset()
        depth_ratio  = min(new_y/max(total_rows,1), 1.0)
        return torch.tensor([-100.0+(depth_ratio*60.0)], device=device)

    reward=0.0; oxy=new_oxy; d_score=new_score-prev_score
    dy=new_y-prev_y; dx=abs(new_x-prev_x); d_oxy=new_oxy-prev_oxy
    earned_dodge=False; earned_avoid=False

    if action_idx==0: reward-=0.3
    if action_idx!=0 and new_x==prev_x and new_y==prev_y:
        if not is_hard_block and not is_delayed_block: reward-=5.0

    if blocks_around and action_idx in [1,2,3,4]:
        bu=blocks_around.get("up")
        if bu and getattr(bu,'isFalling',lambda:False)():
            reward+=15.0; earned_dodge=True

    if is_hard_block:
        reward-=5.0
        if dy==0 and new_x==prev_x: reward-=8.0
    elif is_delayed_block:
        reward-=2.0
        if dy==0 and new_x==prev_x: reward-=4.0

    if blocks_around and dx>0:
        bd=blocks_around.get("down")
        if bd and bd.hpAccess()>0:
            bt=bd.typeAccess()
            if bt=="unbreakable":   reward+=6.0; earned_avoid=True
            elif bt=="delayed":     reward+=4.0; earned_avoid=True

    if oxy<20 and action_idx!=0: reward+=0.5

    if dy>0:
        depth_progress=min(new_y/max(total_rows,1),1.0)
        reward += dy*5.0 + depth_progress*2.0
        if action_idx==5: reward+=1.0
    elif dy<0:
        reward += dy*1.0

    if d_oxy>0:
        if oxy<30: reward+=50.0
        elif oxy<60: reward+=20.0
        else: reward+=5.0

    if action_idx in [1,2] and not earned_dodge and not earned_avoid:
        if d_oxy<=0 and dy==0: reward-=0.5
    if action_idx in [3,4]:
        if d_score>0 and not is_hard_block: reward+=1.5
        elif not is_hard_block: reward-=1.5

    if d_score>0: reward+=min(d_score/100.0, 2.0)

    drill_tracker.update(new_x, new_y)
    reward -= drill_tracker.oscillation_penalty(new_x, new_y)
    reward -= drill_tracker.column_lock_penalty()*2.0

    oxy_deficit = max(0.0, 100.0-oxy)/100.0
    reward -= (0.1 + oxy_deficit*0.4)
    reward += shaping_bonus
    return torch.tensor([reward], device=device)


# ─────────────────────────────────────────────────────────────────────────────
# 9. OPTIMIZE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def optimize_model(policy_net, target_net, memory, optimizer, batch_size=64, gamma=0.99):
    if len(memory) < batch_size: return None

    trans, tree_idx, is_w = memory.sample(batch_size)
    batch = Transition(*zip(*trans))

    sg = torch.stack(batch.state_grid)
    sv = torch.stack(batch.state_vars)
    ac = torch.cat(batch.action)
    rw = torch.cat(batch.reward)

    done_batch        = torch.cat(batch.done).bool()
    non_terminal_mask = ~done_batch

    policy_net.sample_noise()
    policy_net.train()
    sav = policy_net(sg,sv).gather(1,ac)

    nsv = torch.zeros(batch_size, device=device)
    if non_terminal_mask.any():
        indices = non_terminal_mask.nonzero(as_tuple=True)[0]
        nfg_t   = torch.stack([batch.next_state_grid[i] for i in indices.tolist()])
        nfv_t   = torch.stack([batch.next_state_vars[i] for i in indices.tolist()])
        with torch.no_grad():
            policy_net.eval()
            best_a = policy_net(nfg_t,nfv_t).argmax(1,keepdim=True)
            nsv[non_terminal_mask] = target_net(nfg_t,nfv_t).gather(1,best_a).squeeze(1)

    policy_net.train()
    exp_q = (rw + gamma*nsv).unsqueeze(1).detach()

    with torch.no_grad():
        td_err = (exp_q-sav).abs().squeeze(1).cpu().numpy()
    memory.update_priorities(tree_idx, td_err)

    loss = (F.smooth_l1_loss(sav,exp_q,reduction='none').squeeze(1)*is_w).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return float(loss.item()), td_err