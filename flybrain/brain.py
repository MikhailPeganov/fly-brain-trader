"""A Drosophila mushroom-body model that trades, learning from 'pain' and 'happiness'.

Anatomy follows published fly connectome statistics (FlyWire / hemibrain):
  * Projection neurons (PNs): sensory input. In the fly each glomerulus is tuned to
    one odour; here each quant factor is encoded by a small population of PNs with
    Gaussian tuning curves (population code).
  * Kenyon cells (KCs): ~2,000 per hemisphere. Each KC samples ~7 PNs ("claws") and the
    PN->KC wiring is statistically close to random (Caron et al. 2013, Nature).
  * APL neuron: one giant inhibitory neuron giving global feedback inhibition that keeps
    only ~5% of KCs active (sparse coding). Implemented as k-winners-take-all.
  * Mushroom-body output neurons (MBONs): read out KC activity. Here one MBON drives
    "go long" (approach) and one drives "go short" (avoid).
  * Dopaminergic neurons (DANs): PAM cluster signals reward ("happiness"), PPL1 cluster
    signals punishment ("pain"). Dopamine gates plasticity at the KC->MBON synapses of
    whichever KCs were active when the outcome arrived (three-factor learning rule).

Extras grounded in neuroscience / behavioural finance:
  * Loss aversion: pain from a loss is weighted lambda (~2x, Kahneman & Tversky) more
    than happiness from an equal gain.
  * Fictive (counterfactual) learning: a weaker signal about the trade not taken,
    as observed in human striatum during a market game (Lohrenz et al. 2007, PNAS).
  * Tonic mood: a slow 'pain level' that makes the fly more cautious after a run of losses.
"""
import numpy as np


class MushroomBodyTrader:
    def __init__(self, n_factors, n_kc=2000, claws=7, sparsity=0.05, n_tuning=5,
                 lr=0.01, threshold=1.0, gain=1.0, loss_aversion=2.0, fictive=0.3,
                 homeostasis=2e-4, mood_kappa=1.0, wmax=3.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.F, self.T = n_factors, n_tuning
        self.n_pn = n_factors * n_tuning
        self.n_kc, self.k = n_kc, max(1, int(round(sparsity * n_kc)))
        self.centers = np.linspace(-2.4, 2.4, n_tuning)
        self.sigma = self.centers[1] - self.centers[0]
        # PN -> KC: each KC receives `claws` distinct random PNs (binary synapses)
        J = np.zeros((self.n_pn, n_kc), dtype=np.float32)
        for j in range(n_kc):
            J[self.rng.choice(self.n_pn, size=claws, replace=False), j] = 1.0
        # slight synaptic weight heterogeneity, as in real claws
        self.J = J * self.rng.lognormal(0, 0.2, size=J.shape).astype(np.float32)
        # KC -> MBON synapses (non-negative, like real excitatory synapses)
        self.w0 = 1.0
        self.w_long = np.full(n_kc, self.w0)
        self.w_short = np.full(n_kc, self.w0)
        self.lr, self.threshold, self.gain = lr, threshold, gain
        self.lam, self.fictive, self.homeo = loss_aversion, fictive, homeostasis
        self.mood_kappa, self.wmax = mood_kappa, wmax
        self.d_var = 1e-4         # running variance of MBON imbalance (gain control)
        self.base_long = 0.0      # expected dopamine for each action (prediction)
        self.base_short = 0.0
        self.pain_fast = 0.0      # tonic pain level (short memory)
        self.pain_slow = 1e-3     # long-run typical pain (baseline)

    # ---- sensory pathway -------------------------------------------------
    def kenyon(self, x):
        """x: (N, F) standardized factors -> sparse binary KC activity (N, n_kc)."""
        pn = np.exp(-0.5 * ((x[:, :, None] - self.centers) / self.sigma) ** 2).reshape(len(x), -1)
        drive = pn.astype(np.float32) @ self.J
        idx = np.argpartition(-drive, self.k, axis=1)[:, :self.k]   # APL: keep top-k
        kc = np.zeros_like(drive)
        np.put_along_axis(kc, idx, 1.0, axis=1)
        return kc

    # ---- decision ----------------------------------------------------------
    def decide(self, kc, explore=0.0):
        L = kc @ self.w_long / self.k
        S = kc @ self.w_short / self.k
        d = (L - S) / (L + S + 1e-9)
        # divisive gain control: express the MBON imbalance relative to its recent spread
        self.d_var += 0.01 * (float(np.mean(d ** 2)) - self.d_var)
        zd = d / np.sqrt(self.d_var + 1e-12)
        if explore > 0:
            zd = zd + self.rng.normal(0, explore, size=zd.shape)
        mood = np.clip(self.pain_fast / self.pain_slow - 1.0, 0.0, 2.0)
        th = self.threshold * (1.0 + self.mood_kappa * mood)       # cautious when hurting
        pos = np.sign(zd) * np.clip((np.abs(zd) - th) * self.gain, 0.0, 1.0)
        return pos, zd

    # ---- feelings + plasticity --------------------------------------------
    def feel(self, r):
        happiness = np.maximum(r, 0.0)
        pain = self.lam * np.maximum(-r, 0.0)
        return happiness, pain

    def learn(self, kc, pos, outcome_z, cost_z):
        """outcome_z: vol-normalised return of the stock over the holding period.
        cost_z: trading cost actually paid (vol units). Returns (happiness, pain) per stock."""
        r = pos * outcome_z - cost_z
        happy, pain = self.feel(r)
        valence = happy - pain                                   # PAM minus PPL1 dopamine
        # counterfactual valence of a full long / full short position
        hl, pl = self.feel(outcome_z)
        hs, ps = self.feel(-outcome_z)
        v_long_cf, v_short_cf = hl - pl, hs - ps
        a_long = np.clip(pos, 0, 1)
        a_short = np.clip(-pos, 0, 1)
        # experienced dopamine for the side actually traded (already scaled by size),
        # weaker fictive dopamine for whatever part of a full position was not taken
        g_long = np.where(pos > 0, valence, 0.0) + self.fictive * v_long_cf * (1 - a_long)
        g_short = np.where(pos < 0, valence, 0.0) + self.fictive * v_short_cf * (1 - a_short)
        # dopamine neurons signal *prediction errors*: outcome relative to what this
        # action usually feels like (otherwise loss aversion just erodes every synapse)
        self.base_long += 0.01 * (g_long.mean() - self.base_long)
        self.base_short += 0.01 * (g_short.mean() - self.base_short)
        n = max(1, len(kc))
        self.w_long += self.lr * ((g_long - self.base_long) @ kc) / n
        self.w_short += self.lr * ((g_short - self.base_short) @ kc) / n
        # homeostatic drift back toward baseline + hard synaptic bounds
        self.w_long += self.homeo * (self.w0 - self.w_long)
        self.w_short += self.homeo * (self.w0 - self.w_short)
        np.clip(self.w_long, 0.0, self.wmax, out=self.w_long)
        np.clip(self.w_short, 0.0, self.wmax, out=self.w_short)
        # tonic mood update
        mp = float(np.mean(pain)) if len(pain) else 0.0
        self.pain_fast += 0.10 * (mp - self.pain_fast)
        self.pain_slow += 0.005 * (mp - self.pain_slow)
        self.pain_slow = max(self.pain_slow, 1e-3)
        return happy, pain
