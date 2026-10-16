# Test 1: Typicality-based Bayesian inference on Gaussians

Implementation plan for Claude Code, repository `kaneis1/Robust-likelihood`.
Suggested location in the repo: `docs/gaussian_test_plan.md`.

---

## 0. Instructions for Claude Code

**Goal.** Implement and run a no-training test of the method proposed by Tan Zhi-Xuan on 12 October 2026: replace the observation $x$ by its typicality under each hypothesis, optionally bin it, and do Bayesian inference on that. Use Gaussian hypotheses, where every quantity has a closed form, so any failure belongs to the method and not to estimation.

**Questions this test must answer** (from the 5 and 12 October meetings):

1. Does posterior inference on typicality, or on binned typicality, still converge to the true hypothesis as observations accumulate?
2. How much information is lost relative to conditioning on $x$, from a single observation and over many?
3. How should bin boundaries be chosen?
4. Is binned typicality more robust than the exact likelihood, and against which kinds of misspecification?

**Ground rules.**

- Work on a new branch `gaussian-test`. Do not modify existing modules except `pyproject.toml` (new optional extra) and `README.md` (new section). In particular, leave `src/robust_likelihood/typicality.py` and its tests unchanged.
- New code goes in a new subpackage `src/robust_likelihood/gaussian/`, run as `python -m robust_likelihood.gaussian ...`, following the precedent of `formal_rerun.py`. Do not register it in `cli.py`.
- Results go under `results/gaussian/`, which is already gitignored. Never commit results.
- Follow the repo's style: type hints, frozen dataclasses for configuration, docstrings that state exactly which quantity a function computes, and no placeholder values.
- **Section 7 lists pre-registered predictions. If a result contradicts a prediction, do not change the method, the bins, or the grids to make it pass. Record the failure in `summary.md` and stop to report it.** A closed-form unit test failing is different: that is a bug in the code; fix the code, never the test.
- Use common random numbers: within a trial, every scorer sees exactly the same observations.
- Implement in the phases of Section 10, running `pytest` after each phase.

---

## 1. Model and definitions

Hypothesis $z$ fixes a Gaussian over $x \in \mathbb{R}^d$:

$$x \mid z \sim \mathcal{N}(\mu_z, \sigma_z^2 I_d)$$

Define the scaled squared distance

$$q_z(x) = \frac{\lVert x - \mu_z \rVert^2}{\sigma_z^2}$$

Then, in nats:

| Quantity | Formula |
|---|---|
| Log-likelihood | $\log p(x \mid z) = -\tfrac{d}{2}\log(2\pi\sigma_z^2) - \tfrac12 q_z(x)$ |
| Surprisal | $S(x,z) = -\log p(x \mid z)$ |
| Entropy | $H(z) = \mathbb{E}_{x\sim z}[S] = \tfrac{d}{2}\log(2\pi e\,\sigma_z^2)$ |
| Variance of surprisal | $V(z) = \operatorname{Var}_{x\sim z}[S] = \tfrac{d}{2}$ |
| Standardized typicality | $T(x,z) = \dfrac{S - H}{\sqrt{V}} = \dfrac{q_z(x) - d}{\sqrt{2d}}$ |

Properties to rely on and test:

- Under the true hypothesis, $q_z(x) \sim \chi^2_d$ exactly, so the distribution of $T$ is **the same for every hypothesis** with the same $d$ (it is pivotal). Mean 0, variance 1, skewness $\sqrt{8/d}$.
- $T \ge -\sqrt{d/2}$, because $q \ge 0$. At small $d$, bins below this bound are empty.
- $T > 0$: less probable than typical. $T < 0$: more probable than typical, i.e. closer to the mode than samples usually are.

**Sign convention.** Positive $T$ means more surprising, matching the meeting. The existing `typicality.signed_deviation` uses $\log p - \mathbb{E}[\log p] = H - S$, the opposite sign. For $d = 1$: $T = -\sqrt{2}\cdot$`signed_deviation`. Add a test that checks this relation (Section 9).

Compute $q$ for all hypotheses at once with $\lVert x-\mu\rVert^2 = \lVert x\rVert^2 - 2\,x\cdot\mu + \lVert\mu\rVert^2$ as a matrix product, in float64.

---

## 2. Scorers

Each scorer maps one observation $x$ to a per-hypothesis log-score $\ell_z(x)$. The posterior after $n$ observations is

$$\log \pi_n(z) = \log \pi_0(z) + \sum_{i=1}^{n} \ell_z(x_i) - \operatorname{logsumexp}_{z'}(\cdot)$$

with a uniform prior $\pi_0$ unless stated. Per-observation constants that do not depend on $z$ cancel, so they may be dropped.

| Name | $\ell_z(x)$ | Role |
|---|---|---|
| `exact` | $\log p(x\mid z)$ | Reference: conditioning on $x$ itself |
| `tempered_0.5`, `tempered_0.1` | $\zeta \log p(x\mid z)$ | Coarsening baseline (power likelihood, Miller & Dunson) |
| `clipped` | $\max\big(\log p(x\mid z) - \max_{z'}\log p(x\mid z'),\ -c\big)$ | Bounded-influence baseline. $c$ is matched to the cap of `bins_abs3` at the same $d$: $c = \log(\max_k m_k / \min_k m_k)$ |
| `typ_normal` | $\log\phi(T(x,z)) \equiv -\tfrac12 T^2$ | Continuous typicality rule from the lab's qualitative-likelihood notes |
| `typ_density` | $\log f_{\chi^2_d}(q_z(x))$ | Exact density of the statistic, without the Jacobian. Secondary variant |
| `bins_<scheme>` | $\log m_{k(z,x)}$, the reference mass of the bin that $T(x,z)$ falls into | Xuan's proposal |

`clipped` exists because a reviewer will ask whether the robustness of bins comes from typicality or simply from capping each observation's evidence. It must be reported wherever bins are compared for robustness.

At $d = 2$, $f_{\chi^2_2}(q) \propto e^{-q/2}$, so `typ_density` and `exact` give identical posteriors when all $\sigma_z$ are equal. Use this as a unit test.

---

## 3. Bin schemes

### 3.1 Exact bin masses

A scheme is a partition of the $T$ axis. Because $T$ is pivotal, the mass of bin $k$ is the same under every hypothesis with the same $d$. Compute it exactly from the $\chi^2_d$ distribution, not from the normal approximation:

- A **signed** bin $T \in [e_k, e_{k+1})$ is the interval $q \in [d + e_k\sqrt{2d},\ d + e_{k+1}\sqrt{2d})$ clipped to $[0, \infty)$, with mass $F_d(\text{hi}) - F_d(\text{lo})$, where $F_d$ is the $\chi^2_d$ CDF.
- An **absolute** bin $\lvert T\rvert \in [a_k, a_{k+1})$ is the union of the two corresponding intervals: $q \in (d - a_{k+1}\sqrt{2d},\ d - a_k\sqrt{2d}] \cup [d + a_k\sqrt{2d},\ d + a_{k+1}\sqrt{2d})$, clipped at 0.
- For numerical stability, compute upper-tail masses with `scipy.stats.chi2.sf` differences and lower-tail masses with `cdf` differences. Store log-masses.
- Bins with mass below $10^{-300}$ at a given $d$ are empty, which happens on the lower side at small $d$. Drop them from the scheme at that $d$, and assert that no observation ever falls into a dropped bin.

The bin index of an observation under hypothesis $z$ is computed from $T(x, z)$. The likelihood used is the log-mass of that bin.

### 3.2 Schemes to implement

| Name | Type | Edges | Meaning |
|---|---|---|---|
| `bins_abs2` | absolute | $\lvert T\rvert$: 1.96 | likely / unlikely (Xuan's simplest case) |
| `bins_abs3` | absolute | $\lvert T\rvert$: 1, 2 | typical / somewhat atypical / very atypical |
| `bins_signed5` | signed | $T$: −2, −1, 1, 2 | very probable, probable, typical, improbable, very improbable |
| `bins_equalmass{K}` | absolute | the $K$ equal-mass quantiles of $\lvert T\rvert$ | **Negative control**, $K \in \{2,3,4\}$ |
| `bins_opt{K}` | absolute | chosen in E5 | Optimized edges |

**Why equal-mass is a control.** Every bin then has mass $1/K$, so every hypothesis receives log-score $-\log K$ for every observation, and the posterior never moves from the prior. The information in binned typicality comes entirely from bins having *different* masses. This must hold exactly in the implementation (Section 9).

---

## 4. Hypothesis families and data

### 4.1 Mean-shift family (E1, E4, E5)

- $M = 20$ hypotheses, all with $\sigma^2 = 1$.
- Hypothesis 0 is the truth with $\mu_0 = 0$. Hypotheses $1, \dots, 19$ have $\mu_j = \delta\,u_j$, with $u_j$ independent uniform random unit vectors redrawn every trial.
- Separation is set relative to the predicted threshold $\delta^*(d) = (2d)^{1/4}$ (Appendix A.2): $\delta = r\,\delta^*(d)$.

### 4.2 Scale family (E2)

- 5 hypotheses with common mean 0 and $\sigma_z^2 \in \{0.25, 0.5, 1, 2, 4\}$.
- Each trial's truth cycles through all 5, so every hypothesis is the truth equally often.
- This is the "sharp versus broad hypothesis" case. Under the exact likelihood, all hypotheses have different entropies $H(z)$, and the typicality rule standardizes that difference away.

### 4.3 Localization family (E3)

The bridge to the 64×64 grid Xuan described.

- Hypotheses are the 441 points of a 21×21 grid with spacing $s$, embedded in coordinates 1–2 of $\mathbb{R}^d$. All other coordinates have mean 0 under every hypothesis. All $\sigma^2 = 1$.
- The **true cell is drawn uniformly from interior points at least 2 cells from the border, never fixed at the center.** A truth at the center would make a diffuse posterior look accurate by symmetry.
- Common random numbers across $d$: generate the 2 in-plane noise coordinates from a seed that does not depend on $d$, and the $d-2$ nuisance coordinates from a separate seed. Then the exact posterior is identical across $d$ for the same trial (Prediction P5).

### 4.4 Contamination (E4)

Observations come from the truth, then are perturbed:

| Code | Perturbation | Purpose |
|---|---|---|
| C0 | none | clean reference |
| C1 | $x = \mu^* + \sqrt{1.1}\,\varepsilon$ | misspecification shared by all hypotheses (variance) |
| C2 | each coordinate of $\varepsilon \sim t_5/\sqrt{5/3}$, unit variance | shared, heavy tails |
| C3 | with probability $\rho$, add $R\,u_1$, where $u_1$ points from the truth to hypothesis 1 | gross outliers toward one wrong hypothesis; $\rho \in \{0, 0.05, 0.1, 0.2, 0.3\}$, $R \in \{20, 50\}$ |
| C4 | with probability $\rho$, add $R\,v$, where $v$ is a fresh uniform random unit vector | gross outliers in random directions (control for C3) |
| C5 | in every observation, each coordinate independently with probability 0.01 gets $+\mathcal{N}(0, 10^2)$ | small corruption present in every observation |

Simulate all $d$ coordinates explicitly for C1–C5. The nuisance shortcut of 4.3 is only valid for clean data.

---

## 5. Inference and metrics

For each trial, compute the $n_{\max} \times M$ matrix of $\ell_z(x_i)$ per scorer, take cumulative sums along observations, and normalize in log space. Then:

| Metric | Definition |
|---|---|
| `p_true(n)` | posterior probability of the true hypothesis after $n$ observations; report the mean over trials and a 95% CI (1.96 × SE) |
| `acc(n)` | fraction of trials whose **unique** argmax is the truth; a tie counts as incorrect (repo convention) |
| `n90` | smallest $n$ with mean `p_true(n)` ≥ 0.9, or `null` if not reached by $n_{\max}$ |
| `bits(n)` | mean of $\log_2 \pi_n(z^*) - \log_2 \pi_0(z^*)$; `bits(1)` is the single-observation information gain |
| `kl_to_exact(n)` | mean of $\mathrm{KL}(\pi_n^{\text{exact}} \,\Vert\, \pi_n^{\text{scorer}})$, computed in log space; clean data only |
| `coverage80(n)` | fraction of trials whose smallest set of hypotheses with posterior mass ≥ 0.8 contains the truth |
| `loc_error(n)` | E3 only: Euclidean distance, in grid units, from the posterior mean location to the true cell |
| `post_entropy(n)` | E3 only: entropy of the posterior in bits |

Paired comparisons between scorers use per-trial differences, which common random numbers make valid.

---

## 6. Experiments

Two presets. `quick` is a smoke run for a laptop (target under 3 minutes). `full` is the reportable run (target under 60 minutes on one CPU core; no GPU needed). If `full` exceeds the target, reduce trials, never grid points.

### E0: Validity checks (no inference)

Purpose: verify the closed forms and the Monte Carlo procedure that the language version will rely on, since there the entropy and variance must be estimated by sampling.

- Grid: $d \in \{1, 2, 10, 100, 1000\}$; $\sigma^2 \in \{0.25, 1, 4\}$; Monte Carlo sample sizes $S \in \{10, 100, 1000, 10000\}$.
- For each cell: sample $S$ observations from $z$, estimate $\hat H$ and $\hat V$ from their surprisals, compare with the closed forms, and report the error in units of the standard error $\sqrt{V/S}$.
- Moments of $T$ under the truth with $10^5$ draws: mean, variance, skewness against $(0, 1, \sqrt{8/d})$.
- Exact bin masses against Monte Carlo frequencies for every scheme and $d$ (chi-square goodness-of-fit $p$-value).
- Report how a plug-in estimate $\hat H$ biases $T$: with $S$ samples, the standard error of $T$'s center is about $1/\sqrt{S}$.

### E1: Information loss and convergence (mean-shift)

- Grid: $d \in \{10, 100, 1000\}$; $r \in \{0.5, 0.75, 1, 1.25, 1.5, 2, 3\}$; $n_{\max} = 50$; trials 500 (`quick`: $d \in \{10, 100\}$, $r \in \{1, 2\}$, 50 trials, $n_{\max} = 20$).
- Scorers: all from Section 2 with bins `abs2`, `abs3`, `signed5`, `equalmass3`.
- Outputs: `p_true`, `acc`, `bits`, `kl_to_exact`, `coverage80` for every $n$; `n90` per cell; a **single-observation table** of `bits(1)` per scorer and cell (Xuan's question at 34:22 of the 12 October meeting).

### E2: Sharp versus broad hypotheses (scale family)

- Grid: $d \in \{1, 2, 5, 10, 100, 1000\}$; $n_{\max} = 1000$; trials 500, 100 per true hypothesis (`quick`: $d \in \{1, 10\}$, 50 trials, $n_{\max} = 200$). The long horizon is needed: at $d = 1$ and $d = 5$ the per-observation margin between neighboring hypotheses is small, and a pilot showed agreement with the predicted limit of only 70–86% at $n = 200$ but 92–98% at $n = 1000$.
- Every scorer depends on $x$ only through $R = \lVert x \rVert^2$, and under the truth $R \sim \sigma_*^2\,\chi^2_d$ exactly. E2 is clean-only, so sample $R$ directly from the chi-square distribution instead of simulating $d$ coordinates. Add a test that the scores computed from directly sampled $R$ match those from full simulation in distribution.
- Store metrics at $n \in \{1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000\}$ rather than every $n$.
- Scorers: `exact`, `typ_normal`, `typ_density`, `bins_abs3`, `bins_signed5`.
- Additional output, the **predicted limit**: for each scorer and true hypothesis, compute the population objective $\mathbb{E}_{x\sim z^*}[\ell_z(x)]$ for every $z$ (closed form for `exact` and `typ_normal`, Appendix A.3; Monte Carlo with $10^5$ draws for the others). The posterior must concentrate on its argmax. Report the predicted argmax next to the empirical posterior mode at $n_{\max}$.

### E3: Localization with nuisance dimensions

- Grid: spacing $s \in \{0.5, 1, 2\}$; $d \in \{2, 10, 100, 1000\}$; $n_{\max} = 20$; trials 300 (`quick`: $s = 1$, $d \in \{2, 100\}$, 30 trials).
- Scorers: `exact`, `typ_normal`, `typ_density`, `bins_abs3`, plus the oracle ablation `typ_content_oracle` and `bins_abs3_content_oracle`, which compute $T$ from coordinates 1–2 only, i.e. with $d_{\text{eff}} = 2$. These show what discarding the nuisance dimensions would recover, the Gaussian analogue of scoring content instead of surface form.
- Outputs: `p_true`, `loc_error`, `post_entropy` for every $n$; heatmaps for one fixed seeded trial: rows `exact`, `typ_normal`, `bins_abs3`; columns $n \in \{1, 2, 5, 10, 20\}$; one figure per $d \in \{2, 100, 1000\}$ at $s = 1$.

### E4: Robustness

- Setting: mean-shift family with $d \in \{100, 1000\}$, $r = 1.5$, $M = 20$, $n_{\max} = 40$, trials 500 (`quick`: $d = 1000$, C0 and C3 with $\rho \in \{0, 0.2\}$ and $R = 50$, 50 trials).
- Contaminations C0–C5 from 4.4, with the full $\rho$ and $R$ grids for C3 and C4.
- Scorers: `exact`, `tempered_0.5`, `tempered_0.1`, `clipped`, `typ_normal`, `bins_abs2`, `bins_abs3`, `bins_signed5`.
- Outputs: `acc(20)`, `acc(40)`, `p_true(n)` per contamination and setting; accuracy-versus-$\rho$ curves for C3 and C4.

### E5: Bin design

Answers the open question of how to choose the bins.

- **Calibration setting** (its own seed, never reused for evaluation): E1 with $d = 100$, $r = 1.5$; 200 trials; clean, plus C3 with $\rho = 0.2$, $R = 50$.
- **Candidates:** absolute schemes with $K \in \{2, 3, 4\}$, edges strictly increasing from the grid $\{0.25, 0.5, \dots, 4.0\}$. That's 16 + 120 + 560 = 696 schemes. Cache $T$ per trial and only re-bin, so the search is cheap.
- **Objective:** maximize mean `bits(10)` on clean calibration data, subject to `acc(40)` ≥ 0.9 on the contaminated calibration data. If no candidate satisfies the constraint, take the maximum of the objective and flag it.
- **Held-out evaluation:** E1 cells $d \in \{10, 1000\}$, $r \in \{1, 1.5, 2\}$, and E4 at $d = 1000$, C3, $\rho \in \{0.1, 0.3\}$. Compare `bins_opt{K}` with `bins_abs2`, `bins_abs3`, and `bins_equalmass{K}`.
- Outputs: chosen edges per $K$; held-out table; a scatter of clean information (`bits(10)`) against contaminated accuracy (`acc(40)`) for all candidates, with the named schemes highlighted.

**Out of scope for Test 1:** training a typicality regressor or classifier (Test 2), and any language-model reference generator (Test 3).

---

## 7. Pre-registered predictions

`summary.md` must report each as PASS, FAIL, or INCONCLUSIVE, with the numbers. Where possible, evaluate them automatically.

| ID | Prediction | Pass criterion |
|---|---|---|
| P1 | Closed forms and Monte Carlo agree. | $\lvert\hat H - H\rvert$ within 4 SE in every E0 cell; moments of $T$ within 4 SE of $(0, 1, \sqrt{8/d})$; no bin-mass goodness-of-fit $p < 0.001$ |
| P2 | The exact likelihood is dimension-free; typicality-based scorers have a threshold at $\delta^*(d) = (2d)^{1/4}$. | `exact` reaches `p_true(20)` ≥ 0.99 in every E1 cell with $\delta \ge 2$. For `typ_normal` and `bins_abs3`, `n90` at fixed $r$ agrees across $d$ within a factor of 1.5 (the curves collapse when plotted against $r$). Evaluate only the $r$ values where `n90` is reached at all three $d$; if fewer than 3 such values exist, report INCONCLUSIVE |
| P3 | Most of the information loss comes from reducing $x$ to $T$, not from binning. | In at least 80% of E1 cells with $r \le 2$: `bits(1)` gap between `typ_normal` and `bins_abs3` is smaller than the gap between `exact` and `typ_normal` |
| P4 | With hypotheses of different spread at small $d$, `typ_normal` is **inconsistent**: it converges to a hypothesis broader than the truth, by the factor $1 + 2/d$ (Appendix A.3). | For each $d$ and truth, the empirical posterior mode at $n = 1000$ equals the predicted argmax in at least 90% of trials. `exact` is consistent at every $d$ |
| P5 | In localization, the exact posterior does not depend on the nuisance dimensions, while typicality degrades as they grow. | `exact` posteriors identical across $d$ for the same trial (max absolute difference $< 10^{-9}$). For `typ_normal`, `p_true(20)` decreases monotonically in $d$ at every spacing. `typ_content_oracle` at any $d$ is identical to `typ_normal` at $d = 2$ for the same trial (it sees the same in-plane distances with $d_{\text{eff}} = 2$) |
| P6 | Bins bound each observation's evidence, so gross outliers cannot dominate; tempering cannot bound it. | C3 with $R = 50$: `bins_abs3` `acc(40)` ≥ 0.9 for all $\rho \le 0.3$; `exact`, `tempered_*`, and `typ_normal` `acc(40)` < 0.5 for $\rho \ge 0.2$ |
| P6b | The matched-cap `clipped` likelihood is as robust as bins on C3. | `clipped` `acc(40)` ≥ 0.9 for all $\rho \le 0.3$. **If this passes, `summary.md` must state that in this test the robustness advantage comes from bounded influence, not from typicality specifically** |
| P6c | Under misspecification shared by all hypotheses (C1, C2), `exact` loses nothing. Under corruption in every observation (C5), bins fail safe: the posterior stays near the prior instead of becoming wrong. | C1, C2: `exact` `acc(40)` ≥ `bins_abs3` `acc(40)`. C5: `bins_abs3` mean `p_true(20)` ≤ 0.2, and no single wrong hypothesis receives mean posterior mass > 0.5 |
| P7 | Equal-mass bins carry no information; optimized edges help. | `bins_equalmass{K}` posterior equals the prior exactly (max absolute difference $< 10^{-12}$) at every $n$ and $d$. `bins_opt3` held-out `bits(10)` ≥ `bins_abs3` in at least 4 of 6 held-out E1 cells |

---

## 8. Code layout and interfaces

```
src/robust_likelihood/gaussian/
    __init__.py
    __main__.py        # CLI: run / report
    model.py           # closed forms: quadratic forms, log-likelihood, H, V, T
    families.py        # mean-shift, scale, localization families
    sampling.py        # observation sampling, contamination C0-C5, seeding
    bins.py            # BinScheme, exact chi-square log-masses, bin indices, equal-mass constructor
    scorers.py         # per-observation log-score matrices for every scorer
    inference.py       # cumulative posteriors and all metrics
    experiments.py     # E0-E5 definitions, presets, registry
    report.py          # figures and summary.md
tests/
    test_gaussian_model.py
    test_gaussian_bins.py
    test_gaussian_scorers.py
    test_gaussian_inference.py
    test_gaussian_experiments.py
templates/
    gaussian_test_report.md   # hand-filled report skeleton (Section 11)
```

Key interfaces:

```python
# model.py
def quadratic_forms(x: np.ndarray, mus: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """q[i, z] = ||x_i - mu_z||^2 / sigma2_z. x: (n, d); mus: (M, d); sigma2: (M,). Returns (n, M)."""
def log_likelihood(q: np.ndarray, sigma2: np.ndarray, d: int) -> np.ndarray: ...
def entropy(sigma2: np.ndarray, d: int) -> np.ndarray: ...          # nats
def surprisal_variance(d: int) -> float: ...                         # d / 2
def typicality(q: np.ndarray, d: int) -> np.ndarray: ...             # (q - d) / sqrt(2d)

# bins.py
@dataclass(frozen=True)
class BinScheme:
    name: str
    kind: Literal["abs", "signed"]
    edges: tuple[float, ...]        # interior edges only
def log_masses(scheme: BinScheme, d: int) -> np.ndarray: ...         # exact chi-square, empty bins dropped
def bin_index(scheme: BinScheme, t: np.ndarray, d: int) -> np.ndarray: ...
def equal_mass_scheme(k: int, d: int) -> BinScheme: ...

# scorers.py
def score_matrix(name: str, q: np.ndarray, sigma2: np.ndarray, d: int) -> np.ndarray:
    """Per-observation log-scores, shape (n, M). Dispatches on scorer name from Section 2."""

# inference.py
def cumulative_log_posterior(scores: np.ndarray, log_prior: np.ndarray) -> np.ndarray: ...  # (n, M), normalized
def metrics(log_post: np.ndarray, true_index: int, ...) -> dict[str, np.ndarray]: ...
```

**Seeding.** One base seed in the run config (default `20261019`). Each (experiment, cell, trial) gets its own `numpy.random.Generator` from `SeedSequence([base_seed, crc32(experiment), crc32(cell_key), trial])`, so results do not depend on execution order. E3 in-plane and nuisance noise use separate keys (Section 4.3).

**CLI.**

```powershell
python -m pip install -e ".[dev,gaussian]"
python -m pytest
python -m robust_likelihood.gaussian run --preset quick
python -m robust_likelihood.gaussian run --preset full --experiments e0,e1,e2,e3,e4,e5
python -m robust_likelihood.gaussian report results\gaussian\<run-folder>
```

`run` writes the report automatically at the end; `report` regenerates figures and `summary.md` from stored metrics.

**Dependencies.** Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest>=8"]
gaussian = ["numpy>=1.26", "scipy>=1.11", "matplotlib>=3.8"]
```

The new tests start with `pytest.importorskip("numpy")` and `pytest.importorskip("scipy")`, so the existing suite still passes without the extra. Use the matplotlib `Agg` backend.

**Run folder** `results/gaussian/<UTC timestamp>-<preset>/`:

| File | Contents |
|---|---|
| `config.json` | the resolved preset and grids |
| `manifest.json` | git commit hash and dirty flag, Python and package versions, base seed, start and end time, wall-clock time per experiment |
| `metrics.jsonl` | one row per (experiment, cell, contamination, scorer, $n$) with every metric from Section 5, plus mean, SE, and trial count; every $n$ up to 50, and the logarithmic grid of E2 beyond that |
| `cells.jsonl` | one row per (experiment, cell, scorer) with `n90`, predicted limits (E2), and chosen edges (E5) |
| `figures/*.png` | Section 8.1 |
| `summary.md` | generated tables and the prediction checklist |

### 8.1 Figures

1. **E1 convergence:** `p_true` against $n$, small multiples over $d$ (rows) and $r$ (columns), one line per scorer.
2. **E1 collapse:** `n90` against $r$, one line per $d$, for `typ_normal` and `bins_abs3`. Overlapping lines confirm P2.
3. **E1 single-observation information:** `bits(1)` against $r$ per scorer, one panel per $d$.
4. **E2 consistency:** `p_true` against $n$ per $d$; and the empirical posterior mode against the truth, with the predicted $(1 + 2/d)$ line.
5. **E3 heatmaps:** the posterior over the grid for one seeded trial, for $d \in \{2, 100, 1000\}$.
6. **E3 nuisance dimensions:** `p_true(20)` against $d$ per scorer, including the content oracle.
7. **E4 robustness:** `acc(40)` against $\rho$ for C3 and C4, per $R$ and $d$; a table for C1, C2, C5.
8. **E5 trade-off:** clean `bits(10)` against contaminated `acc(40)` for all candidate schemes.

---

## 9. Unit tests

Fast tests only (under 30 seconds total). Each gets its own named test function.

**`test_gaussian_model.py`**
- `log_likelihood` matches `scipy.stats.multivariate_normal.logpdf` with isotropic covariance, for $d \in \{1, 3, 10\}$ and several $\sigma^2$.
- `quadratic_forms` via the matrix identity matches a direct loop.
- Monte Carlo mean and variance of the surprisal with $10^5$ draws match `entropy` and `surprisal_variance` within 5 SE.
- For $d = 1$, $T = -\sqrt{2}\cdot$`typicality.signed_deviation` for the existing 1-D `GaussianHypothesis`.
- $T \ge -\sqrt{d/2}$ always.

**`test_gaussian_bins.py`**
- Log-masses of every scheme sum to 1 in probability space within $10^{-12}$, for $d \in \{1, 2, 10, 1000\}$.
- Exact masses agree with Monte Carlo frequencies of $T$ under the truth ($10^5$ draws, $\chi^2$ goodness-of-fit $p > 0.001$).
- Masses are identical for different $\mu$ and $\sigma^2$ at the same $d$ (pivotality).
- Empty bins are dropped at $d = 1$ for `signed5`, and `bin_index` never returns a dropped bin.
- An `abs` scheme gives the same index for $T$ and the reflected value with equal $\lvert T\rvert$.

**`test_gaussian_scorers.py`**
- At $d = 2$ with equal $\sigma^2$, `typ_density` and `exact` give the same posterior within $10^{-10}$.
- `bins_equalmass{K}` posterior equals the prior within $10^{-12}$ after 50 observations.
- `clipped` per-observation evidence never exceeds the cap $c$ between any two hypotheses.
- Adding one observation at distance $10^4$ from every mean leaves the `bins_abs3` posterior unchanged (it lands in the outer bin under every hypothesis).
- `tempered_1.0` equals `exact`.

**`test_gaussian_inference.py`**
- Posteriors are normalized and finite at $d = 1000$, $n = 50$, $M = 441$ (no NaN or overflow).
- Ties count as incorrect in `acc`.
- `coverage80` on a hand-built posterior.
- E2 closed-form predicted argmax for `typ_normal` matches Appendix A.3: truth $\sigma^2 = 1$ at $d = 1$ predicts $\sigma^2 = 4$; at $d = 2$ predicts $\sigma^2 = 2$; at $d = 10$ predicts $\sigma^2 = 1$.

**`test_gaussian_experiments.py`**
- Determinism: the same seed gives identical metrics; a different trial order gives identical per-trial results.
- E3 common random numbers: the `exact` posterior for one trial is identical at $d = 2$ and $d = 100$.
- E2 shortcut: per-observation scores computed from directly sampled $R \sim \sigma_*^2\chi^2_d$ and from fully simulated $x$ agree in distribution (two-sample KS test, $p > 0.001$, $10^4$ draws, $d = 10$).
- A one-cell, five-trial run of each experiment writes well-formed rows.

---

## 10. Implementation phases

Commit at the end of each phase with a descriptive message. Run `pytest` before each commit.

1. **Skeleton.** Branch, `pyproject.toml` extra, empty subpackage, CLI stub that prints the plan of a preset (like `--dry-run` elsewhere in the repo).
2. **Core math.** `model.py`, `bins.py`, and their tests.
3. **Scorers and inference.** `scorers.py`, `inference.py`, and their tests.
4. **Families and sampling.** `families.py`, `sampling.py` with all contaminations; determinism tests.
5. **Experiments.** E0–E5 and presets; run `--preset quick` and inspect `summary.md`.
6. **Report.** Figures and `summary.md` with the automatic prediction checklist.
7. **Full run.** Run `--preset full`. Do not commit results. Write a short note in the PR description with the run folder name, the wall-clock time, and the prediction checklist.
8. **Docs.** Add a "Gaussian typicality test" section to `README.md` (commands, what each experiment answers, where results go) and add `templates/gaussian_test_report.md`.

Stop and report instead of proceeding if: a closed-form test cannot be made to pass; the `full` preset exceeds 2 hours; or any prediction in Section 7 fails. A failed prediction is a result, not a bug, and must be reported as such.

---

## 11. Report

`summary.md` is generated. `templates/gaussian_test_report.md` is a skeleton that Kaniel fills in by hand, in the style of the existing templates. Sections:

1. **Question.** The four questions of Section 0.
2. **Setup.** Families, scorers, bins, presets, run folder, commit.
3. **Convergence and information loss.** E1 figures 1–3, the `n90` table, the single-observation `bits(1)` table.
4. **Sharp versus broad hypotheses.** E2 figure 4 and the predicted-versus-observed limit table.
5. **Nuisance dimensions.** E3 figures 5–6.
6. **Robustness.** E4 figure 7 and table, with `clipped` reported next to every bin scheme.
7. **Bin choice.** E5 edges and held-out comparison.
8. **Predictions.** The checklist of Section 7.
9. **Questions for Xuan.** Including, at minimum:
   - Hypothesis-dependent bins are a different event for each hypothesis, so $P(z \mid \text{bin})$ is not a likelihood of one shared observation. E2 shows the practical consequence (inconsistency at small $d$). Which justification does she intend: generalized Bayes, coarsening, or something else?
   - If `clipped` matches bins on robustness (P6b), the case for typicality must rest on something other than robustness, for example normalization across hypotheses and learnability. Does she agree?
   - E3 shows typicality cannot tell content dimensions from nuisance dimensions. For language, that is the form-versus-content problem from her notes. Should the regressor's input be a content representation?

---

## Appendix A: Derivations

### A.1 Typicality for an isotropic Gaussian

$S = \tfrac{d}{2}\log(2\pi\sigma^2) + \tfrac12 q$ with $q \sim \chi^2_d$ under the truth. Then $H = \mathbb{E}[S] = \tfrac{d}{2}\log(2\pi\sigma^2) + \tfrac{d}{2}$ and $V = \operatorname{Var}(q)/4 = 2d/4 = d/2$. So $T = (\tfrac12 q - \tfrac{d}{2})/\sqrt{d/2} = (q - d)/\sqrt{2d}$.

### A.2 The separation threshold in the mean-shift family

For a wrong hypothesis at distance $\delta$, $q_{\text{wrong}} = \lVert \varepsilon - \delta u \rVert^2 = q_{\text{true}} - 2\delta\,\varepsilon\cdot u + \delta^2$. Its $T$ differs from the truth's by $(\delta^2 - 2\delta\,\varepsilon\cdot u)/\sqrt{2d}$: a mean shift of $\delta^2/\sqrt{2d}$, while $T$ itself has unit spread. One observation separates the two hypotheses only when $\delta^2/\sqrt{2d} \gtrsim 1$, i.e. $\delta \gtrsim (2d)^{1/4}$.

The exact log-likelihood ratio is $\delta^2/2 - \delta\,\varepsilon\cdot u$, independent of $d$. So exact inference needs $\delta \gtrsim 1$ in any dimension, while typicality needs $\delta \gtrsim (2d)^{1/4}$: about 2.1 at $d = 10$, 3.8 at $d = 100$, 6.7 at $d = 1000$.

### A.3 Inconsistency for hypotheses of different spread

With common mean and truth $\sigma_*^2$, let $R = \lVert x \rVert^2$, so $\mathbb{E}[R] = d\sigma_*^2$ and $\mathbb{E}[R^2] = \sigma_*^4(d^2 + 2d)$. Under hypothesis $s = \sigma_z^2$, $T_s = (R/s - d)/\sqrt{2d}$, and

$$\mathbb{E}[\log\phi(T_s)] = -\tfrac12\,\mathbb{E}[T_s^2] + \text{const} = -\frac{1}{4d}\left(\frac{\mathbb{E}[R^2]}{s^2} - \frac{2d\,\mathbb{E}[R]}{s} + d^2\right) + \text{const}$$

Setting the derivative in $1/s$ to zero gives the limit of the `typ_normal` posterior:

$$s^\dagger = \sigma_*^2\left(1 + \frac{2}{d}\right)$$

Typicality-based inference converges to a hypothesis broader than the truth by the factor $1 + 2/d$: 3× at $d = 1$, 2× at $d = 2$, 1.4× at $d = 5$, 1.2× at $d = 10$. On a discrete grid, the posterior concentrates on the grid point with the largest value of the expression above, which is what P4 checks.

The cause is that the typicality rule drops the log-volume term $-\tfrac{d}{2}\log\sigma_z^2$ that the exact likelihood contains. That term is the same for every hypothesis in the mean-shift family, so mean-shift inference stays consistent; it differs across hypotheses in the scale family. At large $d$ the bias vanishes because the signal grows with $d$ while the dropped term's effect on the limit shrinks as $2/d$.

### A.4 Why equal-mass bins carry no information

The binned log-score of hypothesis $z$ is the log-mass of the bin that $T(x,z)$ falls into. With equal masses, that is $-\log K$ whatever the bin, for every hypothesis and observation, so the posterior equals the prior.

---

## Appendix B: Pilot results (ballpark only, not targets)

From a quick pilot with the normal approximation for bin masses, $M = 20$ hypotheses, 150–400 trials per cell. Use these to sanity-check the implementation's direction and magnitude, not as pass criteria.

**Mean-shift, `p_true(20)`, clean:**

| $d$ | $\delta$ | exact | typ_normal | bins_abs3 |
|---|---|---|---|---|
| 10 | 2 | 1.00 | 0.87 | 0.71 |
| 100 | 2 | 1.00 | 0.15 | 0.11 |
| 100 | 4 | 1.00 | 0.96 | 0.90 |
| 1000 | 4 | 1.00 | 0.22 | 0.17 |
| 1000 | 8 | 1.00 | 0.99 | 0.97 |

**Threshold collapse, `bins_abs3` `n90`:** at $r = 1$: 30, 26, 26 for $d = 10, 100, 1000$; at $r = 1.5$: 8, 6, 6; at $r = 2$: 4, 3, 3.

**Robustness, $d = 1000$, $\delta = 8$, C3 with $R = 50$, `p_true(20)`:**

| $\rho$ | exact | clipped | typ_normal | bins_abs3 |
|---|---|---|---|---|
| 0 | 1.00 | 1.00 | 0.99 | 0.98 |
| 0.1 | 0.41 | 1.00 | 0.13 | 0.97 |
| 0.2 | 0.10 | 0.99 | 0.02 | 0.97 |
| 0.3 | 0.01 | 0.96 | 0.00 | 0.92 |

Tempering with $\zeta = 0.1$ matched `exact` at $n \ge 20$. C5 (sparse corruption in every observation): `bins_abs3` stalled at about 0.07 after 20 observations; `exact` and `typ_normal` were unaffected.

**Scale family, `typ_normal` posterior mode after 50 observations, by true $\sigma^2$ in $(0.25, 0.5, 1, 2, 4)$:** $d = 1$: $(1, 2, 4, 4, 4)$; $d = 2$: $(0.5, 1, 2, 4, 4)$; $d = 10$: correct for every truth. `typ_density` at $d = 1$ chose $\sigma^2 = 4$ for every truth.

**Localization, spacing 1, truth at the center (the final design moves the truth off center), `p_true(20)`:**

| $d$ | exact | typ_density | bins_abs3 |
|---|---|---|---|
| 2 | 0.88 | 0.88 | 0.68 |
| 10 | 0.92 | 0.42 | 0.28 |
| 100 | 0.91 | 0.10 | 0.07 |
| 1000 | 0.91 | 0.03 | 0.03 |
