import numpy as np


def survival_at(logits, boundaries, times):
    boundaries = np.asarray(boundaries, dtype=float)
    logs = -np.logaddexp(0, np.asarray(logits, dtype=float)).cumsum()
    result = np.exp(np.interp(times, np.r_[0, boundaries], np.r_[0, logs]))
    result[np.asarray(times) > boundaries[-1]] = np.nan
    return result


def censoring_distribution(time, event):
    unique = np.unique(time)
    probabilities, current = [], 1.0
    for value in unique:
        deaths = np.sum((time == value) & (event == 1))
        censored = np.sum((time == value) & (event == 0))
        risk = np.sum(time >= value) - deaths
        if risk > 0:
            current *= 1 - censored / risk
        probabilities.append(current)
    return unique, np.asarray(probabilities)


def metrics(time, event, survival, grid):
    time, event = np.asarray(time), np.asarray(event)
    if grid[0] != 0 or grid[-1] != 24 or not np.isfinite(survival).all():
        raise ValueError("Evaluation requires complete prediction coverage on 0–24 months")
    risk = 1 - survival[:, -1]
    unique, probabilities = censoring_distribution(time, event)

    def g(values, left=False):
        indices = np.searchsorted(unique, values, side="left" if left else "right") - 1
        return np.r_[1, probabilities][indices + 1]

    event_g = g(time, left=True)
    concordance = (risk[:, None] > risk[None, :]).astype(float) + 0.5 * (risk[:, None] == risk[None, :])
    comparable = (time[:, None] < time[None, :]) | ((time[:, None] == time[None, :]) & (event[None, :] == 0))
    comparable &= event[:, None] == 1
    np.fill_diagonal(comparable, False)
    harrell_den = comparable.sum()
    harrell = float((concordance * comparable).sum() / harrell_den) if harrell_den else np.nan
    eligible = (event == 1) & (time <= 24)
    if np.any(event_g[eligible] <= 0):
        uno = np.nan
    else:
        weights = np.zeros(len(time))
        weights[eligible] = 1 / event_g[eligible] ** 2
        weighted = comparable * weights[:, None]
        uno = float((weighted * concordance).sum() / weighted.sum()) if weighted.sum() else np.nan
    cases, controls = np.flatnonzero(eligible), np.flatnonzero(time > 24)
    if len(cases) and len(controls) and np.all(event_g[cases] > 0) and g(np.array([24]))[0] > 0:
        weights = 1 / event_g[cases]
        auc = float((concordance[np.ix_(cases, controls)] * weights[:, None]).sum() / (weights.sum() * len(controls)))
    else:
        auc = np.nan
    scores = []
    for index, value in enumerate(grid):
        dead = (time <= value) & (event == 1)
        alive = time > value
        gt = g(np.array([value]))[0]
        if (alive.any() and gt <= 0) or np.any(event_g[dead] <= 0):
            scores.append(np.nan)
        else:
            error = (survival[dead, index] ** 2 / event_g[dead]).sum()
            if alive.any():
                error += ((1 - survival[alive, index]) ** 2 / gt).sum()
            scores.append(error / len(time))
    ibs = float(np.trapezoid(scores, grid) / 24) if np.isfinite(scores).all() else np.nan
    return {"uno_c_24": uno, "harrell_c": harrell, "auc_24": auc, "ibs_0_24": ibs}


def evaluate(time, event, survival, grid, bootstrap=2000, seed=20260819):
    estimate = metrics(time, event, survival, grid)
    samples = {key: [] for key in estimate}
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap):
        indices = rng.integers(0, len(time), len(time))
        result = metrics(time[indices], event[indices], survival[indices], grid)
        for key, value in result.items():
            if np.isfinite(value):
                samples[key].append(value)
    return {key: {"estimate": value if np.isfinite(value) else None,
                  "ci95": np.quantile(samples[key], [0.025, 0.975]).tolist() if samples[key] else None,
                  "valid_bootstraps": len(samples[key]), "invalid_bootstraps": bootstrap - len(samples[key])}
            for key, value in estimate.items()}
