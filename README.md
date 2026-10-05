# FPL Optimizer

[![CI](https://github.com/piotrstoklosa144-jpg/fpl-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/piotrstoklosa144-jpg/fpl-optimizer/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piotrstoklosa144-jpg/fpl-optimizer/blob/main/notebooks/demo.ipynb)

Finds the best **Fantasy Premier League** transfer for your team. It pulls live data from the
official FPL API, predicts every player's points with a **Poisson model**, chooses the optimal
transfer, starting XI and captain with **mixed-integer linear programming**, and measures the
risk of the decision with a **Monte Carlo simulation**.

## Example

```text
$ fpl-optimizer --squad-file examples/squad.txt

==================== BEST MOVE (max 1 transfer(s), budget £99.0m) ====================
OUT: Cherki  ->  IN: E.Le Fée

    web_name position team_name  cost  x_mins         role  xp_total
        Raya      GKP       ARS  6.10   90.00      Starter     18.96
      Thomas      DEF       COV  4.00   87.40      Starter     20.20
        Hall      DEF       NEW  5.30   89.80      Starter     19.80
      Mbeumo      MID       MUN  7.90   90.00 Captain (x2)     28.33
    E.Le Fée      MID       SUN  5.70   87.60      Starter     24.49
     Haaland      FWD       MCI 15.60   90.00 Vice-captain     27.83
         ...
Expected points after transfer: 266.25 (+8.20)

==================== MONTE CARLO (10,000 simulations) ====================
                mean  median   p10   p90  std
Current squad  257.7   256.6 219.0 297.9 30.9
After transfer 265.9   265.0 226.4 306.4 31.1

Probability the transfer scores more points: 77.0%
Check: analytical 258.05 vs simulated mean 257.71
```

*(Gameweeks 6–10 of the 2026/27 season, 5-gameweek horizon.)*

## How it works

```
FPL API ──► data.py ──► model.py ──► optimizer.py ──► simulation.py
            players,     expected     best transfer,   distribution of
            fixtures,    points (xP)  XI and captain   points & risk
            team form                 (MILP)           (Monte Carlo)
```

**1. Player rates with Bayesian shrinkage.** A player's xG, xA and defensive actions per 90
minutes are blended with a positional prior worth 300 minutes:

$$\text{rate}_{90} = \frac{\text{total} + \frac{300}{90}\cdot\text{baseline}_{pos}}{\frac{\text{minutes}}{90} + \frac{300}{90}}$$

A defender with one lucky goal in 90 minutes is not projected to score every week; as minutes
grow, the observed rate takes over.

**2. Poisson expected-points model.** For every fixture in the horizon:

- goals and assists ~ Poisson(rate × minutes × opponent defensive weakness × home/away factor),
- team goals conceded ~ Poisson(opponent xG × own xGA / league average), so
  P(clean sheet) = e<sup>−λ</sup> and the "−1 per 2 goals conceded" rule uses E[⌊G/2⌋],
- defensive contribution points (2025/26 rule) = P(Poisson(actions) ≥ 10 or 12),
- playing time is split into *starts* and *substitute appearances*, so clean sheets and
  the 2-point appearance bonus only go to players likely to play 60+ minutes.

Double gameweeks are handled by summing all fixtures in a gameweek.

**3. MILP optimization (PuLP).** Binary variables decide who is in the squad, who starts and who
is captain. The model maximises expected points of the XI + captain (+ a small weight for the
bench) subject to: budget, 2/5/5/3 squad shape, max 3 players per club, a valid formation and at
most *N* transfers from the current squad.

**4. Monte Carlo simulation.** 10,000 seasons-in-miniature are sampled from the same
distributions. Goals conceded are drawn once per team and match, so clean sheets of defenders
from the same club are correlated, as in real life. The output is the full distribution
(median, 10th/90th percentiles) and the probability that the transfer actually beats keeping
the current squad. The simulated mean matching the analytical value is a built-in consistency
check of both implementations.

## Quick start

```bash
git clone https://github.com/piotrstoklosa144-jpg/fpl-optimizer.git
cd fpl-optimizer
pip install .

# using your FPL team id (the number in fantasy.premierleague.com/entry/<ID>/event/...)
fpl-optimizer --team-id 1234567

# or a text file with 15 player names (see examples/squad.txt)
fpl-optimizer --squad-file examples/squad.txt --bank 0.5
```

| Option | Default | Description |
|---|---|---|
| `--team-id` | – | FPL team id; squad and bank are downloaded |
| `--squad-file` | – | 15 names, one per line; use `Name\|TEAM` for duplicates |
| `--bank` | 0.0 | money in the bank (with `--squad-file`) |
| `--transfers` | 1 | maximum number of transfers |
| `--horizon` | 5 | gameweeks to plan for |
| `--sims` | 10000 | Monte Carlo simulations |

Or run it in the browser without installing anything: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/piotrstoklosa144-jpg/fpl-optimizer/blob/main/notebooks/demo.ipynb)

## Project structure

```
src/fpl_optimizer/
├── config.py       # FPL scoring rules and model parameters
├── data.py         # API client and data preparation
├── squad.py        # resolving player names, squad validation
├── model.py        # shrinkage + Poisson expected-points model
├── optimizer.py    # MILP: transfers, line-up, captain
├── simulation.py   # Monte Carlo risk analysis
└── cli.py          # command-line interface
tests/              # pytest suite, runs offline on synthetic data
```

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

The test suite runs without network access. Among others it checks that the Monte Carlo mean
converges to the analytical model, that clean sheets are correlated only within a team, and that
the optimizer respects the budget, squad shape and 3-per-club rule. CI runs lint and tests on
Python 3.10–3.13 for every push.

## Limitations and ideas

- Playing time is estimated from season-long starts, so players returning from injury or new
  signings are underrated.
- Automatic substitutions, goalkeeper save points and points hits for extra transfers are not
  modelled yet.
- Selling prices are approximated by current prices (the real selling price requires login).
- Next steps: backtesting on past gameweeks, multi-week transfer planning, chip strategy.

## Disclaimer

Not affiliated with the Premier League or Fantasy Premier League. Data comes from the public
FPL API.

## License

[MIT](LICENSE)
