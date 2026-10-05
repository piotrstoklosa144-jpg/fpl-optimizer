"""Command-line interface: `fpl-optimizer --team-id 123456`."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import config
from .data import fetch_team_squad, load_fpl_data
from .model import add_expected_points
from .optimizer import lineup_points, optimize_transfers, pick_lineup
from .simulation import lineup_distribution, simulate_points, summarize
from .squad import read_squad_file, resolve_squad_names, validate_squad

TABLE_COLS = ["web_name", "position", "team_name", "cost", "x_mins", "role", "xp_total"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fpl-optimizer",
        description="Find the best Fantasy Premier League transfer using a Poisson model, "
                    "MILP optimization and Monte Carlo simulation.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--team-id", type=int,
                        help="FPL team id (number in the URL .../entry/<ID>/event/...)")
    source.add_argument("--squad-file",
                        help="text file with 15 player names, one per line ('Name' or 'Name|TEAM')")
    parser.add_argument("--bank", type=float, default=0.0,
                        help="money in the bank in £m (only with --squad-file)")
    parser.add_argument("--transfers", type=int, default=1, help="max number of transfers (default: 1)")
    parser.add_argument("--horizon", type=int, default=config.DEFAULT_HORIZON,
                        help="number of gameweeks to plan for (default: %(default)s)")
    parser.add_argument("--sims", type=int, default=config.DEFAULT_N_SIMS,
                        help="Monte Carlo simulations (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="random seed")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    print("Fetching FPL data...")
    data = load_fpl_data(args.horizon)
    players = add_expected_points(data)
    gws = data.target_gws
    print(f"Planning horizon: gameweeks {gws[0]}-{gws[-1]}")

    if args.team_id is not None:
        if data.current_gw is None:
            raise ValueError("The season has not started yet - use --squad-file instead.")
        squad_ids, bank = fetch_team_squad(args.team_id, data.current_gw)
    else:
        squad_ids, bank = resolve_squad_names(players, read_squad_file(args.squad_file)), args.bank
    validate_squad(players, squad_ids)

    current = pick_lineup(players[players["id"].isin(squad_ids)])
    current_xp = lineup_points(current)
    _print_header("CURRENT SQUAD")
    print(current[TABLE_COLS].to_string(index=False))
    print(f"Expected points (XI + captain, {len(gws)} GWs): {current_xp:.2f}")

    budget = players.loc[players["id"].isin(squad_ids), "cost"].sum() + bank
    new_squad = optimize_transfers(players, squad_ids, budget, max_transfers=args.transfers)
    new = pick_lineup(new_squad)
    new_xp = lineup_points(new)

    name_of = players.set_index("id")["web_name"]
    out_ids = [i for i in squad_ids if i not in set(new_squad["id"])]
    in_ids = [i for i in new_squad["id"] if i not in set(squad_ids)]

    _print_header(f"BEST MOVE (max {args.transfers} transfer(s), budget £{budget:.1f}m)")
    if out_ids:
        print(f"OUT: {', '.join(name_of[i] for i in out_ids)}  ->  IN: {', '.join(name_of[i] for i in in_ids)}\n")
        print(new[TABLE_COLS].to_string(index=False))
    else:
        print("No transfer - the current squad is already optimal.")
    print(f"Expected points after transfer: {new_xp:.2f} ({new_xp - current_xp:+.2f})")

    _print_header(f"MONTE CARLO ({args.sims:,} simulations)")
    sims = simulate_points(players, set(squad_ids) | set(new_squad["id"]), data, args.sims, args.seed)
    dist_now, dist_new = lineup_distribution(current, sims), lineup_distribution(new, sims)
    summary = pd.DataFrame({"Current squad": summarize(dist_now), "After transfer": summarize(dist_new)}).T
    print(summary.to_string(float_format="{:.1f}".format))
    if out_ids:
        print(f"\nProbability the transfer scores more points: {(dist_new > dist_now).mean():.1%}")
    print(f"Check: analytical {current_xp:.2f} vs simulated mean {dist_now.mean():.2f}")


def _print_header(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page that can't print names like "Guéhi" or "£"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    pd.set_option("display.float_format", "{:.2f}".format)
    try:
        run(args)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
