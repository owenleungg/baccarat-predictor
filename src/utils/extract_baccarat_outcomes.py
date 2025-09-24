
#!/usr/bin/env python3
"""
Extract baccarat outcomes from JSON files and create CSV files.
Input JSON format: {"hands": [[player_cards], [banker_cards]], ...}
Each card is a rank+suite like "8S", "0D", "AS" where "0" means ten.

Usage:
  python extract_baccarat_outcomes.py

Behavior:
  - Reads all .json files from the 'data' folder
  - Writes <stem>_outcomes.csv files to the 'raw_data' folder
"""

import json, re, sys
from pathlib import Path
from typing import List
import pandas as pd

# ---------- Card parsing and outcome labelling ----------

RANK_VAL = {**{str(i): i % 10 for i in range(2,10)}, 'A':1, '0':0, 'J':0, 'Q':0, 'K':0}

def card_value(card: str) -> int:
    m = re.match(r'([0A23456789JQK])', card)
    if not m:
        raise ValueError(f"Bad card format: {card}")
    return RANK_VAL[m.group(1)]

def hand_total(cards: List[str]) -> int:
    return sum(card_value(c) for c in cards) % 10

def outcome_from_hands(p_cards: List[str], b_cards: List[str]) -> str:
    pt, bt = hand_total(p_cards), hand_total(b_cards)
    if pt > bt: return "P"
    if bt > pt: return "B"
    return "T"

def load_outcomes_from_json(path: Path) -> List[str]:
    d = json.loads(path.read_text())
    outcomes = []
    for pair in d["hands"]:
        if isinstance(pair, list) and len(pair) == 2:
            p_cards, b_cards = pair
        elif isinstance(pair, dict):
            p_cards = pair.get("player") or pair.get("P")
            b_cards = pair.get("banker") or pair.get("B")
            if p_cards is None or b_cards is None:
                raise ValueError(f"Dict hand missing keys in {path.name}")
        else:
            raise ValueError(f"Unexpected hand structure in {path.name}")
        outcomes.append(outcome_from_hands(p_cards, b_cards))
    return outcomes


# ---------- Main ----------

def main():
    in_dir = Path("../../data/raw")
    out_dir = Path("../../data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob("*.json") if p.is_file()])
    if not files:
        print(f"No JSON files in {in_dir}", file=sys.stderr)
        sys.exit(1)

    for path in files:
        try:
            seq = load_outcomes_from_json(path)
            stem = path.stem
            pd.DataFrame({"outcome": seq}).to_csv(out_dir / f"{stem}_outcomes.csv", index=False)
            print(f"Processed {path.name}: {len(seq)} hands -> outcomes.csv")
        except Exception as e:
            print(f"ERROR {path.name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
