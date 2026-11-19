"""Return a sample zone so the UI can be rehearsed without a trained model."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--pitcher", type=int, required=True)
    parser.add_argument("--game-pk", type=int, required=True)
    parser.add_argument("--at-bat-number", type=int, required=True)
    parser.add_argument("--pitch-number", type=int, required=True)
    args = parser.parse_args()

    zones = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14)
    index = (args.game_pk + args.at_bat_number + args.pitch_number) % len(zones)
    print(json.dumps({"zone": zones[index]}))


if __name__ == "__main__":
    main()
