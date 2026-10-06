"""
Process the 10 EPL 2021-2022 games with preprocess/extract_clips_frames_label.py.
Output is organized by EVENT LABEL:

    results_by_label_epl_2021_2022/game_NN_half_H/{clips,frames}/<label>/...

Usage:
    python process_10_games.py
    python process_10_games.py --filter-half      # only keep each half's own events
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 10 EPL 2021-2022 games (with date prefixes); index + 1 = game number
GAMES = [
    "2021-08-22_arsenal-fc-chelsea-fc-premier-league-2021-2022",
    "2021-08-28_manchester-city-arsenal-fc-premier-league-2021-2022",
    "2021-09-11_manchester-united-newcastle-united-premier-league-2021-2022",
    "2021-09-19_tottenham-hotspur-chelsea-fc-premier-league-2021-2022",
    "2021-09-25_manchester-united-aston-villa-premier-league-2021-2022",
    "2021-10-03_liverpool-fc-manchester-city-premier-league-2021-2022",
    "2021-10-16_leicester-city-manchester-united-premier-league-2021-2022",
    "2021-10-24_manchester-united-liverpool-fc-premier-league-2021-2022",
    "2021-10-24_west-ham-united-tottenham-hotspur-premier-league-2021-2022",
    "2021-11-06_manchester-united-manchester-city-premier-league-2021-2022",
]

SCRIPT = Path(__file__).parent / "preprocess" / "extract_clips_frames_label.py"


def find_json_file(json_base: Path, game_folder_name: str):
    """Find the annotation JSON for a game folder name."""
    game_name_no_date = game_folder_name.split("_", 1)[1] if "_" in game_folder_name else game_folder_name

    for name in (game_folder_name, game_name_no_date):
        folder = json_base / name
        if folder.exists():
            json_files = list(folder.glob("*.json"))
            if json_files:
                return json_files[0]

    if not json_base.exists():
        return None

    # Fuzzy match on team-name parts
    game_parts = game_name_no_date.replace("-premier-league-2021-2022", "").split("-")
    for folder in json_base.iterdir():
        if folder.is_dir():
            folder_parts = folder.name.split("-")
            matches = sum(1 for part in game_parts if part in folder_parts)
            if matches >= len(game_parts) - 2:
                json_files = list(folder.glob("*.json"))
                if json_files:
                    return json_files[0]
    return None


def process_game_half(video_path: Path, json_path: Path, out_dir: Path, half_num: int, filter_half: bool):
    """Run the extractor on one half of a game."""
    print(f"  [half_{half_num}] {video_path.name}  ({json_path.name})")
    cmd = [sys.executable, str(SCRIPT),
           "--video", str(video_path.resolve()),
           "--json", str(json_path.resolve()),
           "--out", str(out_dir.resolve())]
    if filter_half:
        cmd += ["--half", str(half_num)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        print(f"  [half_{half_num}] timed out (20 min limit)")
        return False
    if result.returncode != 0:
        print(f"  [half_{half_num}] failed: {result.stderr[-300:]}")
        return False
    print(f"  [half_{half_num}] complete")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-base", default="data/videos")
    parser.add_argument("--json-base", default="data/json/Soccer_Data_Json/train/england_epl_2021-2022")
    parser.add_argument("--out", default="results_by_label_epl_2021_2022")
    parser.add_argument("--filter-half", action="store_true",
                        help="Pass --half to the extractor so each half's video only gets that half's events. "
                             "Off by default to reproduce the original 627-event extraction.")
    args = parser.parse_args()

    video_base, json_base, output_base = Path(args.video_base), Path(args.json_base), Path(args.out)
    output_base.mkdir(parents=True, exist_ok=True)

    processed, failed = 0, []
    for i, game_name in enumerate(GAMES, 1):
        print(f"\nGame {i}/{len(GAMES)}: {game_name}")

        video_folder = video_base / game_name
        json_file = find_json_file(json_base, game_name)
        video_h1 = sorted(video_folder.glob("*_1.mkv")) if video_folder.exists() else []
        video_h2 = sorted(video_folder.glob("*_2.mkv")) if video_folder.exists() else []

        if not video_h1 or not video_h2:
            print(f"  missing video halves in {video_folder}")
            failed.append(game_name)
            continue
        if not json_file:
            print(f"  no annotation JSON found under {json_base}")
            failed.append(game_name)
            continue

        ok1 = process_game_half(video_h1[0], json_file, output_base / f"game_{i:02d}_half_1", 1, args.filter_half)
        ok2 = process_game_half(video_h2[0], json_file, output_base / f"game_{i:02d}_half_2", 2, args.filter_half)
        if ok1 and ok2:
            processed += 1
        else:
            failed.append(game_name)

    print(f"\nProcessed {processed}/{len(GAMES)} games -> {output_base.resolve()}")
    if failed:
        print("Failed:")
        for g in failed:
            print(f"  - {g}")


if __name__ == "__main__":
    main()
