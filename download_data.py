"""
Download game videos (.mkv halves) from the SoccerReplay-1988 Dropbox shared folder.

Credentials are never stored in code. Put them in a .env file next to this
script (it is git-ignored) or set them as environment variables:

    DROPBOX_TOKEN=<access token with sharing.read + files.content.read scopes>
    DROPBOX_SHARED_URL=<shared folder link>
    DROPBOX_LINK_PASSWORD=<shared link password>

Usage:
    python download_data.py                       # the 10 games in process_10_games.py
    python download_data.py --games-file my_games.json
    python download_data.py --dry-run             # list files without downloading
"""

import argparse
import json
import os
import pathlib
import sys
import time

DEFAULT_LEAGUE_DIR = "/england_epl_2021-2022"


def load_dotenv(path: pathlib.Path):
    """Minimal .env reader: KEY=VALUE lines, # comments; does not override existing env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"Missing {name}. Set it in .env or the environment (see .env.example).")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games-file", default=None,
                        help="JSON list of game folder names (default: GAMES from process_10_games.py)")
    parser.add_argument("--league-dir", default=DEFAULT_LEAGUE_DIR)
    parser.add_argument("--out", default="data/videos")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(pathlib.Path(__file__).parent / ".env")
    token = require_env("DROPBOX_TOKEN")
    shared_url = require_env("DROPBOX_SHARED_URL")
    link_password = os.environ.get("DROPBOX_LINK_PASSWORD", "")

    import requests  # imported here so --help works without it

    if args.games_file:
        games = json.loads(pathlib.Path(args.games_file).read_text(encoding="utf-8"))
    else:
        from process_10_games import GAMES
        games = GAMES

    session = requests.Session()
    shared_link = {"url": shared_url}
    if link_password:
        shared_link["password"] = link_password

    def list_folder(path):
        r = session.post(
            "https://api.dropboxapi.com/2/files/list_folder",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"path": path, "shared_link": shared_link},
            timeout=60,
        )
        if r.status_code == 401:
            sys.exit("Dropbox returned 401: the access token is invalid or expired. Generate a new one.")
        r.raise_for_status()
        return r.json()["entries"]

    def download_file(remote_path, local_path):
        tmp = local_path.with_suffix(local_path.suffix + ".part")
        api_arg = {"url": shared_url, "path": remote_path}
        if link_password:
            api_arg["link_password"] = link_password
        with session.post(
            "https://content.dropboxapi.com/2/sharing/get_shared_link_file",
            headers={"Authorization": f"Bearer {token}", "Dropbox-API-Arg": json.dumps(api_arg)},
            stream=True, timeout=120,
        ) as r:
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            done, t0 = 0, time.time()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    done += len(chunk)
        tmp.replace(local_path)
        speed = done / max(time.time() - t0, 1e-6) / 1e6
        print(f"    {local_path.name}  {done / 1e9:.2f} GB  ({speed:.1f} MB/s)", flush=True)
        return done

    out_base = pathlib.Path(args.out)
    total_bytes, failures = 0, []
    for gi, game in enumerate(games, 1):
        print(f"[{gi}/{len(games)}] {game}", flush=True)
        try:
            entries = list_folder(f"{args.league_dir}/{game}")
        except Exception as e:
            print(f"    list failed: {e}", flush=True)
            failures.append(game)
            continue
        out_dir = out_base / game
        out_dir.mkdir(parents=True, exist_ok=True)
        for e in entries:
            if e[".tag"] != "file" or not e["name"].endswith(".mkv"):
                continue
            local = out_dir / e["name"]
            if local.exists() and local.stat().st_size == e["size"]:
                print(f"    {e['name']}  already downloaded", flush=True)
                continue
            if args.dry_run:
                print(f"    would download {e['name']}  ({e['size'] / 1e9:.2f} GB)", flush=True)
                continue
            for attempt in range(3):
                try:
                    total_bytes += download_file(f"{args.league_dir}/{game}/{e['name']}", local)
                    break
                except Exception as ex:
                    print(f"    attempt {attempt + 1} failed: {ex}", flush=True)
                    time.sleep(5)
            else:
                print(f"    giving up on {e['name']}", flush=True)
                failures.append(f"{game}/{e['name']}")

    print(f"\nDone. Downloaded {total_bytes / 1e9:.1f} GB.")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
