"""Download a Khmer text sample from Khmer Wikipedia for synthetic line rendering.

Text: Khmer Wikipedia contributors, CC BY-SA 4.0. The text is only rendered
into training images; it is not shipped with the model or the SDK.

Requests are batched (intros of 20 random articles per call), spaced one
second apart, and back off when the API answers 429.

    python training/fetch_corpus.py --requests 100   # -> training/data/corpus/kmwiki.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://km.wikipedia.org/w/api.php"
USER_AGENT = "khmerocr-training/0.1 (offline OCR research; low request rate)"


def fetch_batch() -> list[str]:
    params = {"action": "query", "format": "json", "prop": "extracts", "explaintext": "1",
              "exintro": "1", "exlimit": "20", "generator": "random",
              "grnnamespace": "0", "grnlimit": "20"}
    url = f"{API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    return [p.get("extract", "") for p in data.get("query", {}).get("pages", {}).values()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()
    out = HERE / "data" / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "kmwiki.txt"
    seen = set(path.read_text(encoding="utf-8").split("\n")) if path.exists() else set()
    added = 0
    delay = 1.0
    done = 0
    while done < args.requests:
        try:
            texts = fetch_batch()
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                delay = min(delay * 2, 60)
                retry_after = exc.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else delay)
                continue
            raise
        delay = 1.0
        done += 1
        with path.open("a", encoding="utf-8") as handle:
            for paragraph in "\n".join(texts).split("\n"):
                if paragraph.strip() and paragraph not in seen:
                    seen.add(paragraph)
                    handle.write(paragraph + "\n")
                    added += len(paragraph)
        time.sleep(1.0)
    (out / "SOURCE.txt").write_text(
        f"Khmer Wikipedia random article intros, fetched {time.strftime('%Y-%m-%d')}. "
        "License: CC BY-SA 4.0, Wikipedia contributors.\n")
    print(f"added {added:,} characters; corpus now {path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
