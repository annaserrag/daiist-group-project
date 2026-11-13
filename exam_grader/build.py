"""Offline build job. No startup downloads. Run as python -m exam_grader.build."""
import argparse
import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .catalog import DATA, QUESTIONS

MODEL = "gemini-embedding-2"
def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def pull(src):
    from lxml import html
    sid, url, title = src
    req = Request(url, headers={"User-Agent": "DAIIST-course-retrieval/1.0"})
    with urlopen(req, timeout=25) as response:
        root = html.fromstring(response.read())
    for node in root.xpath("//nav|//aside|//footer|//script|//style|//header|//button|//*[contains(@class,'page-summary')]"):
        node.drop_tree()
    frag = urlsplit(url).fragment
    areas = (root.xpath("//*[@id=$id]", id=frag) if frag else []) or root.xpath(
        "//*[contains(@class,'devsite-article-body')] | //*[@id='section-content'] | //article | //main | //*[@id='mw-content-text']")
    if not areas:
        raise ValueError("no article body")
    area = next((a for a in areas if "devsite-article-body" in a.get("class","")), areas[-1])
    heading, records, taken = title, [], {}
    for el in area.xpath(".//h2|.//h3|.//p"):
        t = " ".join(el.text_content().split())
        if el.tag in ("h2","h3"):
            heading = t.replace("¶","").strip() or title
            continue
        if len(t) < 60 or len(t) > 2200 or taken.get(heading,0) >= 3:
            continue
        if re.search(r"cookie|JavaScript is disabled|enable JavaScript|sign in to|last updated|creative commons|skip to",t,re.I):
            continue
        records.append(dict(source=sid, title=title, url=url, heading=heading, text=t))
        taken[heading] = taken.get(heading,0)+1
        if len(records) >= 18:
            break
    if not records:
        raise ValueError("no usable paragraphs")
    return records

def corpus():
    manifest = json.loads(Path(__file__).with_name("sources.json").read_text())
    def safely(s):
        try:
            return pull(s)
        except Exception as exc:
            print(f"Skipped {s[0]}: {type(exc).__name__}")
            return []
    with ThreadPoolExecutor(max_workers=5) as pool:
        rows = [r for group in pool.map(safely, manifest) for r in group]
    if not rows:
        raise SystemExit("No sources available; kept earlier corpus.")
    return rows

def chunks(rows, design):
    groups = []
    for row in rows:
        if (design == "section" and groups and all(groups[-1][k] == row[k] for k in ("source","heading"))
                and len(groups[-1]["text"]) + len(row["text"]) < 2400):
            groups[-1]["text"] += " " + row["text"]
        else:
            groups.append(dict(row))
    for row in groups:
        row["id"] = row["source"] + "-" + digest((row["heading"]+row["text"]).encode())[:12]
    return groups

def vector(text, key):
    body = {"model":"models/"+MODEL, "content":{"parts":[{"text":text}]}, "outputDimensionality":256}
    req = Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:embedContent",
                  data=json.dumps(body).encode(), headers={"x-goog-api-key":key,"Content-Type":"application/json"})
    for attempt in range(4):
        try:
            with urlopen(req, timeout=45) as response:
                v = json.load(response)["embedding"]["values"]
            break
        except HTTPError as error:
            if error.code not in (429, 503) or attempt == 3:
                raise
            delay = error.headers.get("Retry-After", "")
            seconds = int(delay) if delay.isdigit() else (20, 40, 60)[attempt]
            if seconds > 120:
                raise
            print(f"Embedding service returned {error.code}. Resuming in {seconds}s.", flush=True)
            time.sleep(seconds)
    norm = math.sqrt(sum(x*x for x in v))
    if len(v) != 256 or not math.isfinite(norm) or not norm:
        raise ValueError("Invalid embedding")
    return [round(x/norm,8) for x in v]

def main():
    import env
    env.load_env()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["sources","vectors","teaching-vectors"])
    p.add_argument("--design", choices=["section","paragraph"],default="section")
    opt = p.parse_args()
    source = DATA/"corpus.json"
    if opt.action == "sources":
        rows = corpus()
        source.write_text(json.dumps(rows,ensure_ascii=False,indent=1),encoding="utf-8")
        print(f"{len(rows)} paragraphs from {len(set(r['source'] for r in rows))} pages. Inspect before building vectors.")
        return
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY to build public-material embeddings.")
    if opt.action == "teaching-vectors":
        notes = (DATA / "teaching-notes.json").read_bytes()
        passages = chunks(json.loads(notes), "paragraph")
        docs = {r["id"]: vector(f"title: {r['title']} / {r['heading']} | text: {r['text']}", key)
                for r in passages}
        target = DATA / "teaching-vectors.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps({"model": MODEL, "notes_hash": digest(notes), "docs": docs},
                                        separators=(",", ":")), encoding="utf-8")
        temporary.replace(target)
        return
    raw = source.read_bytes()
    rows = chunks(json.loads(raw), opt.design)
    cache_path = DATA/".embedding-cache.json"
    cache = json.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    def get(t):
        h = digest((MODEL+":256:"+t).encode())
        if h not in cache:
            cache[h] = vector(t,key)
            temp = cache_path.with_suffix(".tmp")
            temp.write_text(json.dumps(cache))
            temp.replace(cache_path)
        return cache[h]
    obj = {"model":MODEL,"design":opt.design,"corpus_hash":digest(raw),
           "question_hash":digest((DATA/"Question.txt").read_bytes()),
           "docs":{r["id"]:get(f"title: {r['title']} / {r['heading']} | text: {r['text']}") for r in rows},
           "queries":{i:get("task: question answering | query: "+q) for i,q in QUESTIONS.items()}}
    destination = DATA / f"vectors-{opt.design}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj,separators=(",",":")), encoding="utf-8")
    temporary.replace(destination)
    print(f"Built {len(rows)} passages and {len(QUESTIONS)} public exam queries; no student data used.")

if __name__ == "__main__":
    main()
