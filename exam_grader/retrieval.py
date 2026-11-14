"""Fixed exam: precompute the question embeddings, so search never sends OCR to an embedding API."""
import json
import math
import re
from collections import Counter
from .build import chunks, digest, MODEL
from .catalog import DATA, QUESTIONS

STOP = set("the a an in of on to with from by its is was and or for it this what which be does how are you".split())
def words(s):
    return Counter(w for w in re.findall(r"[a-z0-9]+",s.lower()) if w not in STOP)

NEIGHBORS = re.compile(r"\b(?:k[\s-]*nn|(?:k[\s-]*)?(?:nearest[\s-]*)?neighbou?rs?(?:classifier|regressor)?)\b", re.I)
SCALING = re.compile(r"\b(?:scaling|scaled|rescal(?:e[ds]?|ing)|normali[sz](?:ation|e[ds]?|ing)|standardi[sz](?:ation|e[ds]?|ing)|standardscaler|feature scales?)\b", re.I)
DISTANCE = re.compile(r"\bdistances?\b", re.I)
OUTLIER = re.compile(r"\boutliers?\b", re.I)
TREATMENT = re.compile(r"\b(?:clipping|capping|trimming|winsori[sz]\w*|retain\w*|discard\w*)\b", re.I)


def covers_treatment(hit):
    body = hit["text"]
    if not OUTLIER.search(body):
        return False
    return (any(OUTLIER.search(sentence) and TREATMENT.search(sentence)
                for sentence in re.split(r"[.!?]\s+", body))
            or bool(re.search(r"\b(?:correct|remove|discard)\b.{0,40}\b(?:error|observation)s?\b", body, re.I)))

# The teaching notes use names for the concepts behind these broader exam prompts.
# Expand only the published question so student wording cannot select a topic.
ML = re.compile(r"\b(?:machine[\s-]+learning|ml)\b", re.I)
CHALLENGE = re.compile(r"\b(?:challenges?|difficulties|obstacles?|pitfalls?)\b", re.I)
DISCRETIZATION = re.compile(r"\bdiscreti[sz](?:e[ds]?|ing|ation)\b", re.I)

# In an open question, use the concepts the student actually discusses to diversify
# the evidence window. A match nominates a passage; the grader still verifies the quote.
CHALLENGE_FACETS = [
    (r"\b(?:over|under)[ -]?fit", r"\b(?:over|under)[ -]?fit\w*\b.{0,140}\b(?:training|new data|generaliz)"),
    (r"\b(?:find\w*|collect\w*|gather\w*)\s+(?:a\s+|the\s+)?data|\bdata\s+collection",
     r"\b(?:collect|construct|gather)\w*\b.{0,150}\b(?:time|cost|effort)\b"),
    (r"\bmissing\b", r"\b(?:missing|incomplete)\b.{0,180}\b(?:fix|eliminate|imput|discard)"),
    (r"\boutliers?\b", r"\boutliers?\b.{0,180}\b(?:error|investigat|detect|flag)"),
]

def challenge_evidence(ranked, answer):
    selected = []
    for cue, standard in CHALLENGE_FACETS:
        if re.search(cue, answer, re.I):
            hit = next((r for r in ranked[:30] if re.search(standard, r["text"], re.I)), None)
            if hit and hit not in selected:
                selected.append(hit)
    # Preserve the leading relevance score used for the retrieval threshold.
    front = [ranked[0], *[r for r in selected if r != ranked[0]][:2]]
    return front + [r for r in ranked if r not in front]

def query_expansion(question):
    if ML.search(question) and CHALLENGE.search(question):
        return words("overfitting underfitting overfit underfit generalization training data quality quantity noisy labels")
    if DISCRETIZATION.search(question):
        return words("binning bins buckets bucketing group numerical values intervals categories equally spaced buckets")
    return Counter()

def topic_terms(topic, *, distance_evidence):
    """Group equivalent K-NN and scaling terminology in the lexical vocabulary."""
    if not (NEIGHBORS.search(topic) and SCALING.search(topic)):
        return Counter()
    terms = Counter({":neighbors_scaling": 1})
    if distance_evidence:
        # Weight the three linked concepts more than a broad topic/title match.
        terms[":neighbors_scaling_distance"] = 3
    return terms

class Index:
    def __init__(self, design="section", semantic=False, *, multipart=True, teaching_notes=False):
        if design not in ("section","paragraph"):
            raise ValueError("Unknown chunking design")
        raw = (DATA/"corpus.json").read_bytes()
        self.rows = chunks(json.loads(raw),design)
        self.multipart = multipart
        self.vectors = None
        self.version = f"{design}:{digest(raw)[:12]}:lexical-v3"
        if semantic:
            self.vectors = json.loads((DATA/f"vectors-{design}.json").read_bytes())
            if (self.vectors["model"] != MODEL or self.vectors["corpus_hash"] != digest(raw)
                    or self.vectors["question_hash"] != digest((DATA/"Question.txt").read_bytes())
                    or self.vectors["design"] != design):
                raise ValueError("Rebuild index: manifest does not match")
            self.version = f"{design}:{digest(raw)[:12]}:{MODEL}:hybrid-v3"
        if multipart:
            self.version += ":multipart-v2"
        if teaching_notes:
            raw_notes = (DATA / "teaching-notes.json").read_bytes()
            notes = chunks(json.loads(raw_notes), "paragraph")
            if self.vectors:
                extra = json.loads((DATA / "teaching-vectors.json").read_bytes())
                if (extra["notes_hash"] != digest(raw_notes) or extra["model"] != MODEL
                        or set(extra["docs"]) != {row["id"] for row in notes}):
                    raise ValueError("Rebuild teaching vectors: manifest does not match")
                self.vectors["docs"].update(extra["docs"])
            self.rows.extend(notes)
            self.version += ":notes-" + digest(raw_notes)[:12]
        self._prepare_bags()

    @classmethod
    def from_passages(cls, rows, vectors, version, *, multipart=True):
        """Build an independent index for the active course in a teacher's session."""
        index = cls.__new__(cls)
        index.rows, index.vectors = rows, vectors
        index.version, index.multipart = version, multipart
        index._prepare_bags()
        return index

    def _prepare_bags(self):
        bags = []
        for row in self.rows:
            topic = " ".join((row["title"], row["heading"], row["text"]))
            bags.append(words(topic) + topic_terms(topic, distance_evidence=DISTANCE.search(row["text"])))
        df = Counter(w for b in bags for w in b)
        self.idf = {w: math.log((len(bags)+1)/(n+1))+1 for w,n in df.items()}
        self.bags = [self.weight(b) for b in bags]

    def weight(self,bag):
        vals = {w:(1+math.log(n))*self.idf[w] for w,n in bag.items() if w in self.idf}
        norm = math.sqrt(sum(x*x for x in vals.values())) or 1
        return {w:v/norm for w,v in vals.items()}

    def for_questions(self, questions, query_vectors, exam_id):
        from copy import copy
        index = copy(self)
        index.questions = questions
        index.vectors = (dict(self.vectors, queries=query_vectors)
                         if self.vectors is not None and query_vectors else None)
        index.version = self.version + ":" + exam_id + (":semantic" if index.vectors else ":lexical")
        return index

    def search(self, qid, answer="", k=3):
        question = getattr(self, "questions", QUESTIONS)[qid]
        # Distance is implicit when the question asks about scaling in a neighbor method.
        q = self.weight(words(question+" "+answer) + query_expansion(question)
                        + topic_terms(question, distance_evidence=True))
        results = []
        for row,b in zip(self.rows,self.bags):
            lex = sum(v*b.get(w,0) for w,v in q.items())
            score = lex
            if self.vectors:
                vec = self.vectors["docs"][row["id"]]
                sem = sum(x*y for x,y in zip(vec,self.vectors["queries"][str(qid)],strict=True))
                score = (0.8*sem + 0.2*lex) if answer else sem
            results.append(dict(row,score=score))
        ranked = sorted(results,key=lambda r:(-r["score"],r["id"]))
        if self.multipart and ranked and answer and ML.search(question) and CHALLENGE.search(question):
            ranked = challenge_evidence(ranked, answer)
        if (self.multipart and OUTLIER.search(question) and re.search(r"\b(?:detect|identif|find)\w*", question, re.I)
                and re.search(r"\b(?:handl|treat|manag)\w*", question, re.I)
                and not any(covers_treatment(hit) for hit in ranked[:3])):
            position = next((i for i, hit in enumerate(ranked[3:30], start=3) if covers_treatment(hit)), None)
            if position is not None:
                ranked.insert(2, ranked.pop(position))
        return ranked[:k]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect candidate teaching pages before filling in your own source labels.")
    parser.add_argument("question",type=int,choices=range(1,36),metavar="Q")
    parser.add_argument("--design",choices=["section","paragraph"],default="section")
    parser.add_argument("--semantic",action="store_true")
    args = parser.parse_args()
    index = Index(args.design,args.semantic)
    print(QUESTIONS[args.question])
    for hit in index.search(args.question,k=5):
        print(f"\n{hit['source']}  {hit['score']:.3f}  {hit['title']} / {hit['heading']}\n{hit['url']}\n{hit['text']}")
