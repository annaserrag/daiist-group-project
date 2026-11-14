"""Question-level decisions. All numeric marks are proposals for teacher sign-off."""
import math
import re
import time
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from .catalog import KEY, KEY_URL, OPTIONS, QUESTIONS
from .rubric import CRITERIA
from .rubric import check_interval_example, states_binning_rule, supports_criterion
from .evaluation_cache import MissingFrozenResponse
from . import hyperparameters

@dataclass(frozen=True)
class Policy:
    # Engineering defaults for exercising branches, NOT calibrated decision thresholds.
    min_read: float = 0.80
    min_id: float = 0.85
    min_similarity: float = 0.40
    figure_questions: tuple = (34,)  # Q22 supplies the training and test accuracies in the prompt.
    pass_mark: float | None = None     # The dataset does not establish a pass rule.
    pass_band: float = 0.0
    written_review_at_or_below: float | None = None

    def __post_init__(self):
        if not all(_prob(x) for x in (self.min_read,self.min_id,self.min_similarity)):
            raise ValueError("Thresholds must be finite numbers in [0,1].")
        if (self.pass_mark is not None and not 0 <= self.pass_mark <= 50) or not 0 <= self.pass_band <= 50:
            raise ValueError("Invalid pass boundary for a 50-point exam.")
        cutoff = self.written_review_at_or_below
        if cutoff is not None and (type(cutoff) not in (float,int) or cutoff not in (0,.5,1,1.5,2)):
            raise ValueError("Written review cutoff must be a valid half-point mark out of two.")

def _prob(v):
    return type(v) in (float,int) and math.isfinite(v) and 0 <= v <= 1

def row(q, reason, mark=None, *, feedback="", citation=None, score=None, searches=0):
    return dict(question_id=q, max_mark=1 if q <= 20 else 2, mark=mark,
                status="review" if mark is None else "proposed", reason=reason,
                feedback=feedback, citation=citation, retrieval_score=score, searches=searches)

# The teaching collection is open-ended. Absence from its examples cannot justify a deduction.
_SOURCE_GAP = re.compile(
    r"\b(?:not|never|neither)\s+(?:(?:explicitly|specifically|directly|clearly)\s+)?"
    r"(?:covered|established|mentioned|listed|included|supported|discussed|found|illustrated)\b"
    r".{0,115}?\b(?:course|material|source|excerpt|passage|reference|answer key)\b"
    r"|\b(?:course material|sources?|excerpts?|passages?|answer key)\b.{0,60}?"
    r"\b(?:does not|do not|doesn't|don't|never)\s+(?:cover|establish|mention|list|include|support|discuss)\b",
    re.I,
)


def source_quote(value, text):
    """Return verbatim source text, allowing sentence-initial capitalization in a copied fragment."""
    quote = value.strip() if isinstance(value, str) else ""
    if len(quote) < 12:
        return ""
    if quote in text:
        return quote
    if re.match(r"^(?:A|An|The|This|These|Those)\s", quote):
        original = quote[0].lower() + quote[1:]
        if original in text:
            return original
    return ""


def checked_mark(q, response, hits, searches, *, guide=None):
    score = hits[0]["score"]
    fields = dict(score=score,searches=searches)
    if not isinstance(response,dict) or response.get("needs_review") is not False:
        return row(q,"insufficient_evidence",**fields)
    m = response.get("mark")
    f = response.get("feedback")
    if type(m) not in (int,float) or m not in (0,.5,1,1.5,2) or not isinstance(f,str) or not 1 <= len(f.strip()) <= 350:
        return row(q,"invalid_model_output",**fields)
    deductions = response.get("deductions")
    if not isinstance(deductions, list):
        return row(q, "unexplained_deduction", **fields)
    guide = CRITERIA[q] if guide is None else guide
    seen = set()
    lost = 0
    explanations = []
    positive_deductions = []
    for item in deductions:
        if not isinstance(item, dict):
            return row(q, "unexplained_deduction", **fields)
        criterion, points, reason = (item.get(k) for k in ("criterion", "points", "reason"))
        if (not isinstance(criterion, str) or criterion not in guide or criterion in seen
                or type(points) not in (int, float) or points not in (0, .5, 1, 1.5, 2)
                or points > guide[criterion][0] or not isinstance(reason, str)
                or not 5 <= len(reason.strip()) <= 500):
            return row(q, "unexplained_deduction", **fields)
        seen.add(criterion)
        if points == 0:
            # A zero-loss annotation cannot resolve disagreement between the two assessments.
            second = response.get("independent_mark")
            if type(second) not in (int, float) or second != m:
                return row(q, "mark_disagreement", **fields)
            continue
        positive_deductions.append(item)
        lost += points
        explanations.append(f"−{points:g}: {' '.join(reason.split())}")
    if lost != 2 - m:
        return row(q, "unexplained_deduction", **fields)
    hit = next((r for r in hits if r["id"] == response.get("passage_id")),None)
    quote = source_quote(response.get("quote"), hit["text"]) if hit else ""
    if not quote:
        return row(q,"unverified_citation",**fields)
    cite = {"id":hit["id"],"title":hit["title"],"heading":hit["heading"],"url":hit["url"],"quote":quote.strip()}
    if "page" in hit:
        cite["page"] = hit["page"]
    feedback = " ".join([" ".join(f.split()), *explanations])
    return row(q,"grounded_proposal",m,feedback=feedback,citation=cite,**fields) | {"deductions": positive_deductions}


def checked(q, response, hits, searches, student_text="", *, custom_guide=None):
    """Require source-backed evidence for every criterion, including deductions."""
    decision = checked_mark(q, response, hits, searches, guide=custom_guide)
    guide = CRITERIA[q] if custom_guide is None else custom_guide
    entries = response.get("criterion_evidence") if isinstance(response, dict) else None
    entries = entries if isinstance(entries, list) else []
    sources = {hit["id"]: hit for hit in hits}
    citations = {}
    for criterion in guide:
        # Some evidence reports echo the complete rubric description in the ID field.
        # Only an exact, unambiguous label from this question can identify the criterion.
        accepted = [criterion]
        description = guide[criterion][1]
        if sum(text == description for _, text in guide.values()) == 1:
            accepted.append(description)
        matching = [e for e in entries if isinstance(e, dict) and e.get("criterion") in accepted]
        if len(matching) != 1:
            continue
        e = matching[0]
        hit = sources.get(e["passage_id"]) if isinstance(e.get("passage_id"), str) else None
        quote = source_quote(e.get("quote"), hit["text"]) if hit else ""
        if not quote or e.get("coverage") not in ("full", "partial"):
            continue
        labels = [hit["title"], hit["heading"], *hit.get("labels", [])]
        if any(quote.casefold() in label.casefold() for label in labels):
            continue
        if custom_guide is None and not supports_criterion(q, criterion, quote):
            continue
        citations[criterion] = {k:hit[k] for k in ("id","title","heading","url","page") if k in hit}
        citations[criterion].update(quote=quote, coverage=e["coverage"])
    if custom_guide is None and q == 30 and decision["mark"] is not None:
        valid = check_interval_example(student_text)
        arithmetic = valid is not None
        application = response.get("binning_application") or {}
        if valid is None and isinstance(application, dict):
            quote = application.get("student_quote")
            if (isinstance(quote, str) and quote.strip() and quote in student_text
                    and isinstance(application.get("explanation"), str) and application["explanation"].strip()
                    and application.get("verdict") in ("valid", "invalid")):
                valid = application["verdict"] == "valid"
        example_deducted = any(d["criterion"] == "example" for d in decision["deductions"])
        method = citations.get("method")
        if (valid is True and not example_deducted and method and method["coverage"] == "full"
                and states_binning_rule(method["quote"])):
            basis = ("checked numeric interval membership" if arithmetic else application["explanation"])
            citations["example"] = dict(method, basis=basis)
        elif application and valid is None and not example_deducted:
            citations.pop("example", None)
        elif (valid is False and not example_deducted) or (valid is True and example_deducted):
            # The proposed credit must also agree with the actual interval assignments.
            citations.pop("example", None)
    missing = [c for c in guide if citations.get(c, {}).get("coverage") != "full"]
    if (not missing and decision["reason"] == "unverified_citation"
            and type(response.get("independent_mark")) in (int, float)
            and response.get("needs_review") is False):
        # A malformed summary quote can be replaced only by independently verified body
        # evidence from that SAME source, with full coverage of every teacher criterion.
        replacement = next((cite for cite in citations.values() if cite["id"] == response.get("passage_id")), None)
        if replacement:
            decision = checked_mark(q, response | {"quote":replacement["quote"]}, hits, searches, guide=guide)
    if missing:
        details = " ".join(f"{c}: {guide[c][1]}" for c in missing)
        decision = row(q, "insufficient_evidence", score=hits[0]["score"], searches=searches,
                       feedback="Teacher review. The passages do not fully support these rubric points. " + details)
    elif decision["mark"] is not None and any(_SOURCE_GAP.search(d["reason"]) for d in decision["deductions"]):
        decision = row(q, "unsupported_deduction", score=hits[0]["score"], searches=searches,
                       feedback="Teacher review. A deduction relied on what the course excerpts omit. "
                                "Check whether the student's example satisfies the criterion.")
    elif decision["mark"] is not None and (custom_guide is not None or "independent_mark" in response):
        second = response.get("independent_mark")
        if (type(second) not in (int, float) or second not in (0, .5, 1, 1.5, 2)
                or abs(decision["mark"] - second) >= 1):
            decision = row(q, "mark_disagreement", score=hits[0]["score"], searches=searches,
                           feedback="Teacher review. The two rubric assessments could not establish a "
                                    "consistent mark. Check the student's demonstrated understanding "
                                    "against the cited criteria.")
    decision["citation"] = next(iter(citations.values()), None)
    decision["criterion_evidence"] = citations
    decision["criteria_needing_evidence"] = missing
    return decision

def grade_exam(payload, *, index=None, agent=None, policy=None, allow_remote=False, written_exam=None):
    """Return a teacher view and a separate record suitable for the course logger.
    allow_remote must come from a trusted server-side input provenance check, never from OCR JSON.
    agent.open_session() creates a per-exam meter (concurrent requests cannot mix usage).
    """
    started = time.perf_counter()
    p = policy or Policy()
    envelope = isinstance(payload,dict) and payload.get("schema_version") == "vision.exam.v1"
    status = payload.get("exam_status") if envelope else "invalid_contract"
    if status not in ("in_scope","out_of_scope","unreadable","invalid_contract"):
        status = "invalid_contract"
    answers = payload.get("answers") if envelope else None
    if not isinstance(answers,list) or len(answers)>100 or any(not isinstance(a,dict) for a in (answers or [])):
        status, answers = "invalid_contract", []
    custom = {item.question_id: item for item in written_exam.questions} if written_exam else {}
    questions = {q: item.question for q, item in custom.items()} if custom else QUESTIONS
    if status == "in_scope":
        expected_id = written_exam.exam_id if written_exam else "mendeley-sf3kvjwknt-v1"
        # Legacy exports implied the bundled paper. A custom definition always needs an explicit match.
        supplied_id = payload.get("exam_id")
        if supplied_id is None and not custom:
            supplied_id = expected_id
        if supplied_id != expected_id:
            status = "out_of_scope"
    buckets = {}
    unassigned = 0
    for ans in answers:
        q = ans.get("question_id")
        if type(q) is not int or q not in questions:
            unassigned += 1
        else:
            buckets.setdefault(q,[]).append(ans)
    # An unassigned response could belong to ANY question, even one apparently filled.
    issue = status if status != "in_scope" else ("unassigned_answer" if unassigned else None)
    session = agent.open_session() if agent is not None and allow_remote and not issue else None

    review_passages = {}

    def decide(q):
        if issue:
            return row(q,issue)
        matches = buckets.get(q,[])
        if len(matches) != 1:
            return row(q,"missing_detection" if not matches else "duplicate_detection")
        a = matches[0]
        state,txt,conf,idc = (a.get(k) for k in ("state","text","confidence","id_confidence"))
        if not (_prob(conf) and _prob(idc)) or state not in ("read","confirmed_blank","unreadable") or not isinstance(txt,str) or len(txt)>4000:
            return row(q,"invalid_answer")
        if idc < p.min_id:
            return row(q,"uncertain_question_id")
        if state == "unreadable":
            return row(q,"unreadable")
        txt = txt.strip()
        if state == "confirmed_blank":
            if txt or conf < p.min_read:
                return row(q,"unverified_blank")
            return row(q,"confirmed_blank",0,feedback="Vision explicitly confirmed an empty answer area.")
        if not txt:
            return row(q,"empty_transcription")
        if not custom and q <= 20:
            if conf < p.min_read:
                return row(q,"low_read_confidence")
            letter = txt.upper()
            if letter not in OPTIONS[q]:
                return row(q,"invalid_choice")
            m = int(letter == KEY[q])
            return row(q,"matches_key" if m else "differs_from_key",m,
                       feedback=f"Read {letter}; the published key gives {KEY[q]}.",
                       citation={"id":f"key-Q{q}","title":"Published answer key","url":KEY_URL})
        if (custom[q].requires_figure if custom else q in p.figure_questions):
            return row(q,"context_check")
        if not custom and q == 32:
            if conf < p.min_read:
                return row(q, "low_read_confidence")
            mark, feedback, deductions, credited, reason = hyperparameters.assess(txt)
            cite = {"id": "key-Q32", "title": "Published Q32 answer key", "url": KEY_URL, "quote": KEY[q]}
            return row(q, reason, mark, feedback=feedback, citation=cite) | {
                "credited_examples": credited, "deductions": deductions}
        if index is None:
            return row(q,"no_index")
        hits = index.search(q,txt)
        searches = 1
        if conf < p.min_read or not hits or hits[0]["score"] < p.min_similarity:
            hits = index.search(q)
            searches = 2
        score = hits[0]["score"] if hits else None
        fields = dict(score=score, searches=searches)
        if conf < p.min_read:
            return row(q,"low_read_confidence",**fields)
        if score is None or score < p.min_similarity:
            return row(q,"weak_retrieval",**fields)
        # Keep passages that cleared retrieval available for the teacher even if generation fails.
        review_passages[q] = hits
        if custom and (proof := custom[q].match_full_credit(txt, hits)):
            decision = row(q, "teacher_approved_answer", 2, citation=next(iter(proof.values())),
                           feedback="Matches a complete answer approved for this rubric.", **fields)
            decision.update(criterion_evidence=proof, criteria_needing_evidence=[], deductions=[])
            if p.written_review_at_or_below == 2:
                decision.update(mark=None, status="review", reason="written_mark_review", candidate_mark=2)
            return decision
        if session is None:
            return row(q,"provider_disabled",**fields)
        try:
            if custom:
                answer = session.grade(questions[q], custom[q].reference_answer, txt, hits, guide=custom[q].guide())
            else:
                answer = session.grade(questions[q], KEY[q], txt, hits)
        except MissingFrozenResponse:
            return row(q,"missing_frozen_response",**fields)
        except Exception as error:
            # Do not persist provider error strings; they can echo the prompt.
            failed = row(q,"provider_failure",**fields)
            if hasattr(session,"failure_reason") and (detail := session.failure_reason(error)):
                failed["execution_issue"] = detail
            return failed
        visible = session.visible_evidence(hits) if hasattr(session,"visible_evidence") else hits
        proposal = checked(q,answer,visible,searches,txt, custom_guide=custom[q].guide() if custom else None) | {"opened_passages":list(getattr(session,"last_opened",[]))}
        # Published partial-credit decisions vary. An unconfirmed zero must not
        # become an anchor for the teacher assessing the train/test explanation.
        if not custom and q == 22 and proposal["mark"] is not None and proposal["mark"] < 2:
            proposal.update(mark=None, status="review", reason="diagnosis_partial_review", deductions=[],
                            feedback="Teacher review. Check whether the answer identifies overfitting and "
                                     "explains why performance drops on unseen data. Decide partial credit "
                                     "from the student's wording and the cited course evidence.")
            return proposal
        below_cutoff = (p.written_review_at_or_below is not None and proposal["mark"] is not None
                        and proposal["mark"] <= p.written_review_at_or_below)
        if below_cutoff:
            candidate = proposal["mark"]
            proposal.update(mark=None, status="review", reason="written_mark_review", candidate_mark=candidate,
                            feedback=f"Teacher check requested for the suggested {candidate:g}/2. " + proposal["feedback"])
        return proposal

    out = [decide(q) for q in sorted(questions)]
    for item in out:
        q = item["question_id"]
        item["question"] = questions[q]
        if item["mark"] is None and q in review_passages:
            item["review_passages"] = [
                {k: hit[k] for k in ("id", "title", "heading", "url", "page") if k in hit}
                | {"excerpt": hit["text"][:1200]} for hit in review_passages[q]
            ]
        if custom:
            item["max_mark"] = 2
        item["readings"] = [a["text"][:4000] for a in buckets.get(q, []) if isinstance(a.get("text"), str)]
        item["reading_states"] = [a.get("state") for a in buckets.get(q, []) if isinstance(a.get("text"), str)]
    reviewed = sum(r["mark"] is None for r in out)
    subtotal = sum(r["mark"] for r in out if r["mark"] is not None)
    total = subtotal if not reviewed else None
    near_pass = total is not None and p.pass_mark is not None and abs(total-p.pass_mark) <= p.pass_band
    teacher = {"exam_status":status,"answers":out,"proposed_subtotal":subtotal,"proposed_total":total,
               "maximum":sum(r["max_mark"] for r in out),"review_count":reviewed,"near_pass_boundary":near_pass,"teacher_signoff_required":True}
    def avg(vals):
        return sum(vals)/len(vals) if vals else 0.
    readings = [a["confidence"] for ms in buckets.values() for a in ms if _prob(a.get("confidence"))]
    sims = [r["retrieval_score"] for r in out if r["retrieval_score"] is not None]
    # Only derived finite counts, trusted catalogue IDs and controlled reason codes enter logs.
    a_summary = {str(q):{"detections":len(items),"mean_confidence":avg([a["confidence"] for a in items if _prob(a.get("confidence"))])} for q,items in buckets.items()}
    b_summary = []
    for answer in out:
        summary = {k:answer[k] for k in ("question_id","mark","reason","searches","execution_issue") if k in answer}
        summary["source_id"] = answer["citation"]["id"] if answer["citation"] else None
        b_summary.append(summary)
    elapsed = time.perf_counter()-started
    usage = session.usage() if session else {"tokens":0,"cost_eur":0.,"usage_complete":True,"calls":0}
    record = {"input_summary":f"Exam OCR: {len(answers)} records, {len(buckets)} recognized question IDs.",
              "features":{"mean_read_confidence":avg(readings),"reading_count":len(readings),
                          "mean_retrieval_similarity":avg(sims),"retrieval_count":len(sims),
                          "review_share":reviewed/len(out),"unassigned_answers":unassigned},
              "outputs":{"component_a":a_summary,"component_b":{"answers":b_summary,"near_pass_boundary":near_pass,
                        "usage_complete":usage["usage_complete"],"cost_known":usage.get("cost_known",usage["usage_complete"]),
                        "calls":usage["calls"],"cache_hits":usage.get("cache_hits",0),
                        "input_tokens":usage.get("input_tokens",0 if not usage["calls"] else None),
                        "output_tokens":usage.get("output_tokens",0 if not usage["calls"] else None),
                        "uncached_tokens":usage.get("uncached_tokens",usage["tokens"]),
                        "uncached_cost_eur":usage.get("uncached_cost_eur",usage["cost_eur"] if usage.get("cost_known",True) else None),
                        "latency_s":elapsed}},
              "versions":{"component_a":"UNSET-by-vision-owner","component_b":getattr(agent,"version","no-provider"),
                          "retrieval":getattr(index,"version","no-index"),"contract":"vision.exam.v1","q32_rule":hyperparameters.VERSION,
                          "evidence_rule":"per-criterion-v6-verified-summary",
                          "policy":sha256(json.dumps(asdict(p),sort_keys=True).encode()).hexdigest()[:12]},
              "review":bool(reviewed or near_pass),"tokens":usage["tokens"],"cost_eur":usage["cost_eur"]}
    return {"teacher":teacher,"record":record}
