"""Count distinct examples on the three-item hyperparameter question."""
import re
import unicodedata

VERSION = "distinct-count-v5"

# Use whole list items so a mention inside a negation or explanation cannot earn credit.
# Unrecognized wording that could change the mark goes to the teacher.
EXAMPLES = {
    "neighbor count": ("k", "k in k nn", "k in knn", "k in k nearest neighbors", "number of neighbors", "n neighbors", "n neighbours", "number of neighbours"),
    "learning rate": ("learning rate", "lr", "step size"),
    "number of estimators": ("number of estimators", "number of trees", "number of estimators/trees", "n estimators", "n estimator", "n estimater", "ntree"),
    "maximum tree depth": ("max depth", "maximum depth", "max depth tree", "tree depth", "maximum depth of a tree", "max tree depth"),
    "batch size": ("batch size", "mini batch size", "minibatch size"),
    "maximum leaf nodes": ("max leaf", "max leaves", "max leaf nodes", "maximum leaf nodes", "maximum number of leaves", "max terminal nodes"),
    "number of clusters": ("number of clusters", "n clusters", "k in k means"),
    "training epochs": ("epoch", "number of epochs", "epochs", "n epochs"),
    "dropout rate": ("dropout", "dropout rate", "dropout probability"),
    "regularization strength": ("regularization strength", "regularisation strength", "regularization coefficient", "weight decay"),
    "SVM penalty C": ("c", "svm c", "c in svm"),
    "kernel coefficient": ("gamma", "kernel coefficient"),
    "kernel choice": ("kernel", "kernel type"),
    "momentum": ("momentum",),
    "minimum samples per leaf": ("min samples leaf", "minimum samples per leaf"),
    "minimum samples to split": ("min samples split", "minimum samples to split"),
    "maximum features per split": ("max features", "maximum features per split"),
}
BY_NAME = {alias: name for name, aliases in EXAMPLES.items() for alias in (name.lower(), *aliases)}
EXCLUSIONS = {
    "random forest": "Random Forest names an algorithm.",
    "min depth tree": "Minimum tree depth has no specified model or training setting here.",
    "min tree depth": "Minimum tree depth has no specified model or training setting here.",
    "minimum tree depth": "Minimum tree depth has no specified model or training setting here.",
    "rmse": "RMSE measures prediction error.",
    "root mean squared error": "RMSE measures prediction error.",
    "r2": "R² measures model fit.",
    "r2 score": "R² measures model fit.",
    "accuracy": "Accuracy measures performance.",
}
INTRO = re.compile(
    r"^(?:name\s+)?(?:(?:three|3|a few|few|common|the|some)\s+)*hyper\s*parameters?\b"
    r"(?:(?:\s+(?:of|in)\s+)(?:machine learning|ml)(?:\s+algorithms?)?)?"
    r"(?:\s+(?:are|include))?\s*[:=]?\s*", re.I
)
# A bare model name earns zero only when it accounts for the entire answer.
# Allow a copied question stem and the observed spelling "suport".
SVM_NAME = r"sup{1,2}ort\s+vector\s+machines?"
SVM_ONLY = re.compile(
    rf"(?:is\s+)?(?:svm(?:\s*\(\s*{SVM_NAME}\s*\))?|{SVM_NAME}(?:\s*\(\s*svm\s*\))?)"
)

# Match a complete affirmative definition, not just the words "K" and "clusters".
# Keep the comma in "In K-means, K ..." from splitting the definition in two.
KMEANS = r"k\s*means?(?: (?:clustering|algorithm))?"
CONTEXT = rf"(?:in|for|of) (?:the )?{KMEANS}"
K_SUBJECT = r"(?:the )?(?:value of k|k(?: value)?)"
CLUSTERS = (r"(?:(?:the )?(?:number|count) of clusters?|how many clusters)"
            rf"(?: to (?:use|form|create|find)| are (?:formed|created)| (?:{KMEANS}|the algorithm) (?:forms|creates))?")
VERB = r"(?:is|means|represents|stands for|specifies|sets|controls|determines|defines?|defined)"
RELATION = rf"(?:is (?:a |the )?hyperparameters?(?: {CONTEXT})? (?:that|which) )?{VERB}"
CLUSTER_DEFINITION = re.compile(
    rf"(?:{K_SUBJECT} {CONTEXT}|(?:{CONTEXT} )?{K_SUBJECT}(?: {CONTEXT})? "
    rf"(?:{RELATION} )?{CLUSTERS}(?: {CONTEXT})?)"
)

def assess(text):
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    normalized = re.sub(r"[_‐‑–—-]", " ", normalized)
    model_answer = INTRO.sub("", normalized.removeprefix("the ")).strip(" \t\r\n.•:")
    if SVM_ONLY.fullmatch(model_answer):
        feedback = ("SVM names a model. Q32 asks for three tunable settings. "
                    "You named none, so the answer earns 0/2.")
        return 0, feedback, [{"criterion": "examples", "points": 2, "reason": feedback}], [], "counted_hyperparameters"
    normalized = INTRO.sub("", normalized)
    normalized = re.sub(r"^(?:examples (?:are|include)|for example)\s*[:,]?\s*", "", normalized)
    normalized = re.sub(rf"\b({CONTEXT}),\s*(?={K_SUBJECT}\b)", r"\1 ", normalized)
    credited, notes, unknown = [], [], []
    node_terms = []
    for part in re.split(r"[,;\n•&]|\b(?:and|or)\b", normalized):
        item = " ".join(part.strip(" .:\t").split())
        item = re.sub(r"^\(?\d+[.)]\s*", "", item)
        # A concrete value can accompany the name, e.g. learning_rate=0.01.
        item = re.sub(r"\s*=\s*[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?$", "", item)
        if not item:
            continue
        if item in {"max node", "max nodes", "maxnodes", "maximum nodes", "maximum number of nodes"}:
            node_terms.append(item)
            continue
        name = BY_NAME.get(item)
        if name is None and CLUSTER_DEFINITION.fullmatch(item):
            name = "number of clusters"
        if name:
            if name in credited:
                notes.append(f"Repeated wording for {name} counts once.")
            else:
                credited.append(name)
        elif item in EXCLUSIONS:
            notes.append(EXCLUSIONS[item])
        else:
            unknown.append(item)

    if node_terms and "maximum leaf nodes" in credited:
        if len(credited) < 3:
            feedback = (f"Recognized {', '.join(credited)}. The term '{node_terms[0]}' could mean "
                        "the same limit on leaf nodes or a separate limit on all nodes. "
                        "The teacher should decide whether it counts as another setting.")
            return None, " ".join([feedback, *dict.fromkeys(notes)]), [], credited, "ambiguous_duplicate"
    elif node_terms:
        for term in node_terms:
            if term in ("max node", "maxnodes"):
                if "maximum leaf nodes" not in credited:
                    credited.append("maximum leaf nodes")
                    notes.append("Unqualified max-node is interpreted as a cap on terminal nodes.")
            else:
                unknown.append(term)

    count = len(credited)
    if count == 1:
        feedback = [f"You correctly identified {credited[0]} as a hyperparameter.", *dict.fromkeys(notes)]
        if unknown:
            feedback.append("The teacher will also check the remaining wording for other valid examples.")
        feedback.append("The teacher will decide the mark for this response.")
        return None, " ".join(feedback), [], credited, "single_hyperparameter_review"
    detail = f"Credited {count} distinct setting{'s' if count != 1 else ''}."
    if credited:
        detail += " Counted " + ", ".join(credited) + "."
    detail = " ".join([detail, *dict.fromkeys(notes)])
    if unknown and count < 3:
        return None, detail + " Additional wording needs a teacher to identify or disambiguate it.", [], credited, "uncertain_hyperparameter"
    mark = {0: 0, 2: 1.5, 3: 2}[min(count, 3)]
    deductions = []
    if mark < 2:
        reason = f"The question asks for three distinct hyperparameters and the answer identifies {count}."
        deductions.append({"criterion": "examples", "points": 2 - mark, "reason": reason})
        detail += f" −{2 - mark:g}: {reason}"
    return mark, detail, deductions, credited, "counted_hyperparameters"
