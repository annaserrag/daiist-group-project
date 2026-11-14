"""Written marking guide derived from the public questions and instructor key."""
import re
from decimal import Decimal


# These questions ask for an equivalent definition or an application of a rule.
# The source supplies the rule; the student's words and numbers supply the application.
EVIDENCE_GUIDANCE = {
    24: (
        "Identify the two distinct difficulties the student describes. A short phrase can explain a "
        "challenge by specifying the work it requires, an obstacle it creates, or competing risks "
        "the practitioner must avoid. A topic or operation named alone earns only the rubric's "
        "half point. Do not invent an explanation, count synonyms twice, or require the source to "
        "use the student's wording. Match each challenge to its own supporting body text."
    ),
    27: (
        "Assess detection and handling separately. A detection method or fence formula shows how "
        "to flag an observation. Credit handling only for a stated action on a flagged value. "
        "If the student only explains detection, cite the course treatment for the handling "
        "deduction and explain what was missing. Copy short contiguous quotes without ellipses."
    ),
    23: (
        "For 2 points the response must connect an outcome to the predictors used to model it. "
        "Accept response/explanatory variables or equivalent language. For 1 point accept an explicit "
        "relationship among variables without their roles, or predicting a continuous outcome without "
        "explaining the predictors. Merely naming regression variants or a different statistical task "
        "does not establish either idea. Review an ambiguous reading that could change the credit."
    ),
    28: (
        "For purpose, cleaning and transforming are alternative ways of preparing raw data. "
        "A passage describing either operation and its purpose in analysis or modelling fully supports "
        "an equivalent definition. For instance, transforming raw features into a representation suitable "
        "for estimators establishes preparation for modelling. Do not demand a second quote for cleaning "
        "when transformation already establishes the credited definition. A mere operation name, a title, "
        "or collecting data alone does not establish this purpose."
    ),
    30: (
        "For method and example, cite body text teaching the rule that numerical values are grouped "
        "into discrete bins, intervals or categories. For the example point, apply that rule to the "
        "student's illustration: verify a numeric quantity or set of values and a meaningful grouping "
        "or boundary. Explicit numeric values can stand for an unnamed variable. Check assignments "
        "and boundaries for consistency. The source need not contain the same values, units, or bins. "
        "The same substantive rule quote can fully support both criterion IDs. A method alone earns no "
        "example point. Do not approve an invented, contradictory or ambiguous assignment as a valid "
        "example. Without an excerpt teaching the binning rule, an apparently correct example alone "
        "cannot support a proposal."
    ),
}

_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_INTERVAL = rf"({_NUMBER})\s*(?:[-–—]|\bto\b)\s*({_NUMBER})"


def states_binning_rule(quote):
    """Conservative fallback for reusing a verified method quote as example evidence."""
    return (re.search(r"\b(?:group|divid|partition|assign|convert|split|map|discreti[sz])\w*\b", quote, re.I)
            and re.search(r"\b(?:numer\w*|continuous|values|measurements)\b", quote, re.I)
            and re.search(r"\b(?:bins?|buckets?|intervals?|categor(?:y|ies)|ranges?)\b", quote, re.I)
            and not re.search(r"\b(?:no|not|never|cannot|without)\b", quote, re.I))


def check_interval_example(answer):
    """Check an explicit list mapped to closed numeric ranges. Leave other notation to the assessor."""
    example = re.split(r"\b(?:for example|example|e\.g\.)\s*[:,]?\s*|[.!?]\s+", answer, flags=re.I)[-1]
    parts = re.split(r"\s*(?:->|→|\b(?:grouped|binned|split) into\b)\s*", example.rstrip().removesuffix("."), flags=re.I)
    if len(parts) != 2:
        return None
    values = [v.strip() for v in parts[0].split(",")]
    ranges = [re.fullmatch(_INTERVAL, part.strip(), re.I) for part in parts[1].split(",")]
    if not all(re.fullmatch(_NUMBER, v) for v in values) or not all(ranges):
        return None
    bounds = sorted((Decimal(r[1]), Decimal(r[2])) for r in ranges)
    data = [Decimal(v) for v in values]
    if len(set(data)) < 2 or len(bounds) < 2 or any(low > high for low, high in bounds):
        return False
    if any(right >= next_left for (_, right), (next_left, _) in zip(bounds, bounds[1:])):
        return False
    return (all(any(lo <= value <= hi for lo, hi in bounds) for value in data)
            and sum(any(lo <= value <= hi for value in data) for lo, hi in bounds) >= 2)


def supports_criterion(question, criterion, quote):
    # A distance fact alone cannot establish how a nearest-neighbor method selects examples.
    if (question, criterion) == (29, "distance"):
        neighbor = re.search(r"\b(?:k[\s-]*nn|neighbou?rs?|closest (?:points|examples|samples)|nearest (?:points|examples|samples))\b", quote, re.I)
        body = re.sub(r"\bk[\s-]*nearest[\s-]*neighbou?rs?\b", "", quote, flags=re.I)
        proximity = re.search(r"\b(?:distances?|metrics?|proximity|closest|nearest|euclidean|manhattan)\b", body, re.I)
        return bool(neighbor and proximity)
    return True
# Each criterion lists its maximum and the standard for full credit.
CRITERIA = {
    21: {"definition": (2, "Learns from labelled examples or paired inputs and known outputs. Either wording suffices. No example, application or prediction step is required.")},
    22: {"diagnosis": (1, "Identifies overfitting."), "evidence": (1, "Connects strong training and weak test performance to poor generalization.")},
    23: {"relationship": (2, "Explains estimating a numeric outcome from predictors or modelling a response's relationship to explanatory variables. Naming continuous output alone is partial understanding.")},
    24: {"first": (1, "First distinct, valid ML challenge plus what makes it a challenge. A name alone earns 0.5."), "second": (1, "Another distinct ML challenge with explanation. Accept valid alternatives to the sample key.")},
    25: {"first": (1, "One concrete real-world use of data science. A short recognizable task suffices."), "second": (1, "A different concrete application. Examples in the key are illustrative.")},
    26: {"structured": (1, "Identifies a fixed schema or regular fields, rows and columns."), "semi_structured": (1, "Identifies flexible records organised by tags, keys or markers. Accept accurate examples as supporting detail.")},
    27: {"detection": (1, "Describes a valid way to detect outliers, such as distance from the mean, IQR fences or a plot."), "handling": (1, "Describes a valid treatment, such as trimming, capping, replacing, or justified retention.")},
    28: {"purpose": (2, "Preparing raw data for analysis or modelling by cleaning or transforming it. A clear definition or equivalent description suffices.")},
    29: {"distance": (1, "Recognizes that K-NN chooses neighbors using distances."), "scale": (1, "Explains that unequal feature ranges skew those distances, and scaling makes contributions comparable. Faster arithmetic alone is insufficient.")},
    30: {"method": (1, "Groups continuous numeric values into discrete intervals or categories."), "example": (1, "Gives a concrete illustration with a variable and a grouping or boundary.")},
    31: {"clusters": (2, "K is the number of clusters or groups. This statement alone earns 2. Neighborhood size or votes describe another algorithm.")},
    32: {"examples": (2, "Three or more distinct valid hyperparameters earn 2. Two earn 1.5 and zero earn 0 when the count is clear. For one recognized setting, name it in feedback and refer the mark to the teacher. If treating two terms as duplicates could change the mark, explain any ambiguity and refer it to the teacher. Refer other unresolved wording that could affect the mark to the teacher. Accept equivalents. A model name alone is not a setting.")},
    33: {"objective": (1, "Identifies iterative minimization of loss or an objective."), "update": (1, "Describes changing parameters opposite the gradient, or equivalently moving downhill using the slope.")},
    34: {"classification": (1, "Correctly compares the depicted boundaries and classification errors."), "margin": (1, "Explains which valid separator maximizes the margin in the actual figure. Requires figure context.")},
    35: {"similarity": (1, "Describes distance-based similarity or an implicit higher-dimensional feature mapping."), "boundary": (1, "Connects it to nonlinear separation in input space or linear separation in the transformed space.")},
}
