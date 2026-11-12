"""Written marking guide derived from the public questions and instructor key."""
import re


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
