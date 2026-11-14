# Paired ML evidence prompt comparison

Ran `gemini-3.1-flash-lite` with grader commit `5355a15` on the same 15 readable answers. Both arms used fresh caches, the same course material, a 0.40 retrieval threshold and teacher review for proposals of 1/2 or below.

| Grader | Proposals | Teacher referrals (including failures) | Exact among comparable proposals | Off by ≥1 point | Operational failures |
|---|---:|---:|---:|---:|---:|
| Current | 4/15 | 11/15 | 4/4 (100%) | 0 | 2 |
| New | 3/15 | 12/15 | 3/3 (100%) | 0 | 1 |

Left review: None.
Entered review: Paper 29 Q26.

| Paper | Question | Current | New | Teacher | Change |
|---:|---:|---:|---:|---:|---|
| 30 | Q24 | Review | Review | 2/2 | Both review |
| 30 | Q25 | Review | Review | 0/2 | Both review |
| 30 | Q26 | Review | Model error | 2/2 | Both need a teacher |
| 30 | Q27 | Review | Review | 2/2 | Both review |
| 30 | Q30 | Model error | Review | 1/2 | Both need a teacher |
| 44 | Q24 | Model error | Review | 2/2 | Both need a teacher |
| 44 | Q25 | 2/2 | 2/2 | 2/2 | Same mark |
| 44 | Q26 | 2/2 | 2/2 | 2/2 | Same mark |
| 44 | Q27 | Review | Review | 2/2 | Both review |
| 44 | Q30 | Review | Review | 1/2 | Both review |
| 29 | Q24 | Review | Review | 2/2 | Both review |
| 29 | Q25 | 2/2 | 2/2 | 2/2 | Same mark |
| 29 | Q26 | 2/2 | Review | 2/2 | Enters review |
| 29 | Q27 | Review | Review | 2/2 | Both review |
| 29 | Q30 | Review | Review | 1/2 | Both review |

**Recommendation.** Retain the current grader. On the 12 answers where both graders completed, the new prompt lost one correct 2/2 on paper 29 Q26 and released no answers from review. Three other answers had an invalid model response in one grader after retries. All issued marks were 2/2, so this sample gives no direct evidence of accuracy on partial-credit proposals.

The adoption rule required more proposals, at least as many exact matches and no increase in errors of one point or more. Agreement uses only issued marks with a numeric teacher reference. A referral has no implied mark. This sample covers five questions across three papers.

[Anonymized per-answer decisions](paired_evidence_reset.json)
