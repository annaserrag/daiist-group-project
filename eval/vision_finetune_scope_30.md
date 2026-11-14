# Wrong-exam detection on the 30 integration PDFs

The three unrelated documents now receive `out_of_scope`, zero proposed marks and no exam total. All 26 inputs expected in scope stayed in scope. The obscured exam c27 stayed unreadable. All 21 handwritten inputs from eight source papers retained their previous status. The packet includes six additional constructed exam controls (c22–c27) and three constructed unrelated documents.

| Document | Evidence on the page | Previous status | Current status |
|---|---|---|---|
| c28 | Library stock count with shelf and volume counts | unreadable | out_of_scope |
| c29 | Equipment reservation with room, equipment and collection time | unreadable | out_of_scope |
| c30 | Geography worksheet with a river and an ocean question | in_scope | out_of_scope |

**How the change works.** A short, confidently transcribed first line must contain an exam or administrative heading cue. A further pass of the same local vision model must agree with the type of heading from the whole page at a generated-letter score of at least 0.90. These scores are routing thresholds. Every page must independently meet the conditions before the reader rejects the PDF. A confirmed rejection carries no exam ID or gradeable answers. Uncertain or missing evidence leaves the existing reading and review logic in charge. No case IDs, hashes, answer keys or teacher marks enter the scope decision. Unlabelled different exams, mixed-document uploads and similar data-science exams can still escape detection. The three negative controls were used during development.

**Paired run.** The previous bridge was taken from `cae04ee` and the new bridge was `bridge-v6`. The PDF hashes were checked against the manifest. All 30 PDFs went through the local page reader for each arm: first to capture the unchanged reader outputs for the old adapter, then through the normal integrated PDF entry point for the new one. The grader, policy and question catalogue were shared. Each arm used 11 exact-match cached model responses, zero live grading calls and had zero recorded execution issues. Case pass requires the original marks and referrals plus document and reader-structure checks.

**Results.** Whole-case passes rose from 15/30 to 18/30. Only c28–c30 changed outcome. All 945 question slots on c01–c27 retained their prior mark, reason and execution status. Both arms issued 60 proposals overall; 59 had numeric references and all 59 matched. One written proposal lacked a published reference.

| PDF | Expected status | Previous status | Current status | Whole-case pass before → after | Current proposed marks |
|---|---|---|---|---|---:|
| c01 | in_scope | in_scope | in_scope | Yes → Yes | 5 |
| c02 | in_scope | in_scope | in_scope | Yes → Yes | 5 |
| c03 | in_scope | in_scope | in_scope | Yes → Yes | 6 |
| c04 | in_scope | in_scope | in_scope | Yes → Yes | 4 |
| c05 | in_scope | in_scope | in_scope | No → No | 0 |
| c06 | in_scope | in_scope | in_scope | Yes → Yes | 4 |
| c07 | in_scope | in_scope | in_scope | Yes → Yes | 1 |
| c08 | in_scope | in_scope | in_scope | Yes → Yes | 1 |
| c09 | in_scope | in_scope | in_scope | No → No | 0 |
| c10 | in_scope | in_scope | in_scope | No → No | 0 |
| c11 | in_scope | in_scope | in_scope | Yes → Yes | 1 |
| c12 | in_scope | in_scope | in_scope | No → No | 1 |
| c13 | in_scope | in_scope | in_scope | No → No | 0 |
| c14 | in_scope | in_scope | in_scope | Yes → Yes | 5 |
| c15 | in_scope | in_scope | in_scope | No → No | 3 |
| c16 | in_scope | in_scope | in_scope | No → No | 21 |
| c17 | in_scope | in_scope | in_scope | No → No | 0 |
| c18 | in_scope | in_scope | in_scope | No → No | 0 |
| c19 | in_scope | in_scope | in_scope | No → No | 0 |
| c20 | in_scope | in_scope | in_scope | No → No | 0 |
| c21 | in_scope | in_scope | in_scope | Yes → Yes | 0 |
| c22 | in_scope | in_scope | in_scope | Yes → Yes | 1 |
| c23 | in_scope | in_scope | in_scope | Yes → Yes | 1 |
| c24 | in_scope | in_scope | in_scope | Yes → Yes | 1 |
| c25 | in_scope | in_scope | in_scope | No → No | 0 |
| c26 | in_scope | in_scope | in_scope | Yes → Yes | 0 |
| c27 | unreadable | unreadable | unreadable | Yes → Yes | 0 |
| c28 | out_of_scope | unreadable | out_of_scope | No → Yes | 0 |
| c29 | out_of_scope | unreadable | out_of_scope | No → Yes | 0 |
| c30 | out_of_scope | in_scope | out_of_scope | No → Yes | 0 |

[PDF manifest and references](case_references.json) · [Packet and reproduction instructions](README.md#run-with-annas-reader)
