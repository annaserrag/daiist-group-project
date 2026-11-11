import tempfile
import unittest
from pathlib import Path
from shutil import copyfile
from exam_grader.annotations import SHEET,decision,read,save
from exam_grader.metrics import retrieval
from exam_grader.worksheet import BLOCKS,GUIDE,PAGES,passages

class WorksheetTest(unittest.TestCase):
    def test_save_preserves_other_rows_and_rejects_stale_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"sources.csv"
            copyfile(SHEET,path)
            before=read(path)
            prev=decision(before[33])
            save(34,"missing_context",[],"Comparison figure unavailable in this fixture.",prev,path)
            self.assertEqual(decision(read(path)[33])[0],"missing_context")
            with self.assertRaisesRegex(ValueError,"changed"):
                save(34,"supported",["svm"],"",prev,path)
            self.assertEqual(decision(read(path)[20]),decision(before[20]))
            with self.assertRaisesRegex(ValueError,"registered"):
                save(34,"supported",["invented"],"",decision(read(path)[33]),path)
    def test_partial_metrics_separate_unreviewed_and_gaps(self):
        rows=[r | {"acceptable_sources":"","review_status":"open","note":""} for r in read()]
        rows[20].update(acceptable_sources="supervised",review_status="supported")
        rows[17].update(review_status="gap",note="No matrix layout.")
        rows[33].update(review_status="missing_context",note="Figure absent.")
        with self.assertRaisesRegex(ValueError,"32 questions still open"):
            retrieval(rows,False)
        result=retrieval(rows,False,True)
        self.assertEqual((result["reviewed"],result["supported"],result["corpus_gaps"]),(3,1,[18]))
        self.assertEqual(result["missing_exam_context_excluded"],[34])
        self.assertEqual(result["section"]["hit@5_including_reviewed_gaps"],.5)
        self.assertEqual(result["paragraph"]["hit@5"],1)
    def test_reading_guide_covers_exam_without_changing_labels(self):
        self.assertEqual(set(map(int,GUIDE)),set(BLOCKS))
        self.assertTrue(all(src in PAGES for src,_ in GUIDE.values()))
        self.assertIn("B. Little or no autocorrelation",BLOCKS[1])
        self.assertIn("proline",passages("scales","proline"))
        self.assertNotIn("proline",passages("hamming","proline"))
