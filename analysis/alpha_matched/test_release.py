import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from .partitions import canonical_json
from .release import verify_release,verify_and_seal,sha


class ReleaseTest(unittest.TestCase):
    def test_coverage_replay_and_tampering(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            blob=root/'fixture'
            blob.write_text('synthetic')
            groups='code environment data splits config models generators thresholds prediction_schema statistical_analysis'.split()
            ids=['a','b','c']
            roster=hashlib.sha256(canonical_json(ids)).hexdigest()
            plan=[dict(id='fixture'+arm+m,cohort='invented',endpoint='binary',arm=arm,method=m,
                      eligible_row_hash=roster,row_count=3,requires_curvature=m in ('Ricci','joint'),class_order=['neg','pos'],model_sha256=sha(blob),generator_sha256=sha(blob),graph_rules_sha256=sha(blob))
                  for arm in ['RAW','RELATIVE'] for m in ['H0','Ricci','joint','activity']]
            def save(p,body):
                p.write_text(json.dumps(body))
            save(root/'analysis.json',{'predictions':plan})
            save(root/'eligibility.json',{'rosters':{'invented':{'binary':ids}}})
            files={p.name:sha(p) for p in [blob,root/'analysis.json',root/'eligibility.json']}
            sub={'eligible_primary_binary_cohorts':['invented'],'files':files,
                 'roles':{**{k:['fixture'] for k in groups},'comparisons':['analysis.json'],'test_eligibility':['eligibility.json']},
                 'planned_predictions':plan}
            manifest=root/'submission.json'
            review=root/'review.json'
            def approve():
                save(manifest,sub)
                save(review,{'gate':'G6','timestamp':'2026-11-24T12:00:00+00:00','resolved_findings':[],'decision':'PASS','reviewer':'review','blocking_findings':[],
                             'submission_sha256':sha(manifest),'reviewed_files':files})
            sub['planned_predictions']=plan[:-1]
            approve()
            with self.assertRaises(ValueError):
                verify_release(root,review,'review','eval')
            sub['planned_predictions']=plan
            approve()
            sid=sha(manifest)
            pdir=root/'private'
            pdir.mkdir()
            with self.assertRaises(FileNotFoundError):
                verify_and_seal(root,review,'review','eval',pdir)
            edges=np.array([[0,1]])
            np.save(pdir/'edges.npy',edges)
            np.savez(pdir/'geometry.npz',pairs=edges,curvature=np.zeros(1))
            geom=dict(active_pairs='edges.npy',values='geometry.npz',active_pairs_sha256=sha(pdir/'edges.npy'),
                      values_sha256=sha(pdir/'geometry.npz'))
            registry={x:dict(sample=x,profile_sha256=hashlib.sha256(x.encode()).hexdigest(),graph_rules_sha256=sha(blob),
                             active_pairs_sha256=geom['active_pairs_sha256']) for x in ids}
            save(pdir/'graph_roster.json',dict(submission_sha256=sid,evaluator='eval',cohorts={'invented':registry}))
            for item in plan:
                name=item['id']
                artifact=pdir/(name+'.jsonl')
                rows=[dict(sample=x,probabilities=[.2,.8],submission_sha256=sid,prediction_id=name,
                           class_order=item['class_order'],model_sha256=sha(blob),generator_sha256=sha(blob),
                           geometry={**geom,**registry[x],'prediction_id':name,'generator_sha256':sha(blob)}) for x in ids]
                artifact.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                count=len(rows) if item['requires_curvature'] else 0
                rec=dict(submission_sha256=sid,prediction_id=name,eligible_row_hash=roster,row_count=3,
                         artifact=artifact.name,sha256=sha(artifact),expected_edges=count,computed_finite_edges=count,
                         failed_edges=0,schema_validation_passed=True)
                save(pdir/(name+'.json'),rec)
            seal=verify_and_seal(root,review,'review','eval',pdir)
            self.assertEqual(len(json.loads(seal.read_bytes())['predictions']),len(plan))
            for change in [{'computed_finite_edges':-1},{'schema_validation_passed':'true'}]:
                save(pdir/(name+'.json'),{**rec,**change})
                with self.assertRaises(ValueError):
                    verify_and_seal(root,review,'review','eval',pdir)
            rows[-1]['sample']='other'
            artifact.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save(pdir/(name+'.json'),{**rec,'sha256':sha(artifact)})
            with self.assertRaises(ValueError):
                verify_and_seal(root,review,'review','eval',pdir)
            blob.write_text('changed')
            with self.assertRaises(ValueError):
                verify_release(root,review,'review','eval')
            with self.assertRaises(PermissionError):
                verify_release(root,review,'review','review')
