"""Build a complete private candidate from genuine invented final fits without reading outcomes."""
import json
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import uuid
import unittest

from . import comparator_run as driver, report_g7
from .assemble_submission import assemble
from .export_final import export
from .test_artifacts import fixture, application_fixture
from .test_comparator_run import register


class AssemblyTest(unittest.TestCase):
    def test_candidate_and_whole_plan_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            status=fixture(root,final=True,masked=True)
            fold=driver.core.FINAL
            primary=driver.core.execute(root,fold)
            cmp=register(root,fold,primary=primary)
            driver.execute(root,fold)
            path=export(root,primary,driver.core.bundle.sha(cmp.root/'receipt.json'))
            staged=root/'private-staging'
            exported=staged/'exports/binary'
            shutil.copytree(path.parent,exported)
            assets={}
            def asset(name,data,role):
                rel='assets/'+name
                f=staged/rel
                f.parent.mkdir(parents=True,exist_ok=True)
                f.write_bytes(data)
                assets.setdefault(role,[]).append(dict(file=rel,sha256=driver.sha(data)))
                return rel
            parent_id=status['final_refit_release']['sha256']
            store=driver.Store(root/'jobs/development-final-refits'/parent_id/fold)
            receipt=driver.read(store,json.loads((store.root/'receipt.json').read_bytes())['record'])
            _,panel,prepared=application_fixture(root,[(store,receipt)])
            fixture_roster=json.loads((prepared/'graph_roster.json').read_bytes())
            config,splits,meta,_,_=driver.core.read_inputs(root,json.loads((root/'reviews/test.json').read_bytes()))
            status['stage']='G5'
            (root/'status.json').write_bytes(driver.canonical_json(status))
            folder=Path(__file__).parent
            source_names=set(driver.source_hashes())|set(report_g7.sources())|{'export_final.py','report_comparators.py'}
            for name in sorted(source_names):
                asset('code/'+name,(folder/name).read_bytes(),'code')
            for name in ['environment','data','prediction_schema']:
                asset(name+'.json',driver.canonical_json({'invented_role':name}),name)
            graph=asset('graph-rules.json',(root/'package/graph-rules.json').read_bytes(),'config')
            ids=['independent-1','independent-2']
            asset('eligibility.json',driver.canonical_json(dict(rosters={'invented-new':{'binary':ids}})),'test_eligibility')
            hashed=lambda x:driver.sha(driver.canonical_json(x))
            trip=[[hashed(k),hashed(row.person),hashed(row.group)] for k,row in meta.iterrows()]
            train=asset('train.json',driver.canonical_json(dict(binary=trip)),'splits')
            reserved_rows=[['invented-new','binary',sid,'external-person-'+str(i),'external-family-'+str(i)] for i,sid in enumerate(ids)]
            identity=driver.canonical_json(reserved_rows)
            identity_file='identity-only/identities.json';(staged/'identity-only').mkdir()
            (staged/identity_file).write_bytes(identity)
            labels=b''.join(driver.canonical_json(dict(zip(report_g7.AXES,row),label=config['classes'][i]))+b'\n'
                            for i,row in enumerate(reserved_rows))
            policy=dict(version=1, implementation=report_g7.sources(), contrasts=report_g7.statistics.PRIMARY_PAIRS,
                        aggregate_disclosure=report_g7.AGGREGATE_DISCLOSURE,bootstrap=dict(draws=2000,seed=20261124),public_min_groups=5,unavailable_panels=[],unmet_methods=[],endpoints={'binary':config['classes']},training_identities=train,
                        panel_roles={'invented-new':{'binary':'primary'}},
                        identities_sha256=driver.sha(identity),outcomes_sha256=driver.sha(labels),
                        procedure_exports={'binary':'exports/binary.json'})
            report=asset('policy.json',driver.canonical_json(policy),'statistical_analysis')
            export_ref=dict(file='exports/binary/export.json',sha256=driver.core.bundle.sha(path))
            spec=dict(version=1,reserved_identities=dict(file=identity_file,sha256=driver.sha(identity)),exports=dict(binary=export_ref),assets=assets,graph_rules=graph,report_policy=report)
            source=dict(review=dict(file='test.json',sha256=parent_id))
            (staged/'development').mkdir()
            parent=json.loads((root/'reviews/test.json').read_bytes())
            for key in ('config','splits','meta'):
                name=parent['inputs'][key]['path']
                source[key]='development/'+name
                shutil.copyfile(root/'manifests/development-fold-inputs'/name,staged/source[key])
            spec['development_sources']={'binary':source}
            body=driver.canonical_json(spec);(staged/'assembly.json').write_bytes(body)
            manifest=assemble(root,staged,driver.sha(body))
            self.assertEqual(manifest,assemble(root,staged,driver.sha(body)))
            sub=json.loads(manifest.read_bytes());sid=driver.core.bundle.sha(manifest)
            self.assertEqual(len(sub['planned_predictions']),42)
            review_file=root/'reviews/fixture-assembled.json'
            thread=str(uuid.uuid4())
            review_file.write_bytes(driver.canonical_json(dict(reviewer='test-reviewer',gate='G6',decision='PASS',
                blocking_findings=[],evaluator_session=dict(role='test-evaluator',thread_id=thread),timestamp='2026-01-01T00:00:00+00:00',resolved_findings=[],
                submission_sha256=sid,reviewed_files=sub['files'])))
            (root/'status.json').write_bytes(driver.canonical_json(dict(status='ACTIVE',stage='G7',
                final_prediction_release=dict(file=review_file.name,sha256=driver.core.bundle.sha(review_file)))))
            fixture_roster['submission_sha256']=sid
            work=manifest.parent.parent
            prediction_dir=work/'predictions';prediction_dir.mkdir()
            (prediction_dir/'graph_roster.json').write_bytes(driver.canonical_json(fixture_roster))
            with patch.object(driver.Store, 'probabilities', autospec=True, side_effect=driver.Store.probabilities) as calls:
                outputs=driver.apply_final(root,manifest.parent,panel,'invented-new','binary',prediction_dir)
            species_calls = [call for call in calls.call_args_list if call.args[1] == 'species']
            self.assertEqual(len(species_calls), len(fixture_roster['cohorts']['invented-new']))
            self.assertEqual(len(outputs),42)
            seal=driver.core.clinical.verify_and_seal(manifest.parent,review_file,'test-reviewer','test-evaluator',prediction_dir)
            self.assertEqual(len(json.loads(seal.read_bytes())['predictions']),42)
            grant=json.loads((root/'status.json').read_bytes())
            grant['g7_report_release']=dict(evaluator='test-evaluator',workspace=str(work),submission_sha256=sid,
                review_sha256=driver.core.bundle.sha(review_file),seal_sha256=driver.core.bundle.sha(seal))
            (root/'status.json').write_bytes(driver.canonical_json(grant))
            (work/'outcomes').mkdir();(work/'outcomes/identities.json').write_bytes(identity)
            (work/'outcomes/outcomes.jsonl').write_bytes(labels)
            with patch.dict(os.environ,{'CODEX_THREAD_ID':thread}):   # invented fixture only
                result=report_g7.generate(root)
            self.assertEqual(json.loads((result/'private/report.json').read_bytes())['status'],'complete')
            self.assertTrue((result/'aggregate/report.json').is_file())
            (root/'status.json').write_bytes(driver.canonical_json(status))
            original_export=(exported/'export.json').read_bytes()
            for kind,message in [('missing-method','original final-refit producer'),('wrong-code','exporter')]:
                tampered=json.loads(original_export)
                if kind=='missing-method':
                    tampered['procedures'].pop()
                else:
                    tampered['validator_sha256']={}
                raw=driver.canonical_json(tampered)
                (exported/'altered.json').write_bytes(raw)
                spec['exports']['binary']['file']='exports/binary/altered.json'
                spec['exports']['binary']['sha256']=driver.sha(raw)
                altered_spec=driver.canonical_json(spec)
                (staged/'assembly.json').write_bytes(altered_spec)
                with self.assertRaisesRegex(ValueError,message):
                    assemble(root,staged,driver.sha(altered_spec))
            spec['exports']['binary']['file']='exports/binary/export.json'
            spec['exports']['binary']['sha256']=driver.sha(original_export)
            original_policy=(staged/report).read_bytes()
            policy['unmet_methods']=[dict(endpoint='binary',arm='RAW',method='K0',reason='invented contradiction')]
            wrong_policy=driver.canonical_json(policy)
            (staged/report).write_bytes(wrong_policy)
            spec['assets']['statistical_analysis'][0]['sha256']=driver.sha(wrong_policy)
            wrong_spec=driver.canonical_json(spec)
            (staged/'assembly.json').write_bytes(wrong_spec)
            with self.assertRaisesRegex(ValueError,'method registry'):
                assemble(root,staged,driver.sha(wrong_spec))
            (staged/report).write_bytes(original_policy)
            spec['assets']['statistical_analysis'][0]['sha256']=driver.sha(original_policy)
            original_train=(staged/train).read_bytes()
            trip[0][2]=hashed('wrong-group')
            altered=driver.canonical_json(dict(binary=trip))
            (staged/train).write_bytes(altered)
            spec['assets']['splits'][0]['sha256']=driver.sha(altered)
            changed=driver.canonical_json(spec)
            (staged/'assembly.json').write_bytes(changed)
            with self.assertRaisesRegex(ValueError,'signed development metadata'):
                assemble(root,staged,driver.sha(changed))
            (staged/'assembly.json').write_bytes(body)
            (staged/train).write_bytes(original_train)
            outside=root/'outside'
            outside.mkdir()
            (manifest.parent/'exports/binary.json').unlink()
            (manifest.parent/'exports').rmdir()
            (manifest.parent/'exports').symlink_to(outside,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'Symlink in frozen output'):
                assemble(root,staged,driver.sha(body))
            self.assertEqual(list(outside.iterdir()),[])
            (staged/graph).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                assemble(root,staged,driver.sha(body))
            (root/'STOP').touch()
            with self.assertRaisesRegex(RuntimeError,'STOP'):
                assemble(root,staged,driver.sha(body))
