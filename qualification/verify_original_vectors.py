"""Re-encode the genuine invented target producer's sealed vectors without refitting."""
import json
from pathlib import Path
from unittest.mock import patch

from analysis.alpha_matched import run_fold as core, vector_storage as vs
from analysis.alpha_matched.artifacts import Store
from analysis.alpha_matched.partitions import canonical_json
from qualification.integrated_storage import OUT, TARGET, digest, guard, read, save, usage


def verify():
    guard()
    root = OUT/'reference'
    ref = read(OUT/'evidence/reference.json')['result']
    header = read(root/'reviews/test.json')
    # The reference worker used stock release() against this exact git archive.
    assert digest((root/'reviews/test.json').read_bytes()) == ref['release']
    source = header['source_sha256']
    assert digest(canonical_json(source)) == ref['source']
    for name, checksum in source.items():
        assert digest((OUT/'reference-source/analysis/alpha_matched'/name).read_bytes()) == checksum
    cfg, splits, meta, graph, _ = core.read_inputs(root, header)
    src = Store(root/'jobs/development-final-refits'/ref['release']/core.FINAL)
    marker = src.root/'receipt.json'
    assert digest(marker.read_bytes()) == ref['marker']
    receipt = vs.record(src,read(marker)['record'])
    approved = set(receipt['objects'])
    dest = Store(OUT/'original-vector-replay')
    proc = core.Procedure(meta,splits,None,{},[],None)
    by_scope = {}
    for arm, result in receipt['arms'].items():
        for fitkey in result['fits']:
            assert fitkey in approved
            fit = vs.record(src,fitkey)
            for block in fit['blocks']:
                scope = arm, fit['split'], block['model']
                assert by_scope.setdefault(scope,block) == block
    parents = {}
    counters = {}
    identities = []
    for (arm,node,model), dense in sorted(by_scope.items(), key=lambda item: vs.record(src,item[0][2])['name']=='K0'):
        guard()
        desc = vs.record(src,model)
        kind = desc['name']
        train, valid = proc.frames(node)
        frames = dict(train=train, validate=valid)
        deps = vs.RecipeStore(src.root).closure(model if kind in ('Ricci','K0') else dense)
        assert set(dense.values()) | deps <= approved
        for name in deps - dest.objects:
            src.read(name)
            dest.publish(name,(src.root/name).read_bytes())
            dest.objects.add(name)
        if kind == 'Ricci':
            factory = core.Factory(graph,arm,src.root/'geometry',cfg['tolerances'])
            # This is a reader for existing source-pinned shards, never a new producer.
            factory.version = ref['source']
            before = {side:src.load_vector(dense[side]) for side in frames}
            compact = vs.capture_block(dest,model,before,frames,factory)
            vs.verify_evidence(dest,compact,train,valid,arm,factory=factory)
            parents[arm,node,model] = compact
        elif kind == 'K0':
            compact = vs.projected_block(dest,parents[arm,node,desc['source']],model,len(graph.axis))
            vs.verify_evidence(dest,compact,train,valid,arm)
        else:
            compact = dense
        for side in frames:
            a,b = src.load_vector(dense[side]),dest.load_vector(compact[side])
            assert a.values.dtype==b.values.dtype and a.values.shape==b.values.shape
            assert a.values.tobytes()==b.values.tobytes() and a.identity()==b.identity()
            assert (a.ids,a.features,a.generator,a.partition,a.arm,a.block)==(b.ids,b.features,b.generator,b.partition,b.arm,b.block)
            counters[kind] = counters.get(kind,0)+1
            identities.append([arm,node,model,side,a.identity()])
    assert len(identities)==len(ref['vectors'])
    assert digest(marker.read_bytes())==ref['marker']
    return dict(target=TARGET,source=ref['source'],marker=ref['marker'],exact_vectors=len(identities),
                by_kind=counters,ordered_digest=digest(canonical_json(sorted(identities))),allocated=usage(dest.root))


if __name__ == '__main__':
    with patch('analysis.alpha_matched.features.compute',side_effect=AssertionError('Geometry solve during source replay')):
        result=verify()
    save(OUT/'original-replay.json',result)
    print(json.dumps(result,sort_keys=True))
