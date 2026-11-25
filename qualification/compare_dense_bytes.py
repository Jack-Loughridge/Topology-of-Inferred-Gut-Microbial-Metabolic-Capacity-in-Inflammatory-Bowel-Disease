"""Direct byte comparisons on three independently produced, sealed invented jobs."""
from analysis.alpha_matched.artifacts import Store
from analysis.alpha_matched.partitions import canonical_json, fusion_input
from qualification.integrated_storage import OUT, digest, guard, read, save
from analysis.alpha_matched.vector_storage import record


def bindings(name):
    result = read(OUT/'evidence'/(name+'.json'))['result']
    store = Store(OUT/name/'jobs/development-final-refits'/result['release']/'final_refit')
    marker = store.root/'receipt.json'
    assert digest(marker.read_bytes()) == result['marker']
    body = record(store,read(marker)['record'])
    vectors, joints = {}, {}
    for arm, ledger in body['arms'].items():
        for ref in set(ledger['fits']):
            fit = record(store,ref)
            for block in fit['blocks']:
                model = record(store,block['model'])
                source = record(store,model['source']) if model['name']=='K0' else model
                kind = (model['name'],canonical_json(source['state'].get('setting',{})))
                for side in ('train','validate'):
                    key=(arm,fit['split'],*kind,side)
                    assert vectors.setdefault(key,block[side])==block[side]
            if fit['name']=='joint':
                for side in ('train','validate'):
                    key=(arm,fit['split'],side)
                    refs=[block[side] for block in fit['blocks']]
                    assert joints.setdefault(key,refs)==refs
    return store,vectors,joints


def compare():
    stores = [bindings(name) for name in ('reference','dense','clean')]
    for i in (1,2):
        assert all(stores[0][i].keys()==s[i].keys() for s in stores)
    total = 0
    for key in stores[0][1]:
        guard()
        a,b,c = [s.load_vector(v[key]) for s,v,_ in stores]
        for v in (a,b):
            assert (v.values.dtype,v.values.shape,v.values.tobytes(),v.ids,v.features,v.partition,v.arm,v.block)==(
                c.values.dtype,c.values.shape,c.values.tobytes(),c.ids,c.features,c.partition,c.arm,c.block)
        assert b.identity()==c.identity()
        total+=c.values.nbytes
    joined = 0
    for key in stores[0][2]:
        arrays = [fusion_input(*(s.load_vector(ref) for ref in j[key])) for s,_,j in stores]
        assert all((a.dtype,a.shape,a.tobytes())==(arrays[0].dtype,arrays[0].shape,arrays[0].tobytes()) for a in arrays)
        joined+=arrays[0].nbytes
    return dict(split_vectors=len(stores[0][1]),logical_vector_bytes=total,joint_splits=len(stores[0][2]),
                logical_fusion_bytes=joined)


if __name__=='__main__':
    guard()
    result=compare()
    save(OUT/'memcmp.json',result)
    print(result)
