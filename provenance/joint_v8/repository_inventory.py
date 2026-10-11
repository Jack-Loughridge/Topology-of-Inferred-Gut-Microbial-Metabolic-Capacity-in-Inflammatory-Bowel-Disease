#!/usr/bin/env python3
"""Read-only source inventory for release-gap review. Never reads observations."""
from pathlib import Path
import hashlib,json,re,subprocess
ROOT=Path(__file__).resolve().parents[2]

def main():
    paths=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and
           '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.json','.yml','.yaml','.sh','.csv','.tex')]
    categories={
      'h0_descriptive_names':r'entropy|death.*dens|persistent.*summ',
      'later_representation_audit_names':r'represent.*sens|reverse|validity|direction.*audit|post.?hoc|census',
      'all_task_external_followup_names':r'followup|follow_up|frozen_external|run_external',
      'canonical_interpretation':r'^analysis/interpretation/',
      'locked_run_configs':r'^provenance/locked_run_configs/',
      'canonical_external':r'^external_validation/.*(?:\.py|\.json)$',
    }
    out={'source_baseline':'effc7bf0661cbf36dda4ed82c44a54a455ad60a4',
         'inventory_scope':'tracked-style source/configuration paths, excluding observations and .git',
         'path_count':len(paths), 'categories':{k:[x for x in sorted(paths) if re.search(v,x,re.I)] for k,v in categories.items()}}
    needles={
      'entropy_code':r'normalized_entropy|normalised_entropy|persistent_entropy|histogram.*200',
      'ricci_path_settings':r'n_paths|epsilon_dist|c_single_out',
      'ricci_availability':r'isfinite|notna|finite_mask',
    }
    targets={
      'entropy_code': ['analysis/graph_diagnostics','pipeline/topology'],
      'ricci_path_settings': ['provenance/locked_run_configs','pipeline/topology/compute_ricci_faithful_pairwise_active.py'],
      'ricci_availability':['analysis/ricci','external_validation/serrano_gomez/scripts/serrano_gomez_external_structural_frozen_ibdmdb_v1'],
    }
    out['source_hits']={}
    for k,rx in needles.items():
        hits=[]
        for rel in sorted(paths):
            if not any(rel.startswith(x) for x in targets[k]) or Path(rel).suffix not in ('.py','.json'):continue
            for i,line in enumerate((ROOT/rel).read_text(errors='replace').splitlines(),1):
                if re.search(rx,line,re.I):hits.append({'path':rel,'line':i,'text':line[:250]})
        out['source_hits'][k]=hits
    dest=ROOT/'provenance/joint_v8/REPOSITORY_INVENTORY.json'
    dest.write_text(json.dumps(out,indent=2)+'\n')
    md=['# Source inventory for compatibility review','', 'This is a path and source-text inventory, not proof of output provenance.','']
    for k,v in out['categories'].items():
        md+=['## '+k,'']+[f'- `{x}`' for x in v]+(['No matching source path in this checkout.'] if not v else [])+['']
    for k,v in out['source_hits'].items():
        md+=['## '+k,'']+[f'- `{x["path"]}:{x["line"]}`: `{x["text"].strip()}`' for x in v]+['']
    (dest.with_suffix('.md')).write_text('\n'.join(md))
    print(json.dumps({k:len(v) for k,v in out['categories'].items()},indent=2))
if __name__=='__main__':main()
