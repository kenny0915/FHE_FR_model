"""Summarize completed finite/ROC evaluation without accepting sanitized TAR."""
import argparse,json
from pathlib import Path
from controlled_degree2.polish_abcd import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    ms=json.loads((root/'ms1mv3/manifest.json').read_text())
    audit=json.loads((root/'ijbc/finite_audit.json').read_text())
    raw=json.loads((root/'ijbc/polish_abcd/ijbc_tar_at_far_raw.json').read_text())
    point=raw[0]['points']['0.0001']['at_or_below_requested_far']
    full=(audit.get('source_images')==469375 and audit.get('augmented_embeddings')==938750
          and audit.get('audited_input_rows')==938750 and audit.get('audited_output_rows')==938750)
    finite=audit['embedding_nonfinite_rows']==0 and audit['nonfinite_values']==0
    if not full:raise ValueError('incomplete IJB-C coverage')
    write(root/'summary.json',dict(ms1mv3_nonfinite_orientations=len(ms['output_nonfinite']),
        ijbc_embedding_nonfinite_rows=audit['embedding_nonfinite_rows'],
        ijbc_nonfinite_values=audit['nonfinite_values'],ijbc_finite=finite,
        valid_tar_at_far_1e4=point if finite else None,
        diagnostic_tar_at_far_1e4=point,interpretation='Fixed final checkpoint evaluation; no IJB training/selection. Nonfinite result never qualifies.'))
if __name__=='__main__':main()
