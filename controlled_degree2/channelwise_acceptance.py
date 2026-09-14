"""Strict acceptance of a completed full IJB-C calibration-set evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def assess(raw, audit, certificate, checkpoint_sha256):
    points = raw[0]['points'] if len(raw) == 1 else {}
    point = points.get('0.0001', {})
    tar = point.get('tar_percent', float('nan'))
    far = point.get('actual_far', float('nan'))
    checks = dict(
        tar=math.isfinite(tar) and 96 <= tar <= 100,
        requested_roc_point='0.0001' in points and math.isfinite(far) and 0 < far < .001,
        full_ijbc=audit.get('target') == 'IJBC' and audit.get('source_images') == 469375
                  and audit.get('augmented_embeddings') == 938750
                  and audit.get('audited_input_rows') == 938750
                  and audit.get('audited_output_rows') == 938750,
        finite=audit.get('nonfinite_values') == 0 and audit.get('embedding_nonfinite_rows') == 0,
        graph=certificate.get('pure_polynomial') is True
              and certificate.get('inference_clipping') is False
              and certificate.get('quadratic_sites') == 25
              and certificate.get('coefficient_mode') == 'channelwise'
              and certificate.get('coefficients') == 17664,
        checkpoint=certificate.get('checkpoint_sha256') == checkpoint_sha256,
    )
    return dict(target_met=all(checks.values()), checks=checks,
                tar_percent=tar if math.isfinite(tar) else None,
                actual_far=far if math.isfinite(far) else None,
                requested_far=.0001, checkpoint_sha256=checkpoint_sha256,
                interpretation='IJB-C calibration-set performance; not untouched test accuracy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'audit', 'certificate', 'checkpoint', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    h = hashlib.sha256()
    with open(args.checkpoint, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    report = assess(*[json.loads(Path(path).read_text()) for path in
                      (args.raw, args.audit, args.certificate)], h.hexdigest())
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report['target_met']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
