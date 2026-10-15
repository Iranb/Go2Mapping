"""Read-only summary of saved field runs; no robot connection or commands."""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def audit(folder):
    runs = []
    for path in sorted(Path(folder).glob('*/result.json')):
        raw = path.read_bytes()
        result = json.loads(raw)
        pauses = Counter(p.get('reason', 'unknown') for p in result.get('pauses', []))
        replans = result.get('replans', [])
        runs.append({'run': path.parent.name, 'source': str(path),
                     'sha256': hashlib.sha256(raw).hexdigest(),
                     'outcome': result.get('outcome'), 'message': result.get('message'),
                     'duration_s': round(result.get('finished_unix', 0)-result.get('started_unix', 0), 3),
                     'final_distance_m': result.get('distance_m'),
                     'nonzero_commands': result.get('nonzero_commands', 0),
                     'pauses': dict(pauses), 'replans': len(replans),
                     'replan_reasons': [r.get('trigger', r.get('reason')) for r in replans]})
    counts = Counter(r['outcome'] for r in runs)
    pauses = Counter()
    for run in runs:
        pauses.update(run['pauses'])
    return {'generated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'run_count': len(runs), 'outcomes': dict(counts), 'pause_reasons': dict(pauses),
            'interpretation': 'Observed logs only. Repeated runs are not independent trials; odometry arrival is not external ground truth.',
            'runs': runs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/go2-live-demo/navigation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = audit(args.folder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in data.items() if k != 'runs'}, ensure_ascii=False, indent=2))
