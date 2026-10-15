"""Build a browser replay and comparison report from offline navigation runs."""
import argparse
import json
from pathlib import Path
import numpy as np
from go2_navigation import GroundGrid


def build(capture, before_folder, after_folder, assets, report_folder):
    before = json.loads((before_folder/'simulation-results.json').read_text())
    after = json.loads((after_folder/'simulation-results.json').read_text())
    before_keys = [(r['frame'], r['profile']) for r in before['results']]
    after_keys = [(r['frame'], r['profile']) for r in after['results']]
    if before_keys != after_keys or before['target'] != after['target']:
        raise ValueError('Comparison requires identical frame/profile cases and target')
    for a, b in zip(before['results'], after['results']):
        if (a['response_model'] != b['response_model'] or
                a.get('goal_limit_m', 2.) != b.get('goal_limit_m', 2.)):
            raise ValueError('Comparison requires identical response models')
    completed = [r for r in after['results'] if r['outcome'] == 'complete']
    distances = [r['final_distance_m']*100 for r in completed]
    summary = {'frames': len(set(r['frame'] for r in after['results'])), 'cases': after['cases'],
               'before_completed': before['completed'], 'after_completed': after['completed'],
               'before_violations': before['map_violations'], 'after_violations': after['map_violations'],
               'distance_cm_range': [min(distances), max(distances)] if distances else [0, 0]}
    scans = np.load(capture/'live-frames-centres.npz')
    index = max(int(k.split('_')[1]) for k in scans.files if k.startswith('cloud_'))
    resolution = json.loads((capture/'diagnostics-after.json').read_text())['turn_space']['resolution']
    points = np.rint(scans[f'cloud_{index}']/resolution)*resolution-resolution/2
    grid = GroundGrid(points, scans[f'pose_{index}'], float(scans[f'yaw_{index}']), resolution)
    def coordinates(mask):
        y, x = np.nonzero(mask)
        return np.round((np.column_stack((x, y))+grid.origin)*resolution, 4).tolist()
    payload = {'summary': summary, 'target': after['target'], 'start': grid.start.tolist(), 'start_yaw': grid.yaw,
               'map': {'occupied': coordinates(grid.occupied), 'ground': coordinates(grid.observed)}, 'cases': {}}
    last_frame = max(r['frame'] for r in after['results'])
    for key, folder, run in [('before', before_folder, before), ('after', after_folder, after)]:
        trajectories = np.load(folder/'trajectories.npz')
        for result in run['results']:
            if result['frame'] != last_frame:
                continue
            profile = result['profile']
            trace = trajectories[f'frame_{last_frame}_{profile}'][:, :4].copy()
            trace[:, 0] -= 1000.
            payload['cases'].setdefault(profile, {})[key] = {
                **{k: result[k] for k in ('outcome', 'message', 'elapsed_s', 'final_distance_m', 'plan')},
                'trace': np.round(trace, 5).tolist()}
    assets.mkdir(parents=True, exist_ok=True)
    (assets/'navigation-simulation-data.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    report_folder.mkdir(parents=True, exist_ok=True)
    (report_folder/'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    text = "# Offline navigation comparison\n\n"
    text += f"Target: {after['target']}. Cases: {summary['cases']}.\n\n"
    text += "| Metric | Before | After |\n|---|---:|---:|\n"
    text += f"| Completed | {summary['before_completed']} | {summary['after_completed']} |\n"
    text += f"| Map violations | {summary['before_violations']} | {summary['after_violations']} |\n\n"
    if distances:
        text += f"Successful final distances: {min(distances):.1f}–{max(distances):.1f} cm.\n\n"
    else:
        text += "No completed cases; no successful arrival-distance range.\n\n"
    text += "## Scope\n\n"
    text += ("Each case runs the navigation controller on one fixed lidar frame with an assumed "
             "planar actuator model. Latency, acceleration and deadband are simulated. "
             "It does not simulate new viewpoints, leg dynamics, contact, slip or odometry drift. "
             "Map violations use the planner geometry and are not an independent collision detector. "
             "These results do not estimate physical navigation success rates. No robot is connected.\n\n")
    text += "Generated files: comparison.json, simulation-results.json and trajectories.npz.\n"
    (report_folder/'report.md').write_text(text)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('capture', 'before', 'after', 'assets', 'report'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.capture, args.before, args.after, args.assets, args.report), ensure_ascii=False))
