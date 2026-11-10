"""Replay existing tracks with zone rules; no model loading or inference."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from intrusion_engine import IntrusionEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--tracking-run', required=True, help='Existing run with tracks.csv and summary.json')
    parser.add_argument('--config', default='configs/intrusion_demo.json')
    parser.add_argument('--mode', choices=['schedule', 'armed', 'disarmed'], default='schedule')
    parser.add_argument('--show', action='store_true', help='A toggles armed, S resumes schedule, Q stops')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--output-root', default='outputs')
    parser.add_argument('--run-name')
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error('--max-frames must be positive')
    source = Path(args.source).resolve()
    tracking_run = Path(args.tracking_run).resolve()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    summary = json.loads((tracking_run / 'summary.json').read_text(encoding='utf-8'))
    if source != Path(summary['source']).resolve():
        parser.error('Source must match the source recorded in the tracking run')
    zone = config['zone_normalized']
    if len(zone) < 3 or any(len(p) != 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in p) for p in zone):
        parser.error('zone_normalized must contain at least three 2D points in [0, 1]')
    schedule = config['armed_schedule']
    if not schedule or schedule[0]['time_seconds'] != 0:
        parser.error('Schedule must start at zero')
    last = -1
    for item in schedule:
        t = item['time_seconds']
        if not math.isfinite(t) or t <= last or type(item['armed']) is not bool:
            parser.error('Schedule needs increasing finite times and boolean armed values')
        last = t
    rows_by_frame = defaultdict(list)
    tracks_path = tracking_run / 'tracks.csv'
    with tracks_path.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            index = int(row['frame_index'])
            if not 0 <= index < summary['frames_processed']:
                parser.error('Track frame index is outside tracking summary')
            rows_by_frame[index].append(row)
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
        cap.release()
        parser.error('Cannot open source video/FPS')
    if abs(fps - summary['source_fps']) > 1e-6 or int(cap.get(7)) != summary['frames_processed']:
        cap.release()
        parser.error('Replay requires a complete tracking run with matching FPS and frame count')
    polygon = [[x * (width - 1), y * (height - 1)] for x, y in zone]
    engine = IntrusionEngine(polygon, config['confirm_seconds'])
    name = args.run_name or datetime.now().strftime('intrusion_%Y%m%d_%H%M%S_%f')
    output_root = Path(args.output_root).resolve()
    output = (output_root / name).resolve()
    if output.parent != output_root:
        cap.release()
        parser.error('--run-name must be a single directory name')
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        cap.release()
        parser.error('Output exists; choose a new run name')
    writer = cv2.VideoWriter(str(output / 'intrusion_demo.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise SystemExit('Cannot create output video')
    frame_index, schedule_index, event_count = 0, 0, 0
    override = None if args.mode == 'schedule' else args.mode == 'armed'
    stopped = False
    try:
        with (output / 'events.csv').open('w', encoding='utf-8', newline='') as ef, \
             (output / 'states.csv').open('w', encoding='utf-8', newline='') as sf:
            events_writer = csv.DictWriter(ef, fieldnames=['event_id', 'frame_index', 'time_seconds', 'track_id', 'event_type', 'zone_entered_seconds', 'observed_duration_seconds', 'snapshot'])
            state_writer = csv.DictWriter(sf, fieldnames=['frame_index', 'time_seconds', 'armed', 'control', 'pending_ids', 'alert_ids', 'new_events'])
            events_writer.writeheader()
            state_writer.writeheader()
            while args.max_frames is None or frame_index < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                t = frame_index / fps
                while schedule_index + 1 < len(schedule) and schedule[schedule_index + 1]['time_seconds'] <= t:
                    schedule_index += 1
                armed = schedule[schedule_index]['armed'] if override is None else override
                rows = rows_by_frame.get(frame_index, [])
                states, events = engine.update(frame_index, t, rows, armed)
                cv2.polylines(frame, [np.array(polygon, dtype=np.int32)], True, (255, 180, 0), 2)
                for row in rows:
                    track_id = int(row['track_id'])
                    x1, y1, x2, y2 = [int(float(row[k])) for k in ['x1', 'y1', 'x2', 'y2']]
                    state = states[track_id]
                    color = (0, 0, 255) if state == 'ALERT' else (0, 220, 255) if state == 'PENDING' else (0, 200, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, ((x1 + x2) // 2, y2), 4, color, -1)
                    cv2.putText(frame, f'ID {track_id}: {state}', (x1, max(55, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
                event_count += len(events)
                label = f'{t:.2f}s | {"ARMED" if armed else "DISARMED"} | events {event_count}'
                cv2.putText(frame, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 255), 2)
                cv2.putText(frame, 'A: toggle | S: schedule | Q: stop', (12, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                for event in events:
                    snapshot = f'event_{event["event_id"]:04d}.jpg'
                    if not cv2.imwrite(str(output / snapshot), frame):
                        raise RuntimeError('Cannot save event snapshot')
                    events_writer.writerow({**event, 'snapshot': snapshot})
                state_writer.writerow({'frame_index': frame_index, 'time_seconds': round(t, 6), 'armed': armed,
                                       'control': 'schedule' if override is None else 'override',
                                       'pending_ids': ';'.join(str(k) for k, v in states.items() if v == 'PENDING'),
                                       'alert_ids': ';'.join(str(k) for k, v in states.items() if v == 'ALERT'), 'new_events': len(events)})
                writer.write(frame)
                frame_index += 1
                if args.show:
                    cv2.imshow('Intrusion demo', frame)
                    key = cv2.waitKey(max(1, round(1000 / fps))) & 0xff
                    if key == ord('a'):
                        override = not armed
                    elif key == ord('s'):
                        override = None
                    elif key == ord('q'):
                        stopped = True
                        break
    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source': str(source),
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'tracks_sha256': hashlib.sha256(tracks_path.read_bytes()).hexdigest(),
              'tracking_run': str(tracking_run), 'config': config, 'mode': args.mode,
              'source_fps': fps, 'frames_processed': frame_index, 'events': event_count,
              'complete': frame_index == summary['frames_processed'], 'stopped_by_user': stopped,
              'note': 'Replay of cached tracks, not live inference. Events are rule outputs, not ground truth. Missing observations reset confirmation. ID fragmentation can cause missed or repeated alerts.'}
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Frames: {frame_index}; events: {event_count}; output: {output}')


if __name__ == '__main__':
    main()
