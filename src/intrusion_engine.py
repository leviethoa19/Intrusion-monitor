"""Zone and temporal rules, independent of detector and video playback."""

import math


def inside_polygon(point, polygon):
    """Ray casting; points on the boundary count as inside."""
    x, y = point
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        ax, ay = a
        bx, by = b
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) < 1e-9 and min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by):
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


class IntrusionEngine:
    """One event per observed visit and ID; no identity inference or gap filling."""

    def __init__(self, polygon, confirm_seconds):
        if len(polygon) < 3 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in polygon):
            raise ValueError("Polygon needs at least three finite 2D points")
        area_twice = sum(a[0] * b[1] - b[0] * a[1]
                         for a, b in zip(polygon, polygon[1:] + polygon[:1]))
        if abs(area_twice) < 1e-9:
            raise ValueError("Polygon must have nonzero area")
        if not math.isfinite(confirm_seconds) or confirm_seconds <= 0:
            raise ValueError("confirm_seconds must be finite and positive")
        self.polygon = polygon
        self.confirm_seconds = confirm_seconds
        self.visits = {}
        self.last_frame = -1
        self.last_time = -1.0
        self.event_count = 0

    def update(self, frame_index, time_seconds, rows, armed):
        if not math.isfinite(time_seconds) or time_seconds < 0:
            raise ValueError("Timestamp must be finite and nonnegative")
        if frame_index <= self.last_frame or time_seconds <= self.last_time:
            raise ValueError("Frames and timestamps must increase")
        if frame_index != self.last_frame + 1 or not armed:
            self.visits.clear()
        self.last_frame, self.last_time = frame_index, time_seconds
        states, events, present = {}, [], set()
        for row in rows:
            track_id = int(row['track_id'])
            if track_id in present:
                raise ValueError("Duplicate ID in one frame")
            present.add(track_id)
            point = ((float(row['x1']) + float(row['x2'])) / 2, float(row['y2']))
            in_zone = inside_polygon(point, self.polygon)
            if not armed or not in_zone:
                self.visits.pop(track_id, None)
                states[track_id] = 'DISARMED' if not armed else 'OUTSIDE'
                continue
            visit = self.visits.setdefault(track_id, {'since': time_seconds, 'alerted': False})
            if not visit['alerted'] and time_seconds - visit['since'] + 1e-9 >= self.confirm_seconds:
                visit['alerted'] = True
                self.event_count += 1
                events.append({'event_id': self.event_count, 'frame_index': frame_index,
                               'time_seconds': round(time_seconds, 6), 'track_id': track_id,
                               'event_type': 'suspected_intrusion',
                               'zone_entered_seconds': round(visit['since'], 6),
                               'observed_duration_seconds': round(time_seconds - visit['since'], 6)})
            states[track_id] = 'ALERT' if visit['alerted'] else 'PENDING'
        for track_id in self.visits.keys() - present:
            del self.visits[track_id]
        return states, events
