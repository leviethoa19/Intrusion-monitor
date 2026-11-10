"""Behavioral scenarios for zone/time alerts; no model needed."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from intrusion_engine import IntrusionEngine, inside_polygon

ZONE = [[0, 0], [10, 0], [10, 10], [0, 10]]


def person(track_id=1, x=5, y=5):
    return {'track_id': track_id, 'x1': x - 1, 'x2': x + 1, 'y2': y}


def test_polygon_boundary_and_bottom_center():
    assert inside_polygon((0, 5), ZONE)
    assert inside_polygon((5, 5), ZONE)
    assert not inside_polygon((11, 5), ZONE)
    engine = IntrusionEngine(ZONE, 1)
    assert engine.update(0, 0, [person(y=11)], True) == ({1: 'OUTSIDE'}, [])


def test_disarmed_time_does_not_count_and_alert_does_not_repeat():
    engine = IntrusionEngine(ZONE, 2)
    assert engine.update(0, 0, [person()], False) == ({1: 'DISARMED'}, [])
    assert engine.update(1, 10, [person()], True) == ({1: 'PENDING'}, [])
    assert engine.update(2, 11, [person()], True)[1] == []
    states, events = engine.update(3, 12, [person()], True)
    assert states == {1: 'ALERT'}
    assert len(events) == 1 and events[0]['zone_entered_seconds'] == 10
    assert engine.update(4, 13, [person()], True)[1] == []


@pytest.mark.parametrize('interrupt', ['missing', 'outside', 'disarmed', 'new_id'])
def test_interruption_requires_fresh_confirmation(interrupt):
    engine = IntrusionEngine(ZONE, 2)
    engine.update(0, 0, [person()], True)
    rows = [] if interrupt == 'missing' else [person(x=20)] if interrupt == 'outside' else [person()]
    engine.update(1, 1, rows, interrupt != 'disarmed')
    track_id = 2 if interrupt == 'new_id' else 1
    assert engine.update(2, 2, [person(track_id)], True) == ({track_id: 'PENDING'}, [])
    assert len(engine.update(3, 4, [person(track_id)], True)[1]) == 1


def test_independent_people_and_reentry():
    engine = IntrusionEngine(ZONE, 1)
    engine.update(0, 0, [person(1), person(2, x=20)], True)
    assert [e['track_id'] for e in engine.update(1, 1, [person(1), person(2)], True)[1]] == [1]
    assert [e['track_id'] for e in engine.update(2, 2, [person(1), person(2)], True)[1]] == [2]
    engine.update(3, 3, [], True)
    engine.update(4, 4, [person(1)], True)
    assert len(engine.update(5, 5, [person(1)], True)[1]) == 1


def test_skipped_frames_reset_and_out_of_order_is_rejected():
    engine = IntrusionEngine(ZONE, 1)
    engine.update(0, 0, [person()], True)
    assert engine.update(2, 2, [person()], True)[1] == []
    with pytest.raises(ValueError):
        engine.update(2, 3, [person()], True)


def test_invalid_geometry_and_time_are_rejected():
    with pytest.raises(ValueError):
        IntrusionEngine([[0, 0], [1, 1], [2, 2]], 1)
    with pytest.raises(ValueError):
        IntrusionEngine(ZONE, 0)
    with pytest.raises(ValueError):
        IntrusionEngine(ZONE, 1).update(0, float('nan'), [], True)
