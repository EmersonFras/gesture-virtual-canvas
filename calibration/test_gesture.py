"""Focused tests for the slide-aligned Task 3 pipeline."""
import sys

import numpy as np

sys.path.insert(0, '.')

from calibration.gesture import GestureRecognizer


def test_hu_moment_point_template_matches_draw():
    recognizer = GestureRecognizer(smooth_window=1)
    mask = recognizer._template_mask(160, 'point')
    contour = recognizer._largest_contour(mask)
    label, distance = recognizer._classify_static_gesture(contour, mask.shape)
    command, confidence = recognizer._command_from_slide_methods(
        static_label=label,
        hu_distance=distance,
        swipe_score=0.0,
        clear_triggered=False,
    )
    assert label == 'point'
    assert command == 'draw'
    assert confidence >= 0.55


def test_hu_moment_open_palm_template_matches_erase():
    recognizer = GestureRecognizer(smooth_window=1)
    mask = recognizer._template_mask(160, 'open_palm')
    contour = recognizer._largest_contour(mask)
    label, distance = recognizer._classify_static_gesture(contour, mask.shape)
    command, confidence = recognizer._command_from_slide_methods(
        static_label=label,
        hu_distance=distance,
        swipe_score=0.0,
        clear_triggered=False,
    )
    assert label == 'open_palm'
    assert command == 'erase'
    assert confidence >= 0.55


def test_mhi_swipe_triggers_clear():
    recognizer = GestureRecognizer(smooth_window=1, swipe_cooldown=1)
    recognizer._centroid_history.extend([(30, 60), (90, 62), (150, 61), (220, 63)])
    triggered = recognizer._detect_swipe_clear(0.03, 8.0)
    command, confidence = recognizer._command_from_slide_methods(
        static_label='fist',
        hu_distance=0.4,
        swipe_score=0.03,
        clear_triggered=triggered,
    )
    assert triggered is True
    assert command == 'clear'
    assert confidence >= 0.95


def test_mhi_requires_horizontal_motion():
    recognizer = GestureRecognizer(smooth_window=1)
    recognizer._centroid_history.extend([(30, 30), (35, 90), (40, 150), (44, 220)])
    triggered = recognizer._detect_swipe_clear(0.03, 85.0)
    assert triggered is False


def test_mhi_bootstraps_without_previous_frame():
    recognizer = GestureRecognizer(smooth_window=1)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    swipe_score, angle = recognizer._update_mhi(frame)
    assert swipe_score == 0.0
    assert angle is None


def test_command_stabilizer_requires_hold_frames():
    recognizer = GestureRecognizer(smooth_window=3, command_hold_frames=2)
    first = recognizer._stabilize_command('draw', 0.8, False)
    second = recognizer._stabilize_command('draw', 0.8, False)
    assert first == 'idle'
    assert second == 'draw'


def test_save_and_reload_templates(tmp_path=None):
    recognizer = GestureRecognizer(smooth_window=1)
    path = '/tmp/gesture_templates_test.json' if tmp_path is None else str(tmp_path / 'gesture_templates.json')
    point = recognizer._shape_templates['point']
    recognizer.save_templates(path, {'point': point})
    loaded = GestureRecognizer(template_path=path, smooth_window=1)
    assert np.allclose(loaded._shape_templates['point'], point)


if __name__ == '__main__':
    test_hu_moment_point_template_matches_draw()
    test_hu_moment_open_palm_template_matches_erase()
    test_mhi_swipe_triggers_clear()
    test_mhi_requires_horizontal_motion()
    test_mhi_bootstraps_without_previous_frame()
    test_command_stabilizer_requires_hold_frames()
    test_save_and_reload_templates()
    print('All gesture tests passed.')
