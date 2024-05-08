import time

from class_timer import control_timer

timer = control_timer()


def test_get_time():
    timer.start()
    time.sleep(1)
    timer.end()
    assert timer.get_time() >= 1
