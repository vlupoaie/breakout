import ctypes
import random
import platform
try:
    import tkinter
except ImportError:
    pass
from math import pi


def get_screen_size():
    if platform.system() == 'Windows':
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    else:
        root = tkinter.Tk()
        return root.winfo_screenwidth(), root.winfo_screenheight()


def get_random_angle(around_angle):
    return around_angle + random.choice([-pi / 12, pi / 12])
