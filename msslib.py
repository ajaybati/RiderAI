import mss
import numpy as np
with mss.mss() as mss_instance:  # Create a new mss.mss instance
    print(mss_instance.monitors)
    monitor_1 = mss_instance.monitors[1]  # Identify the display to capture
    screenshot = mss_instance.grab(monitor_1)  # Take the screenshot
print(np.array(screenshot))
