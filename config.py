"""
Configuration file for Sarine Advisor Analysis v2.0
===================================================

Adjust these settings to customize the complete workflow behavior
"""

# Window Settings
WINDOW_TITLE = "Advisor"  # The window title to search for

# Step 1: Zoom Settings
ZOOM_SCROLL_AMOUNT = 6  # Fixed at 6 scroll wheel clicks (automatic, no prompt)
ZOOM_SCROLL_DELAY = 0.15  # Delay between each scroll (seconds)

# Step 2: Green Border Detection Settings
GREEN_MIN_AREA = 10000  # Minimum area in pixels for green border
GREEN_HUE_LOWER = 35  # Lower HSV hue for green (0-180)
GREEN_HUE_UPPER = 85  # Upper HSV hue for green (0-180)
GREEN_SAT_LOWER = 50  # Lower HSV saturation (0-255)
GREEN_SAT_UPPER = 255  # Upper HSV saturation (0-255)
GREEN_VAL_LOWER = 50  # Lower HSV value (0-255)
GREEN_VAL_UPPER = 255  # Upper HSV value (0-255)

# Step 3: Stone Detection Settings
AUTO_DETECT = True  # Use automatic detection (True) or manual center (False)
MANUAL_CENTER_X = 640  # Manual center X (used if AUTO_DETECT = False)
MANUAL_CENTER_Y = 400  # Manual center Y (used if AUTO_DETECT = False)
MANUAL_RADIUS = 150  # Manual radius (used if AUTO_DETECT = False)
DETECT_WITHIN_GREEN_AREA = True  # Limit stone detection to green border area

# Step 4: Rotation Settings
HORIZONTAL_CYCLES = 2  # Number of left-right cycles (Y-axis)
VERTICAL_CYCLES = 2  # Number of up-down cycles (X-axis)
CIRCULAR_ROTATIONS = 2  # Number of complete 360° rotations (Z-axis)

# Speed Settings (lower = slower, more precise)
HORIZONTAL_SPEED = 0.8  # Speed for horizontal rotation
VERTICAL_SPEED = 0.8  # Speed for vertical rotation
CIRCULAR_SPEED = 1.0  # Speed for circular rotation
CIRCULAR_STEPS = 60  # Steps per rotation (higher = smoother but slower)

# Visual Feedback
SHOW_INDICATORS = True  # Show colored boxes at start positions
SHOW_CROSSHAIR = True  # Show crosshair on detected center
INDICATOR_SIZE = 40  # Size of indicator boxes
INDICATOR_DURATION = 0.2  # How long to show indicators (seconds)

# Timing Settings
STARTUP_DELAY = 2  # Delay before starting each major step (seconds)
PHASE_DELAY = 0.5  # Delay between rotation phases (seconds)
CLICK_DELAY = 0.2  # Delay after mouse movements before clicking

# Detection Parameters
DETECTION_MIN_RADIUS = 30  # Minimum stone radius for detection
DETECTION_MAX_RADIUS = 500  # Maximum stone radius for detection

# Advanced Circle Detection Parameters
CIRCLE_PARAM1 = 50  # Higher = more strict edge detection
CIRCLE_PARAM2 = 30  # Lower = more circles detected
CIRCLE_BLUR_SIZE = 9  # Gaussian blur kernel size (must be odd)

# Advanced Contour Detection Parameters
CONTOUR_MIN_AREA = 1000  # Minimum contour area in pixels
CANNY_LOW_THRESHOLD = 50  # Lower threshold for Canny edge detection
CANNY_HIGH_THRESHOLD = 150  # Upper threshold for Canny edge detection

# Safety Settings
FAILSAFE = True  # Enable pyautogui failsafe (move mouse to corner to abort)
PAUSE_BETWEEN_ACTIONS = 0.05  # Pause between PyAutoGUI actions

# Advanced Settings
FALLBACK_TO_SCREEN_CENTER = True  # Use screen center if detection fails
USE_GREEN_AREA_CENTER_AS_FALLBACK = True  # Use green area center as fallback for stone detection