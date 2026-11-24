"""Dataset constants and the hierarchical GrassClover label mapping."""

CLASS_NAMES = ("soil", "grass", "white_clover", "red_clover", "weeds")
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255

# Labels in synthetic_images/ImageLabels are hierarchical (0--14).  Generic
# clover pixels cannot be assigned reliably to red or white clover, so they are
# ignored for the five-class experiment used by the paper.
GRASSCLOVER15_TO_FIVE = {
    0: 0,   # soil
    1: IGNORE_INDEX,  # generic clover
    2: 1,   # grass
    3: 4,   # generic weeds
    4: 2,   # white clover
    5: 3,   # red clover
    6: 4,   # dandelion
    7: 4,   # shepherd's purse
    8: 4,   # thistle
    9: 2,   # white clover flower
    10: 2,  # white clover leaf
    11: 3,  # red clover flower
    12: 3,  # red clover leaf
    13: IGNORE_INDEX,  # unknown clover leaf
    14: IGNORE_INDEX,  # unknown clover flower
}

MASK_COLORS = (
    (32, 32, 32),     # soil
    (40, 100, 220),   # grass
    (245, 210, 45),   # white clover
    (170, 70, 190),   # red clover
    (235, 125, 35),   # weeds
)

TARGET_NAMES = ("grass", "clover", "white_clover", "red_clover", "weeds")

