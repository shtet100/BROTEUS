"""
BROTEUS Open Vocabulary
========================

Large label database for CLIP classification.
Not hardcoded to a small list — covers hundreds of common objects.
Users can add custom labels at runtime.
"""

# Base vocabulary: ~200 common graspable/detectable objects
# Organized by category for maintainability, but CLIP sees them all equally

HOUSEHOLD = [
    'coffee mug', 'water glass', 'wine glass', 'bowl', 'plate', 'spoon',
    'fork', 'knife', 'cutting board', 'pot', 'pan', 'water bottle',
    'thermos', 'tupperware container', 'jar', 'can', 'box', 'basket',
    'vase', 'candle', 'lamp', 'flashlight', 'clock', 'picture frame',
    'pillow', 'blanket', 'towel', 'soap', 'toothbrush', 'comb',
    'umbrella', 'keys', 'wallet', 'sunglasses', 'hat', 'shoe',
    'bag', 'backpack', 'suitcase', 'tissue box',
]

ELECTRONICS = [
    'laptop', 'keyboard', 'computer mouse', 'monitor', 'TV screen',
    'phone', 'smartphone', 'tablet', 'headphones', 'earbuds',
    'speaker', 'microphone', 'camera', 'webcam', 'USB cable',
    'charger', 'power bank', 'remote control', 'game controller',
    'router', 'hard drive', 'USB drive', 'printer', 'scanner',
]

TOOLS = [
    'screwdriver', 'wrench', 'pliers', 'hammer', 'tape measure',
    'scissors', 'utility knife', 'drill', 'saw', 'level',
    'tape', 'duct tape', 'glue', 'clamp', 'wire cutter',
    'soldering iron', 'multimeter', 'allen key',
]

STATIONERY = [
    'pen', 'pencil', 'marker', 'eraser', 'ruler', 'stapler',
    'paper clip', 'binder clip', 'notebook', 'book', 'folder',
    'envelope', 'sticky note', 'calculator', 'tape dispenser',
]

FOOD = [
    'apple', 'banana', 'orange', 'lemon', 'tomato', 'potato',
    'bread', 'sandwich', 'pizza slice', 'cookie', 'candy bar',
    'soda can', 'juice box', 'coffee cup', 'tea cup',
]

TOYS_COLLECTIBLES = [
    'Iron Man helmet', 'Iron Man mask', 'action figure', 'figurine',
    'toy car', 'toy robot', 'stuffed animal', 'teddy bear',
    'Lego set', 'puzzle', 'board game', 'playing cards',
    'trophy', 'medal', 'statue', 'snow globe', 'bobblehead',
    'superhero mask', 'costume helmet', 'replica weapon',
]

FURNITURE = [
    'chair', 'office chair', 'stool', 'table', 'desk',
    'shelf', 'bookshelf', 'drawer', 'cabinet', 'bed',
    'couch', 'bench', 'rug', 'curtain',
]

CLOTHING = [
    'shirt', 'jacket', 'hoodie', 'pants', 'sock', 'glove',
    'scarf', 'belt', 'tie', 'watch', 'bracelet', 'ring',
]

INSTRUMENTS = [
    'acoustic guitar', 'electric guitar', 'ukulele', 'violin',
    'drum', 'drumstick', 'piano keyboard', 'harmonica',
    'guitar pick', 'guitar capo', 'tuner',
]

MISC = [
    'plant', 'potted plant', 'flower', 'rock', 'stone',
    'ball', 'tennis ball', 'basketball', 'soccer ball',
    'rope', 'chain', 'wire', 'cable', 'string',
    'humidifier', 'fan', 'heater', 'air purifier',
    'trash can', 'recycling bin', 'spray bottle', 'bucket',
    'battery', 'light bulb', 'extension cord',
]

# Combined default vocabulary
DEFAULT_VOCABULARY = (
    HOUSEHOLD + ELECTRONICS + TOOLS + STATIONERY + FOOD +
    TOYS_COLLECTIBLES + FURNITURE + CLOTHING + INSTRUMENTS + MISC
)

# Remove duplicates while preserving order
_seen = set()
VOCABULARY = []
for label in DEFAULT_VOCABULARY:
    if label.lower() not in _seen:
        _seen.add(label.lower())
        VOCABULARY.append(label)


class OpenVocabulary:
    """Manages the label vocabulary. Users can add/remove labels at runtime."""

    def __init__(self):
        self.labels = list(VOCABULARY)
        self._custom_labels = []

    def add_label(self, label: str):
        """Add a custom label."""
        if label.lower() not in {l.lower() for l in self.labels}:
            self.labels.append(label)
            self._custom_labels.append(label)
            return True
        return False

    def remove_label(self, label: str):
        """Remove a label."""
        self.labels = [l for l in self.labels if l.lower() != label.lower()]

    def get_all(self) -> list:
        return list(self.labels)

    @property
    def count(self) -> int:
        return len(self.labels)
