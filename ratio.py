def akt1():
    active = 269
    non_active = 14281
    all_ = 14550
    return non_active/active

def ampc():
    active = 48
    non_active = 2802
    all_ = 2850
    return non_active/active

def cp3a4():
    active = 165
    non_active = 11685
    all_ = 11850
    return non_active/active

def cxcr4():
    active = 40
    non_active = 3260
    all_ = 3300
    return non_active/active

def gcr():
    active = 176
    non_active = 11824
    all_ = 12000
    return non_active/active

def hivpr():
    active = 529
    non_active = 35471
    all_ = 36000
    return non_active/active

def hivrt():
    active = 307
    non_active = 14993
    all_ = 15300
    return non_active/active

def kif11():
    active = 116
    non_active = 6634
    all_ = 6750
    return non_active/active


# (active, all_) counts per protein, for computing EF's random hitrate against
# the full nominal dataset composition (not the subset actually rendered/
# tested, which can be smaller due to docking/conformer-generation attrition).
# Kept separate from the functions above so pos_weight's eval("ratio."+name)()
# contract is untouched.
_COUNTS = {
    "akt1": (269, 14550),
    "ampc": (48, 2850),
    "cp3a4": (165, 11850),
    "cxcr4": (40, 3300),
    "gcr": (176, 12000),
    "hivpr": (529, 36000),
    "hivrt": (307, 15300),
    "kif11": (116, 6750),
}


def counts(name):
    """Returns (active, all_) nominal dataset compound counts for `name`."""
    return _COUNTS[name]
