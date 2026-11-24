"""
Loading training units as arrays: texts, section-type ids, the
multi-label target matrix Y and the loss mask M.

    Y[i, j] = 1 if unit i has deficiency j, else 0
    M[i, j] = 1 if label j applies to unit i's section type AND is known

Example - an education chunk:
    labels  : edu_missing_degree=1, edu_missing_institution=0, edu_missing_date=None
    Y row   : [1, 0, 0, 0, 0, 0, 0, 0, 0]
    M row   : [1, 1, 0, 0, 0, 0, 0, 0, 0]   (date unknown, experience/summary/skills n/a)
"""

import json

import numpy as np

from rca import config


def load_units(split=None):
    units = []
    with open(config.UNITS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            unit = json.loads(line)
            if split is None or unit["split"] == split:
                units.append(unit)
    return units


def to_arrays(units):
    texts = [u["text"] for u in units]
    type_ids = np.array([config.SECTION_TYPES.index(u["type"]) for u in units])

    Y = np.zeros((len(units), len(config.LABEL_NAMES)), dtype=np.float32)
    M = np.zeros_like(Y)
    for i, unit in enumerate(units):
        for j, name in enumerate(config.LABEL_NAMES):
            value = unit["labels"].get(name)
            if value is not None:
                Y[i, j] = value
                M[i, j] = 1
    return texts, type_ids, Y, M
