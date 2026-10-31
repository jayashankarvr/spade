"""Tests for the provenance graph."""

import numpy as np

from spade import ForensicsEngine
from spade.bench.synthetic import make_copy_move, make_negative_pair, make_recolored_splice
from spade.provenance import build_provenance, to_dict, to_dot


def test_links_shared_regions_and_copy_moves():
    splice = make_recolored_splice(seed=0)
    images = {
        "donor": splice.donor,
        "edited": splice.tampered,
        "unrelated": make_negative_pair(seed=300).tampered,
        "cm": make_copy_move(seed=1).tampered,
    }
    engine = ForensicsEngine()
    for name, image in images.items():
        engine.index_target(image, name)
    links = build_provenance(engine, {k: (v, 0.0) for k, v in images.items()}, copy_moves=True)

    pairs = {frozenset((link.a, link.b)) for link in links}
    assert frozenset(("donor", "edited")) in pairs
    assert frozenset(("cm",)) in pairs                      # duplicated within itself
    assert not any("unrelated" in p for p in pairs)
    assert links == sorted(links, key=lambda link: link.log10_nfa)

    dot = to_dot(links)
    assert dot.startswith("graph provenance")
    assert '"donor" -- "edited"' in dot or '"edited" -- "donor"' in dot
    record = to_dict(links)[0]
    assert {"a", "b", "log10_nfa", "regions"} <= set(record)
