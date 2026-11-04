# Reference data for PDQ

The eight `bridge-*.jpg` images and the expected hashes in `expected.csv` come
from Meta's ThreatExchange repository (https://github.com/facebook/ThreatExchange),
commit `24bfce666e83b58336a819ab391073ceab170f75`:

- images: `pdq/data/reg-test-input/dih/`
- hashes: first eight result lines of `pdq/cpp/reg_test/expected/out`
  (`filename,hash,quality`)

Licence: BSD (Meta Platforms, Inc.), which permits redistribution in source and
binary form provided the copyright notice, conditions and disclaimer are kept.
The full text is in `LICENSE-ThreatExchange` next to these files.

The images are numbered in Meta's dihedral order: original, rotate 90, rotate
180, rotate 270, flip x, flip y, flip +1 (diagonal), flip -1 (anti-diagonal).

## pHash

`imagehash` publishes no fixed hash vectors: the installed package (4.3.2)
ships no test data or expected hashes. The `phash` tests therefore check
SPADE's output against `imagehash.phash` run directly on the same pixels.
