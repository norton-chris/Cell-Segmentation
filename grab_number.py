#----------------------------------------------------------------------------
# Created By  : Chris Norton
# ---------------------------------------------------------------------------
"""
Grab the number from a filename, and rename to #.tif
Not useful for most applications, just dataset management.
"""
# ---------------------------------------------------------------------------

import os
import re

path = "TrainingDataset/Labels/"
for i in os.listdir(path):
#     #[int(s) for s in i.split() if s.isdigit()]
#
#     for s in i.split():
#         str = ""
#         int(filter(s.isdigit, str))
#         print(str)

    s = re.findall(r'\d+', i)
    os.rename(path + i, path + s[0] + ".tif")
    print(s[0])
