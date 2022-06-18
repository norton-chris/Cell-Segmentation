import splitfolders

# Split your dataset - using a ratio
splitfolders.ratio("input_folder", output="output",
                   seed=420, ratio=(.7, .2, .1), group_prefix=None, move=False)