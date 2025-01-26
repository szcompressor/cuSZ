# 23-12-29

import matplotlib as mpl

# runtime (e.g., in notebook)
mpl.rcParams["figure.figsize"] = [6.0, 3.0]
# mpl.rcParams['figure.dpi'] = 144

# save
mpl.rcParams["savefig.transparent"] = False
mpl.rcParams["savefig.dpi"] = 300

# Let's see what will go on with early setup of tight_layout()
mpl.rcParams["figure.autolayout"] = True

mpl.rcParams["grid.color"] = "k"
mpl.rcParams["grid.linestyle"] = "-"
mpl.rcParams["grid.linewidth"] = 0.25

mpl.rcParams["axes.grid"] = False

# not changing the default, but leaving clue for runtime change
mpl.rcParams["xtick.direction"] = "out"
mpl.rcParams["ytick.direction"] = "out"

mpl.rcParams["font.family"] = "Helvetica"

mpl.rcParams["lines.linewidth"] = 0.75
