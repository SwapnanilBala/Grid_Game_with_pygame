import time
import numpy as np
from gridgame import *

##############################################################################################################################

# You can visualize what your code is doing by setting the GUI argument in the following line to true.
# The render_delay_sec argument allows you to slow down the animation, to be able to see each step more clearly.

# For your final submission, please set the GUI option to False.

# The gs argument controls the grid size. You should experiment with various sizes to ensure your code generalizes.
# Please do not modify or remove lines 18 and 19.

##############################################################################################################################

game = ShapePlacementGrid(GUI=True, render_delay_sec=0.5, gs=6, num_colored_boxes=5)
shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')
np.savetxt('initial_grid.txt', grid, fmt="%d")

##############################################################################################################################

# Initialization

# shapePos is the current position of the brush.

# currentShapeIndex is the index of the current brush type being placed (order specified in gridgame.py, and assignment instructions).

# currentColorIndex is the index of the current color being placed (order specified in gridgame.py, and assignment instructions).

# grid represents the current state of the board. 
    
    # -1 indicates an empty cell
    # 0 indicates a cell colored in the first color (indigo by default)
    # 1 indicates a cell colored in the second color (taupe by default)
    # 2 indicates a cell colored in the third color (veridian by default)
    # 3 indicates a cell colored in the fourth color (peach by default)

# placedShapes is a list of shapes that have currently been placed on the board.
    
    # Each shape is represented as a list containing three elements: a) the brush type (number between 0-8), 
    # b) the location of the shape (coordinates of top-left cell of the shape) and c) color of the shape (number between 0-3)

    # For instance [0, (0,0), 2] represents a shape spanning a single cell in the color 2=veridian, placed at the top left cell in the grid.

# done is a Boolean that represents whether coloring constraints are satisfied. Updated by the gridgames.py file.

##############################################################################################################################

shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')

print(shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done)


####################################################
# Timing your code's execution for the leaderboard.
####################################################

start = time.time()  # <- do not modify this.



##########################################
# Write all your code in the area below. 
##########################################





# -----------------------------
# First-Choice Hill Climbing
# -----------------------------
# Variant used (allowed by assignment): allow limited "sideways" moves (equal score),
# and perform random restarts (undo all placed shapes) if stuck.
# This keeps the method fundamentally first-choice hill climbing, but improves reliability.

import random

# Pull the latest state
shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')

GS = grid.shape[0]
NUM_SHAPES = len(game.shapes)          # 9 shapes in gridgame.py :contentReference[oaicite:6]{index=6}
NUM_COLORS = len(game.colors)          # 4 colors (default) :contentReference[oaicite:7]{index=7}

#  Scoring (lower is better)

def count_conflicts(g: np.ndarray) -> int:
    """Count orthogonal adjacent equal-color pairs (ignore -1). Count each pair once."""
    c = 0
    n = g.shape[0]
    for y in range(n): # so y is basically the row index
        for x in range(n): # and x is the column index
            col = g[y, x]
            if col == -1: # -1 means no color, that's a relief now we are checking the horizontal and the vertical grids which are adjacent to that
                continue
            if x + 1 < n and g[y, x + 1] == col:
                c += 1 # when we are doing x+1, we are going sideways, horizontally
            if y + 1 < n and g[y + 1, x] == col:
                c += 1 # similarly when are doing y+1, we are moving one row down from the previous one
    return c

def objective(g: np.ndarray, shapes_used: int) -> int:
    conflicts = count_conflicts(g) # counts the number of conflicts we have for the grid if it's 0 then its legal otherwise its illegal
    empty = int(np.sum(g == -1)) # so g == -1 creates a boolean grid then tells us how many empty cells are in the grid
    used_colors = set(int(v) for v in np.unique(g) if v != -1) # returns the unique color numeric indicators -1 means it's empty
    num_colors_used = len(used_colors) # tells us how many colors are currently being used

    # Weighting: correctness >> completion >> colors >> shapes, this is what our agent will get taste, for conflict the highest increment is score which is something we are trying to minimize with each step
    return (100000 * conflicts) + (1000 * empty) + (10 * num_colors_used) + shapes_used
    # so basically this is our priority queue: correctness over everything, then when we don't have many conflicts otherwise they are quite low, the agent will put emphasis on filing the grid
    # now comes the coloring, so with num_color_used we are telling the agent that this is the third important thing meaning if the above two seem fine then you can try to minimize the color options
    # lastly we have the least penalty for the shapes, but we still are trying to use as fewer variations of the shapes as possible


#  Helpers to simulate placement (without altering environment)

def apply_shape_to_copy(g: np.ndarray, shape_idx: int, pos_xy: tuple[int, int], color_idx: int) -> np.ndarray:
    """Return new grid after applying a shape at pos (x,y). Assumes placement is legal."""
    x0, y0 = pos_xy # these stand for the column start and row start
    newg = g.copy() # this is the copy of the updated grid so far on which we will visualize our implementation
    shape = game.shapes[shape_idx] # get us the shape we are currently using
    for i, row in enumerate(shape): # `i` is basically the row inside the shape
        for j, cell in enumerate(row): # and conversely j is the column indicator inside the shape
            if cell:
                newg[y0 + i, x0 + j] = color_idx # newg[y0 + i, x0 + j] is our affected grid within the visualization
    return newg

def random_empty_cell(g: np.ndarray) -> tuple[int, int] | None:
    empties = np.argwhere(g == -1) # here we are creating a boolean grid and then returning the indices where it's True
    if len(empties) == 0:
        return None
    y, x = empties[random.randrange(len(empties))] # this gives us one random coordinate pair for the corresponding grid
    return int(x), int(y) # this conversion is done for a purpose that serves later

def candidate_positions_covering_anchor(shape_idx: int, anchor_xy: tuple[int, int]) -> list[tuple[int, int]]:
    """All top-left positions (x,y) where shape covers anchor cell on a '1' cell."""
    ax, ay = anchor_xy
    shape = game.shapes[shape_idx] # the shape indicating which grids will be painted
    positions = []
    # with this basically we are taking our brush to the anchor (empty grid that needs coloring)
    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell: # i and j stand for the row and column index inside the shape
                x0 = ax - j
                y0 = ay - i
                if x0 < 0 or y0 < 0: # if the position is out of bound meaning if we have moved out from the whole grid
                    continue
                # canPlace expects (grid, shape_array, pos_list[x,y]) and checks overlap/fit only :contentReference[oaicite:8]{index=8}
                if game.canPlace(grid, shape, [x0, y0]):
                    positions.append((x0, y0))
    return positions # this returns a set of valid placement options, so that from here we can simulate the rest

def choose_color_for_shape(g: np.ndarray, shape_idx: int, pos_xy: tuple[int, int]) -> int:
    # We have the inputs: g: this is the current grid, shape_idx is the corresponding index of the shape
    # pos_xy starting point from where the coloring will start
    """
    Pick a color that tends to reduce conflicts.
    We try a few candidates (including getAvailableColor suggestions) and take best by objective.
    """
    # Start with a few random colors + a couple 'available' colors from covered cells
    candidate_colors = set(random.randrange(NUM_COLORS) for _ in range(2))
    # only going with 2 random colors, and the use of set is to avoid duplicates
    shape = game.shapes[shape_idx] # it's our 2D mask,
    x0, y0 = pos_xy # position of the anchor
    covered_cells = []

    for i, row in enumerate(shape):
        for j, cell in enumerate(row):
            if cell:
                covered_cells.append((x0 + j, y0 + i))
                # This is how we basically convert the cells into a grid coordinate and store it into covered_cell, and
                # this is essentially the list of all board coordinates the shape is intended to fill when place here
    # Use professor-provided helper (returns a random non-adjacent color for a single cell) :contentReference[oaicite:9]{index=9}

    for (cx, cy) in covered_cells[:2]:
        # Fast heuristic: We sample only a couple covered cells to get "safe color" suggestions
        # safer color implies: We need not require evaluating all 4 colors for every candidate move; usually 2–4 smart picks are enough to find a good one, and it keeps the search fast.
#       # (calling getAvailableColor for every covered cell is comparatively slower and generally unnecessary).
        candidate_colors.add(int(game.getAvailableColor(g, cx, cy)))
        # This delivers us a color that is not adjacent-conflict for that one cell, color options have been mentioned above

    best_color = 0
    best_score = 10**18
    for c in candidate_colors:
        newg = apply_shape_to_copy(g, shape_idx, pos_xy, c)
        # This is our MVP which fabricates a copy of the grid and paints the shape on it using color c
        sc = objective(newg, shapes_used=len(placedShapes) + 1)
        # If we recall we created the objective function that kept track of a parameter called score and
        # the goal for the agent was to try to minimize it in each step as much as it could, so simpler words
        # it computes how "bad" the resulting grid is, if the score get too high that means that it's not espoused to proceed with this option
        if sc < best_score:
            best_score = sc
            best_color = c # replacing the best color since clearly c was the winner

    return best_color

#  Execute-only interaction wrappers

def exec_cmd(cmd: str):
    global shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done
    shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute(cmd)
    # This one sends one command to the environment (game.execute(cmd)),
    # and then it updates our local variables whatever the environment yields
    # the variables above are global, and we update them everywhere

def set_shape(target_idx: int):
    # Cycle forward using 'switchshape'/'h' :contentReference[oaicite:10]{index=10}
    global currentShapeIndex
    steps = (target_idx - currentShapeIndex) % NUM_SHAPES
    for _ in range(steps):
        exec_cmd('switchshape')
        # basically, to reach a specific shape index, we repeatedly cycle forward until we eventually land on it.

def set_color(target_idx: int):
    # Cycle forward using 'switchcolor'/'k' :contentReference[oaicite:11]{index=11}
    global currentColorIndex
    steps = (target_idx - currentColorIndex) % NUM_COLORS
    for _ in range(steps):
        exec_cmd('switchcolor')
        # essentially we have implemented the same methodology

def goto_pos(target_xy: tuple[int, int]):
    """Move brush to (x,y). Note: movement boundaries depend on current shape size."""
    tx, ty = target_xy
    while shapePos[0] < tx:
        exec_cmd('right')
    while shapePos[0] > tx:
        exec_cmd('left')
    while shapePos[1] < ty:
        exec_cmd('down')
    while shapePos[1] > ty:
        exec_cmd('up')
    # This one is quite self-explanatory, the comparison is done to prevent shape pos out of bounds error

def place_current():
    exec_cmd('place')
    # This places the present shape considering everything went smooth

def undo_one():
    exec_cmd('undo')
    # Should things go south, we have the undo function to save us from catastrophe by reverting back our grid to the last stable state

def restart_to_initial():
    # if after a certain time and certain moves the grid situation doesn't seem to improve a lot,
    # Undo everything we placed (initial random colored cells are not in placedShapes).
    while len(placedShapes) > 0:
        undo_one()

# --- Hill climbing loop ---
random.seed(0)

# Limits (keep safe for autograder 10 min; you can tune)
TIME_BUDGET_SEC = 5 * 60
MAX_PLATEAU_SIDEWAYS = 250      # allow equal-score moves up to this many
MAX_TRIES_PER_STEP = 300        # how many random neighbors to sample before declaring "stuck"
MAX_RESTARTS = 50

best_grid = grid.copy() # saves the best board state found so far
best_shapes = list(placedShapes) # this saves the sequence of placed shapes corresponding to that best grid
best_score = objective(grid, shapes_used=len(placedShapes)) # our current best score (less is better)

restarts = 0 # keeps track of how many times we restarted
sideways_used = 0 # similarly this keeps track of the sideways_used

while (time.time() - start) < TIME_BUDGET_SEC:
    # Refresh state
    exec_cmd('export')
    if done:
        break # if the board has been fully colored then we break out of the loop

        #  Just to check that our program is running or not
    if int(time.time() - start) % 5 == 0:
        print("t=", int(time.time() - start),
                "placed=", len(placedShapes),
                "score=", objective(grid, shapes_used=len(placedShapes)),
                "restarts=", restarts)


    cur_score = objective(grid, shapes_used=len(placedShapes))
    # This uses the objective function which is responsible for keeping track of the score (lower is better)

    improved = False
    accepted_sideways = False
    # If none of the flags above become true, that means that we could not place anything acceptable so we have to restart

    # First-choice: This keeps sampling random moves, and then accepts the first that improves objective
    for _ in range(MAX_TRIES_PER_STEP):
        anchor = random_empty_cell(grid)
        if anchor is None:
            break # no cells left to color, meaning we are done

        shape_idx = random.randrange(NUM_SHAPES) # picks a random shape from our bucket
        positions = candidate_positions_covering_anchor(shape_idx, anchor)
        # This computes all the anchor positions pos_xy in a way that placing the shape at
        # pos_xy would include the chosen anchor_cell
        if not positions:
            continue # if that doesn't happen to be the case then try again

        pos_xy = random.choice(positions) # again going with a random position
        color_idx = choose_color_for_shape(grid, shape_idx, pos_xy)
        # this helper function we have tries a few colors and chooses the one that gives us the optimal objective

        # This is the simulation happening before the actual placement
        newg = apply_shape_to_copy(grid, shape_idx, pos_xy, color_idx)
        new_score = objective(newg, shapes_used=len(placedShapes) + 1)

        if new_score < cur_score:
            # Accept move
            set_shape(shape_idx)
            goto_pos(pos_xy)
            set_color(color_idx)
            place_current()
            improved = True
            sideways_used = 0
            break
            # we proceed with this new state of the board

        # Sideways move (optional variant)
        if new_score == cur_score and sideways_used < MAX_PLATEAU_SIDEWAYS:
            set_shape(shape_idx)
            goto_pos(pos_xy)
            set_color(color_idx)
            place_current()
            accepted_sideways = True
            sideways_used += 1
            break
            # in-case we don't get a lower score we keep this one with slight change that is we increase the sideways_used
    # Track best-so-far
    cur_score = objective(grid, shapes_used=len(placedShapes))
    if cur_score < best_score:
        best_score = cur_score
        best_grid = grid.copy()
        best_shapes = list(placedShapes)
        # This is how we are keeping a checkpoint of the best (lowest-score) state that we have found so far,
        # so we can revert/replay it later if restarts or sideways moves end up comparatively worse.

    # If we get stuck, meaning we are neither getting an improved score and accepted sideways
    if not improved and not accepted_sideways:
        restarts += 1
        if restarts > MAX_RESTARTS:
            break
        restart_to_initial()
        sideways_used = 0


# End: export final state so the template writes correct outputs
exec_cmd('export')

# If we didn't finish, revert to best-so-far (by undoing and replaying best_shapes)
# (Optional safety to avoid ending in a worse restart state.)
if not done and best_score < objective(grid, shapes_used=len(placedShapes)):
    restart_to_initial()
    # Replay best shapes through execute (still legal since it was achieved earlier)
    for (sidx, spos, cidx) in best_shapes:
        set_shape(int(sidx))
        goto_pos((int(spos[0]), int(spos[1])))
        set_color(int(cidx))
        place_current()
    exec_cmd('export')





########################################

# Do not modify any of the code below. 

########################################

end=time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end-start))
