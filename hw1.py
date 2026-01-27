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

game = ShapePlacementGrid(GUI=True, render_delay_sec=0.1, gs=6, num_colored_boxes=5)
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






# First-Choice Hill Climbing Method

# In our method we have allowed limited "sideways" moves ,limited time for finishing the task,
# a metric namely score which is a great indicator of how well optimized our method becomes
# And we have also implemented a feature that lets the agent perform random restarts if stuck.

# This keeps the method fundamentally first-choice hill climbing, but improves reliability.

import random  # importing the random module

# Here we are Pulling the latest state of the entire board

shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute('export')

# Below are our environment constants

grid_size = grid.shape[0]              # This is our shape of the grid (n X n), where n can be changed
num_shapes = len(game.shapes)          # number of brushes we have available, presently we have 9
num_colors = len(game.colors)          # As the name suggests, this represents the number of colors available as of now we have 4 colors

#  Scoring (lower is better) -> is a variable (tracker)
#  The idea is to keep a track of a variable that lets the agent know how good or bad it's last move was

def count_conflicts(grid_array: np.ndarray) -> int:
    """ This is to count the orthogonal adjacent equal-color pairs (ignore -1). Count each pair once."""
    conflict_count = 0  # Conflict counter
    n = grid_array.shape[0]
    for y in range(n):  # so y is basically the row index
        for x in range(n):  # and x is the column index
            cell_color = grid_array[y, x]  # This is the color value of the current cell
            if cell_color == -1:  # -1 means no color, that's a relief now we are checking the horizontal and the vertical grids which are adjacent to that
                continue
            if x + 1 < n and grid_array[y, x + 1] == cell_color:
                conflict_count += 1  # when we are doing x+1, we are going sideways, horizontally
            if y + 1 < n and grid_array[y + 1, x] == cell_color:
                conflict_count += 1  # similarly when are doing y+1, we are moving one row down from the previous one
    return conflict_count

def objective(grid_array: np.ndarray, shapes_used: int) -> int:
    conflicts = count_conflicts(grid_array)  # counts the number of conflicts we have for the grid if it's 0 then its legal otherwise its illegal
    empty = int(np.sum(grid_array == -1))  # so g == -1 creates a boolean grid then tells us how many empty cells are in the grid
    used_colors = set(int(v) for v in np.unique(grid_array) if v != -1)  # returns the unique color numeric indicators; -1 means it's empty
    num_colors_used = len(used_colors)  # tells us about the number of distinct colors that are currently being used

    # (This acts like a priority order) Weighting: correctness > completion > colors > shapes, this is what our agent will get taste, for conflict the highest increment is score which is something we are trying to minimize with each step
    return (100000 * conflicts) + (1000 * empty) + (10 * num_colors_used) + shapes_used  # This is the score the agent is supposed to decrease
    # so basically this is our priority queue: correctness over everything, then when we don't have many conflicts otherwise they are quite low, the agent will put emphasis on filing the grid
    # now comes the coloring, so with num_color_used we are telling the agent that this is the third important thing meaning if the above two seem fine then you can try to minimize the color options
    # lastly we have the least penalty for the shapes, but we still prefer fewer total placements (fewer 'place' actions)



#  Below are our agent's helpers who are to simulate placement (without making any alterations to the environment)

def apply_shape_to_copy(grid_array: np.ndarray, shape_index: int, top_left_xy: tuple[int, int], color_index: int) -> np.ndarray:
    """This special function simulate placing a shape on a COPY of the grid (no execute calls); also assumes the placement is already legal."""
    top_left_x, top_left_y = top_left_xy  # these stand for the column start and row start
    new_grid = grid_array.copy()  # this is the copy of the updated grid so far on which we will visualize our implementation
    shape_mask = game.shapes[shape_index]  # get us the shape we are currently using
    for i, row in enumerate(shape_mask):  # `i` is basically the row inside the shape
        for j, cell in enumerate(row):  # and conversely j is the column indicator inside the shape
            if cell:  # for this True equivalent means this part of the brush covers a particular grid and conversely False represents blank space
                new_grid[top_left_y + i, top_left_x + j] = color_index  # new_grid[top_left_y + i, top_left_x + j] is our affected grid within the visualization, color_index -> which color we are using for this
    return new_grid  # This one returns our simulated board

def random_empty_cell(grid_array: np.ndarray) -> tuple[int, int] | None:
    empty_positions = np.argwhere(grid_array == -1)  # here we are creating a boolean grid and then returning the indices where it's True, basically checking which cells are empty and which ones are not
    if len(empty_positions) == 0:  # No empty cells
        return None
    y, x = empty_positions[random.randrange(len(empty_positions))]  # this gives us one random coordinate pair for the corresponding grid
    # So numpy gives us (row,col) = (y,x), and not (x,y)
    return int(x), int(y)  # we have converted the dtype for a safer parameter dtype handling and also as our code expects (x,y) we flip the order

def candidate_positions_covering_anchor(shape_index: int, anchor_xy: tuple[int, int]) -> list[tuple[int, int]]:
    """ We can think of this function as (how come the brush covers some empty cell) """
    anchor_x, anchor_y = anchor_xy  # this is the empty cell that we wish to fill with some color
    shape_mask = game.shapes[shape_index]  # the shape indicating which grids will be painted
    candidate_positions = []  # creating an empty list which will later hold a set of valid placement options
    # with this basically we are taking our brush to the anchor (empty grid that needs coloring)
    for i, row in enumerate(shape_mask):  # row index inside targeted shape
        for j, cell in enumerate(row):  # column index inside targeted shape
            if cell:  # This tells us that if a cell is true then the brush can do it's magic there if not then that part won't get colored
                top_left_x = anchor_x - j
                top_left_y = anchor_y - i
                if top_left_x < 0 or top_left_y < 0:  # if the position is out of bound meaning if we have moved out from the whole grid
                    continue

                if game.canPlace(grid, shape_mask, [top_left_x, top_left_y]):  # This lets us only keep placements which are valid (in-bounds (no out of bounds error) + no overlap with already-colored cells/grids)
                    candidate_positions.append((top_left_x, top_left_y))  # adding up the valid placement options into the candidate_positions
    return candidate_positions  # this returns a set of valid placement options, so that from here we can simulate the rest

def choose_color_for_shape(grid_array: np.ndarray, shape_index: int, top_left_xy: tuple[int, int]) -> int:
    # We have the inputs: grid_array: this is the current grid, shape_index is the corresponding index of the shape
    # top_left_xy starting point from where the coloring will start
    """
    With this we pick a color that aims to reduce conflicts.
    And so, We try a few candidates (including getAvailableColor suggestions) and take best by objective (least score).
    """
    # Start with a few random colors + a couple 'available' colors from covered cells
    candidate_colors = set(random.randrange(num_colors) for _ in range(2))
    # only going with 2 random colors, and the use of set is to avoid duplicates
    shape_mask = game.shapes[shape_index]  # it's our 2D mask, 1 -> will paint a grid, 0 -> will do nothing
    top_left_x, top_left_y = top_left_xy  # position of the anchor
    covered_cells = []  # this is the initiation of the list that will later become a list of board grids which would be painted if we placed that particular shape here

    for i, row in enumerate(shape_mask):  # i -> row index, j -> column index
        for j, cell in enumerate(row):
            if cell:  # similarly, this tells us about the grids that the shape has it's effect on
                covered_cells.append((top_left_x + j, top_left_y + i))
                # This is how we basically convert the cells into a grid coordinate and store it into covered_cell, and
                # this is essentially the list of all board coordinates the shape is intended to fill when place here


    for (cx, cy) in covered_cells[:2]:
        # Fast heuristic: We sample only a couple covered cells(first 2 covered cells/grids) to get "safe color" suggestions
        # safer color implies: We need not require evaluating all 4 colors for every candidate move; usually 2–4 smart picks are enough to find a good one, and it keeps the search fast.
#       # (calling getAvailableColor for every covered cell is comparatively slower and generally unnecessary).
        candidate_colors.add(int(game.getAvailableColor(grid_array, cx, cy)))
        # This delivers us a color that is not adjacent-conflict for that one cell, color options have been mentioned above

    best_color = 0  # it's like a default option
    best_score = 10**18  # We start with a huge value so the first objective() score that we compute, will replace it (as we're trying to minimize it)

    for color_candidate in candidate_colors:
        simulated_grid = apply_shape_to_copy(grid_array, shape_index, top_left_xy, color_candidate)
        # This is our MVP which fabricates a copy of the grid and paints the shape on it using color c
        candidate_score = objective(simulated_grid, shapes_used=len(placedShapes) + 1)
        # If we recall we created the objective function that kept track of a parameter called score and
        # the goal for the agent was to try to minimize it in each step as much as it could, so simpler words
        # it computes how "bad" the resulting grid is, if the score get too high that means that it's not espoused to proceed with this option
        if candidate_score < best_score:
            best_score = candidate_score
            best_color = color_candidate  # replacing the best color since clearly c was the winner

    return best_color

#  Execute-only interaction wrappers

def exec_cmd(cmd: str):
    global shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done
    shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute(cmd)
    # This one sends one command to the environment (game.execute(cmd)),
    # and then it updates our local variables whatever the environment yields
    # the variables above are global, and we update them everywhere

def set_shape(target_idx: int):
    # Cycle forward using 'switchshape' -> This is found in the gridgame.py
    global currentShapeIndex
    steps = (target_idx - currentShapeIndex) % num_shapes
    for _ in range(steps):
        exec_cmd('switchshape')
        # basically, to reach a specific shape index, we repeatedly cycle forward until we eventually land on it.

def set_color(target_idx: int):
    # Cycle forward using 'switchcolor' -> imported from gridgame.py
    global currentColorIndex
    steps = (target_idx - currentColorIndex) % num_colors
    for _ in range(steps):
        exec_cmd('switchcolor')
        # essentially we have implemented the same methodology

def goto_pos(target_xy: tuple[int, int]):
    """ Move brush to (x,y). and also the move depends on the shape and size of the brush """
    target_x, target_y = target_xy
    while shapePos[0] < target_x:
        exec_cmd('right')  # go rightward
    while shapePos[0] > target_x:
        exec_cmd('left')  # go leftward
    while shapePos[1] < target_y:
        exec_cmd('down')  # go downward
    while shapePos[1] > target_y:
        exec_cmd('up')  # go upward
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
        undo_one()  # placedShapes is a list; undo until it's empty to return to the initial state


#  The Hill climbing loop and setting up the parameters

random.seed(42)

# Limits (keep safe for autograder 10 min; you can tune)

TIME_BUDGET_SEC = 2 * 60        # 2 mins tops, since the autograder will run 3 times, so each iteration must finish within 2 mins
MAX_PLATEAU_SIDEWAYS = 150      # allow equal-score moves up to this many
MAX_TRIES_PER_STEP = 175        # the counter for how many random neighbors to sample before declaring "stuck" and we need to move
MAX_RESTARTS = 5                # number of restarts allowed, before we drop everything

best_grid = grid.copy()  # saves the best board state found so far
best_shapes = list(placedShapes)  # this saves the sequence of placed shapes corresponding to that best grid
best_score = objective(grid, shapes_used=len(placedShapes))  # our current best score (less is better)

restarts = 0  # keeps track of how many times we restarted
sideways_used = 0  # similarly this keeps track of the sideways_used

while (time.time() - start) < TIME_BUDGET_SEC:
    # Refresh state
    exec_cmd('export')
    if done:
        used_colors = {int(v) for v in np.unique(grid) if v != -1}
        unique_shape_types = len({shape_index for (shape_index, _, _) in placedShapes})
        print("t=", int(time.time() - start),
              "colors used =", len(used_colors),
              "Times we placed a brush on the board =", len(placedShapes),
              "Unique shapes used =", unique_shape_types,
              "placed=", len(placedShapes),
              "score=", objective(grid, shapes_used=len(placedShapes)),
              "restarts=", restarts,
              "  DONE =", done)
        break  # if the board has been fully colored then we break out of the loop

        #  Just to check that our program is running or not print once every 5 seconds so we don't feel not knowing whether the agent is doing something or not
    if int(time.time() - start) % 5 == 0:
        print("t=", int(time.time() - start),
              "placed=", len(placedShapes),
              "score=", objective(grid, shapes_used=len(placedShapes)),
              "restarts=", restarts)

    current_score = objective(grid, shapes_used=len(placedShapes))
    # This uses the objective function which is responsible for keeping track of the score (lower is better)

    improved = False
    accepted_sideways = False
    # If none of the flags above become true, that means that we could not place anything acceptable so we have to restart

    # First-choice: This keeps sampling random moves, and then accepts the first that improves objective
    for _ in range(MAX_TRIES_PER_STEP):
        anchor_xy = random_empty_cell(grid)
        if anchor_xy is None:
            break  # no cells left to color, meaning we are done

        shape_index = random.randrange(num_shapes)  # picks a random shape from our bucket
        candidate_positions = candidate_positions_covering_anchor(shape_index, anchor_xy)
        # This computes all the anchor positions pos_xy in a way that placing the shape at
        # pos_xy would include the chosen anchor_cell
        if not candidate_positions:
            continue  # if that doesn't happen to be the case then try again

        top_left_xy = random.choice(candidate_positions)  # again going with a random position
        color_index = choose_color_for_shape(grid, shape_index, top_left_xy)
        # this helper function we have tries a few colors and chooses the one that gives us the optimal objective

        # This is the simulation happening before the actual placement
        simulated_grid = apply_shape_to_copy(grid, shape_index, top_left_xy, color_index)
        simulated_score = objective(simulated_grid, shapes_used=len(placedShapes) + 1)

        if simulated_score < current_score:
            # Accept move
            set_shape(shape_index)
            goto_pos(top_left_xy)
            set_color(color_index)
            place_current()
            improved = True
            sideways_used = 0
            break
            # we proceed with this new state of the board

        # Sideways move (optional variant)
        if simulated_score == current_score and sideways_used < MAX_PLATEAU_SIDEWAYS:
            set_shape(shape_index)
            goto_pos(top_left_xy)
            set_color(color_index)
            place_current()
            accepted_sideways = True
            sideways_used += 1
            break
            # in-case we don't get a lower score we keep this one with slight change that is we increase the sideways_used
    # Track best-so-far
    current_score = objective(grid, shapes_used=len(placedShapes))
    if current_score < best_score:
        best_score = current_score
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
# (This is our special move: safety to avoid ending in a worse restart state.)
if not done and best_score < objective(grid, shapes_used=len(placedShapes)):
    restart_to_initial()
    # Replay best shapes through execute (still legal since it was achieved earlier)
    for (shape_index, shape_pos, color_index) in best_shapes:
        set_shape(int(shape_index))
        goto_pos((int(shape_pos[0]), int(shape_pos[1])))
        set_color(int(color_index))
        place_current()

    exec_cmd('export')

########################################
# Do not modify any of the code below.
########################################

end = time.time()

np.savetxt('grid.txt', grid, fmt="%d")
with open("shapes.txt", "w") as outfile:
    outfile.write(str(placedShapes))
with open("time.txt", "w") as outfile:
    outfile.write(str(end-start))
