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

game = ShapePlacementGrid(GUI=False, render_delay_sec=0, gs=6, num_colored_boxes=5)
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

"""
    So apparently, the gridgame.py is our environment/helper layer.
    It tells us about the shapes available and provides us with
    canPlace(), getAvailabeColor() functions which help us check for bounds,
    overlapping, giving us a color for a cell without conflict.
    Our code is intended to work alongside this without making any changes to  
    the original gridgame.py file.

"""




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

    # Goal: To get the number of conflicts, desired outcome is always 0

    conflict_count = 0
    n = grid_array.shape[0]

    for y in range(n):          # y = row index
        for x in range(n):      # x = column index
            cell_color = grid_array[y, x]  # current cell’s color
            if cell_color == -1:           # -1 means "empty", skip it
                continue

            # Check RIGHT neighbor only (so we don't double-count)
            if x + 1 < n and grid_array[y, x + 1] == cell_color:
                conflict_count += 1

            # Check DOWN neighbor only (so we don't double-count)
            if y + 1 < n and grid_array[y + 1, x] == cell_color:
                conflict_count += 1

    return conflict_count  # bigger number = more clashes; 0 means perfect and our desired number


def objective(grid_array: np.ndarray, shapes_used: int) -> int:

    # Goal: to compute a single `badness score` for the present grid state (lower is better)

    # Priority order shown below:

    """
        1. legal state: conflicts must go to 0
        2. completion state: fill empty cells
        3. simplicity constraint: use fewer distinct colors
        4. efficiency metric: use fewer shape placements
    """

    conflicts = count_conflicts(grid_array)  # 0 = legal and >0 = illegal -> (adjacent same colors)
    empty_cells = int(np.sum(grid_array == -1))  # count of unfilled cells (-1 means empty)

    # Unique non-empty colors currently present in the grid

    colors_used = {int(v) for v in np.unique(grid_array) if v != -1}
    num_colors_used = len(colors_used)

    # Weighted penalty: correctness > completion > colors > shapes
    # We want the agent to minimize this score at every step.
    return (
            100_000 * conflicts +
            1_000 * empty_cells +
            10 * num_colors_used +
            shapes_used
    )



# Below are agent helpers: they simulate moves on a COPY of the grid (no execute calls / no env mutation)

def apply_shape_to_copy(
    grid_array: np.ndarray,
    shape_index: int,
    top_left_xy: tuple[int, int],
    color_index: int
) -> np.ndarray:

    # Goal: Simulate the placement of a shape on a COPY of the latest stable grid.

    """
    1. Assumes the placement is already legal e.g. - in-bounds and no overlapping.
    2. Original grid_array remains unaffected unchanged.
    """

    top_left_x, top_left_y = top_left_xy  # (x,y) = (col,row) top-left placement position
    new_grid = grid_array.copy()

    shape_mask = game.shapes[shape_index]  # 2D mask: True/1 means "paint this cell" False/0 means brush cannot reach this cell
    for i, row in enumerate(shape_mask):   # i = row inside the shape mask
        for j, cell in enumerate(row):     # j = col inside the shape mask
            if cell:
                new_grid[top_left_y + i, top_left_x + j] = color_index

    return new_grid  # simulated grid after painting this shape


def random_empty_cell(grid_array: np.ndarray) -> tuple[int, int] | None:


    # Goal: Return a random empty cell for coloring

    """
    1. Returns a random empty cell as (x, y), where empty means grid value == -1.
    2. Returns None if the grid has no empty cells. Other-wise implying teh agent has finished it's work.
    """
    empty_positions = np.argwhere(grid_array == -1)  # returns (row, col) pairs = (y, x)
    if len(empty_positions) == 0:
        return None

    y, x = empty_positions[random.randrange(len(empty_positions))]
    return int(x), int(y)  # convert numpy ints -> normal ints, and return in (x,y) order


def candidate_positions_covering_anchor(
    grid_array: np.ndarray,
    shape_index: int,
    anchor_xy: tuple[int, int]
) -> list[tuple[int, int]]:

    # Goal: To find all valid top-left placements of a shape that would cover our given anchor cell(the cell we are trying to color).

    """
    Given an anchor cell (x,y) that we want to cover, return all the LEGAL top-left (x,y)
    positions where this shape would cover that anchor cell.
    """
    anchor_x, anchor_y = anchor_xy
    shape_mask = game.shapes[shape_index]

    candidate_positions: list[tuple[int, int]] = []

    for i, row in enumerate(shape_mask):      # i = row inside shape
        for j, cell in enumerate(row):        # j = col inside shape
            if not cell:
                continue  # only meaningful where the shape actually paints

            # If mask cell (j,i) should land on (anchor_x, anchor_y),
            # then top-left must be shifted back by (j,i).
            top_left_x = anchor_x - j
            top_left_y = anchor_y - i

            if top_left_x < 0 or top_left_y < 0:
                continue  # shape out of bounds -> reject this immediately

            # Keep only placements that are legal in the environment (within bounds and no overlapping)

            if game.canPlace(grid_array, shape_mask, [top_left_x, top_left_y]):
                candidate_positions.append((top_left_x, top_left_y))

    return candidate_positions



def choose_color_for_shape(
        grid_array: np.ndarray,
        shape_index: int,
        top_left_xy: tuple[int, int]
) -> int:

    # Goal: To choose a color for this shape placement that MINIMIZES objective().

    """
        Approach Style:

        1. Trying out ALL the available colors (since there are only 4).
        2. Simulating placing the shape using each color on a COPY of the grid (No alterations will be made until we have the relatively best color)
        3. Picking the color that gives the lowest objective score (least "badness").

    """

    best_color = 0

    #  We start with a huge value so the FIRST candidate always becomes our baseline "best".
    #  If we started at 0, nothing would beat it (since objective scores are non-negative),
    #  and we'd never update best_color properly, actually the agent would just freeze.

    best_score = 10 ** 18

    # Trying out every possible color

    for color_candidate in range(num_colors):
        simulated_grid = apply_shape_to_copy(
            grid_array, shape_index, top_left_xy, color_candidate
        )

        # shapes_used = len(placedShapes) + 1 because this simulation represents the grid
        # AFTER placing ONE more shape (the move we're currently testing out).

        candidate_score = objective(
            simulated_grid, shapes_used=len(placedShapes) + 1
        )

        # Keep the color that produces the least "bad" grid.

        if candidate_score < best_score:
            best_score = candidate_score
            best_color = color_candidate

    return best_color  # best color for THIS placement, according to objective()



#  Execute-only interaction wrappers, all in accordance to the gridgame.py

def exec_cmd(cmd: str):
    global shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done
    shapePos, currentShapeIndex, currentColorIndex, grid, placedShapes, done = game.execute(cmd)

    # This one sends one command to the environment (game.execute(cmd)),
    # and then it updates our local variables whatever the environment yields
    # the variables above are global, and we update them everywhere

def set_shape(target_idx: int):

    # Goal: To get the relatively better shape

    # We cycle forward using 'switchshape' -> This is found in the gridgame.py

    global currentShapeIndex
    steps = (target_idx - currentShapeIndex) % num_shapes
    for _ in range(steps):
        exec_cmd('switchshape')

        # basically, to reach a specific shape index, we repeatedly cycle forward until we eventually land on it.
        # And keep harmony with the implementation of switchshape we aren't supposed substract the index of the shapes
        # instead we use the modulo operator for handling the smooth switching.

def set_color(target_idx: int):

    # Goal: To get to the relatively correct color

    # Cycle forward using 'switchcolor' -> imported from gridgame.py

    global currentColorIndex
    steps = (target_idx - currentColorIndex) % num_colors
    for _ in range(steps):
        exec_cmd('switchcolor')

        # essentially we have implemented the same methodology

def goto_pos(target_xy: tuple[int, int]):

    # Goal: To be able to move the brush position

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

# Limits (Since the autograder has a time limit of 10 mins, I decided to set the time_budget to 2 mins as mentioned in the requirements the agent will start from 3 various positions, so essentially 3 times full iteration)

TIME_BUDGET_SEC = 2 * 60        # 2 mins tops, since the autograder will run 3 times, so each iteration must finish within 2 mins
MAX_PLATEAU_SIDEWAYS = 150      # allow equal-score moves up to this many
MAX_TRIES_PER_STEP = 175        # the counter for how many random neighbors to sample before declaring "stuck" and we need to move
MAX_RESTARTS = 5                # number of restarts allowed by the agent.

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
              "colors_used =", len(used_colors),
              "Times we placed a brush on the board =", len(placedShapes),
              "Unique shapes used =", unique_shape_types,
              "final_score =", objective(grid, shapes_used=len(placedShapes)),
              "number_of_restarts =", restarts,
              "  DONE =", done)
        break  # if the board has been fully colored then we break out of the loop

        #  Just to check that our program is running or not print once every 5 seconds
        #  so we don't feel not knowing whether the agent is doing something or not

    if int(time.time() - start) % 5 == 0:
        print("t=", int(time.time() - start),
              "placed=", len(placedShapes),
              "score=", objective(grid, shapes_used=len(placedShapes)),
              "restarts=", restarts)

    current_score = objective(grid, shapes_used=len(placedShapes))
    # This uses the objective function which is responsible for keeping track of the score (lower is better)

    improved = False
    accepted_sideways = False
    # If none of the flags above become true, that means that we failed to place anything acceptable,
    # meaning we are going to have to restart

    # First-choice: This keeps sampling random moves, and then accepts the first that improves objective (lowers the score)
    for _ in range(MAX_TRIES_PER_STEP):
        anchor_xy = random_empty_cell(grid)
        if anchor_xy is None:
            break  # no cells left to color, meaning we are done

        shape_index = random.randrange(num_shapes)  # picks a random shape from our bucket
        candidate_positions = candidate_positions_covering_anchor(grid, shape_index, anchor_xy)

        # This computes all the anchor positions pos_xy in a way that placing the shape at
        # pos_xy would include the chosen anchor_cell

        if not candidate_positions:
            continue  # if that doesn't happen to be the case then try again

        top_left_xy = random.choice(candidate_positions)  # again going with a random position
        color_index = choose_color_for_shape(grid, shape_index, top_left_xy)

        # this helper function we have, tries a few colors and chooses the one that gives us the optimal objective

        # This is the simulation taking place before the actual placement

        simulated_grid = apply_shape_to_copy(grid, shape_index, top_left_xy, color_index)
        simulated_score = objective(simulated_grid, shapes_used=len(placedShapes) + 1)

        if simulated_score < current_score:
            # Accepting the move

            set_shape(shape_index)
            goto_pos(top_left_xy)
            set_color(color_index)
            place_current()
            improved = True
            sideways_used = 0
            break

            # we proceed with this new state of the board

        # Sideways move, we can change it as long as we don't exceed the max count for finding the optimal solution

        if simulated_score == current_score and sideways_used < MAX_PLATEAU_SIDEWAYS:
            set_shape(shape_index)
            goto_pos(top_left_xy)
            set_color(color_index)
            place_current()
            accepted_sideways = True
            sideways_used += 1
            break

            # in-case we don't get a lower score we keep this one with slight change that is we increase the sideways_used

    # Tracking the best so far

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


# The End: exporting the final solution state

exec_cmd('export')

# Considering that we failed to finish, revert back to best-so-far (by undoing and replaying best_shapes)
# (This is our special move: safety to avoid ending in a worse restart state.)

if not done and best_score < objective(grid, shapes_used=len(placedShapes)):
    restart_to_initial()
    # This replays the best shapes through execute
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
