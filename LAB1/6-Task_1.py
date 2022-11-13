import DR20API
import numpy as np
import heapq
import matplotlib.pyplot as plt

### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.
def Manhattan(pos1, pos2 = [100, 100]):
    '''
    Given two positions, return their Manhattan distance.

    Args:
    pos1, pos2: 2D vector indicating current position and goal position.

    Return:
    An integer that represents the manhattan distance between pos1 and pos2.
    '''
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def Euclidean(pos1, pos2 = [100, 100]):
    '''
    Given two positions, return their Euclidean distance.

    Args:
    pos1, pos2: 2D vector indicating current position and goal position.

    Return:
    An integer that represents the Euclidean distance between pos1 and pos2.
    '''
    return pow(pow(pos1[0] - pos2[0], 2) + pow(pos1[1] - pos2[1], 2), 0.5)


def heuristic(current_pos, distanceFunc=Manhattan):
    return distanceFunc(current_pos)

def next_pos(current_pos, current_map):
    '''
    Given current position and current map, calculate next possible positions.

    Args:
    current_pos -- A 2D vector indicating the current position of the robot.
    current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.

    Return:
    ans: A N*2 array representing next possible positions.
    '''
    ans = []
    x, y = current_pos
    for dirx, diry in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        nx, ny = x + dirx, y + diry
        if 0 <= nx <= 119 and 0 <= ny <= 119 and not current_map[nx][ny]:  # make sure next position is in the map and is traversible.
            ans.append((nx, ny))
    return ans
###  END CODE HERE  ###

def A_star(current_map, current_pos, goal_pos):
    """
    Given current map of the world, current position of the robot and the position of the goal,
    plan a path from current position to the goal using A* algorithm.

    Arguments:
    current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by A* algorithm.
    """

    ### START CODE HERE ###
    startNode = current_pos.copy()
    # record start position for plot.
    current_pos, goal_pos = tuple(current_pos), tuple(goal_pos)
    # convert list to tuple to make positions hashable and enable comparison.
    openLst, closedLst = [], {}
    #maintain an openLst and a closed list (also used for recording parents of each node)
    g = 0
    # accumulated cost from start point
    previous_pos = False
    # the node goes before the node we're exploring, None for the start node

    while not reach_goal(current_pos, goal_pos):
        if current_pos in closedLst:
            _, g, previous_pos, current_pos = heapq.heappop(openLst)
            continue
        closedLst[current_pos] = previous_pos
        g += 1
        # cost from start node accumulates
        for nxt_pos in next_pos(current_pos, current_map):
            if nxt_pos in closedLst:
                continue
            total_cost = g + heuristic(nxt_pos)
            heapq.heappush(openLst, (total_cost, g, current_pos, nxt_pos))
        _, g, previous_pos, current_pos = heapq.heappop(openLst)
    closedLst[current_pos] = previous_pos
    # note that current_pos == goal_pos now!

    path = []
    while current_pos:
        path.append(current_pos)
        current_pos = closedLst[current_pos]
    path = path[::-1]

    # Visualize the map and path.
    # obstacles_x, obstacles_y = [], []
    # for i in range(120):
    #     for j in range(120):
    #         if current_map[i][j]:
    #             obstacles_x.append(i)
    #             obstacles_y.append(j)

    # path_x, path_y = [x[0] for x in path], [x[1] for x in path]
    # plt.plot(path_x, path_y, "-r")
    # plt.plot(startNode[0], startNode[1], "xr")
    # plt.plot(goal_pos[0], goal_pos[1], "xb")
    # plt.plot(obstacles_x, obstacles_y, ".k")
    # plt.grid(True)
    # plt.axis("equal")
    # plt.show()
    ###  END CODE HERE  ###
    return path


def reach_goal(current_pos, goal_pos):
    """
    Given current position of the robot,
    check whether the robot has reached the goal.

    Arguments:
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    is_reached -- A bool variable indicating whether the robot has reached the goal, where True indicating reached.
    """

    ### START CODE HERE ###
    is_reached = Manhattan(current_pos, goal_pos) <= 2
    ###  END CODE HERE  ###
    return is_reached


if __name__ == '__main__':
    # Define goal position of the exploration, shown as the gray block in the scene.
    goal_pos = [100, 100]
    controller = DR20API.Controller()

    # Initialize the position of the robot and the map of the world.
    current_pos = controller.get_robot_pos()
    startNode = current_pos.copy()
    current_map = controller.update_map()
    whole_path = []

    # Plan-Move-Perceive-Update-Replan loop until the robot reaches the goal.
    while not reach_goal(current_pos, goal_pos):
        # Plan a path based on current map from current position of the robot to the goal.
        path = A_star(current_map, current_pos, goal_pos)
        # Move the robot along the path to a certain distance.
        controller.move_robot(path)
        # Get current position of the robot.
        current_pos = controller.get_robot_pos()
        # Update the map based on the current information of laser scanner and get the updated map.
        current_map = controller.update_map()
        while Euclidean(path[-1], current_pos) > 1:
            path.pop()
        # renew whole path
        whole_path += path[1:]

    # Stop the simulation.
    controller.stop_simulation()

    # Visualize the final map and the whole path.
    obstacles_x, obstacles_y = [], []
    for i in range(120):
        for j in range(120):
            if current_map[i][j]:
                obstacles_x.append(i)
                obstacles_y.append(j)

    path_x, path_y = [x[0] for x in whole_path], [x[1] for x in whole_path]
    plt.plot(path_x, path_y, "-r")
    plt.plot(startNode[0], startNode[1], "xr")
    plt.plot(goal_pos[0], goal_pos[1], "xb")
    plt.plot(obstacles_x, obstacles_y, ".k")
    plt.grid(True)
    plt.axis("equal")
    plt.show()