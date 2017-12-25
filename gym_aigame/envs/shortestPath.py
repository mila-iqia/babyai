# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:44:04 2017

@author: simon
"""


from collections import deque
#
#
#maze=[
#
#    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#
#    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#
#    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
#
#    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#
#    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
#
#    [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
#
#    [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
#
#    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
#
#    [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
#
#    [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
#
#    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
#
#    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


def maze2graph(maze):
    height = len(maze)
    width = len(maze[0]) if height else 0
    graph = {(i, j): [] for j in range(width) for i in range(height) if not maze[i][j]}
    for row, col in graph.keys():
        if row < height - 1 and not maze[row + 1][col]:
            graph[(row, col)].append(("S", (row + 1, col)))
            graph[(row + 1, col)].append(("N", (row, col)))
        if col < width - 1 and not maze[row][col + 1]:
            graph[(row, col)].append(("E", (row, col + 1)))
            graph[(row, col + 1)].append(("W", (row, col)))
    return graph



def find_path_bfs(maze,start,goal):
    queue = deque([("", start)])
    visited = set()
    graph = maze2graph(maze)
    while queue:
        path, current = queue.popleft()
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            queue.append((path + direction, neighbour))
    return "NO WAY!"




