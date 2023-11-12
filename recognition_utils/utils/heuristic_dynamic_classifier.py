import numpy as np 
from collections import Counter 
from math import sqrt 
def is_line(coords, tolerance=0.01, percentage=0.8): 
    # 将输入列表拆分为 x 坐标和 y 坐标 
    x_coords = coords[::2] 
    y_coords = coords[1::2] 
    # 使用 numpy 的 polyfit 函数进行线性回归，获取斜率和截距 
    slope0, intercept0 = np.polyfit(x_coords, y_coords, 1) 
    slope1, intercept1 = np.polyfit(y_coords, x_coords, 1) 
    # 计算每个点应该在的 y 值，与实际的 y 值进行比较，看看是否在容忍范围内
    return sum(abs(y - (slope0 * x + intercept0)) <= tolerance for x, y in 
               zip(x_coords, y_coords)) / len(x_coords) >= percentage or \
            sum(abs(x - (slope1 * y + intercept1)) <= tolerance for x, y in 
               zip(x_coords, y_coords)) / len(x_coords) >= percentage
def determine_direction(coords): 
    # 将输入列表拆分为 x 坐标和 y 坐标 
    x_coords = coords[::2] 
    y_coords = coords[1::2] 
    # 计算每一对相邻点的移动方向 
    directions = ['Right' if abs(x2 - x1) > abs(y2 - y1) and x2 > x1 else 
                  'Left' if abs(x2 - x1) > abs(y2 - y1) and x2 < x1 else 
                  'Down' if abs(y2 - y1) > abs(x2 - x1) and y2 > y1 else 
                  'Up' if abs(y2 - y1) > abs(x2 - x1) and y2 < y1 else 
                  'Random Line' for x1, x2, y1, y2 in zip(x_coords[:-1], x_coords[1:], \
                  y_coords[:-1], y_coords[1:])] 
    # 选取最常见的移动方向
    most_common_direction = Counter(directions).most_common(1)[0][0] 
    return most_common_direction if most_common_direction != 'None' else 'No clear direction' 
def determine_clockwise(coords, distance_threshold=0.01): 
    # 将输入列表拆分为 x 坐标和 y 坐标 
    x_coords = coords[::2] 
    y_coords = coords[1::2] 
    # 计算所有相邻的三个点的叉积，只有当相邻点之间的距离大于阈值时才进行计算 
    cross_products = [(x_coords[i+1] - x_coords[i]) * (y_coords[i+2] - y_coords[i]) - (y_coords[i+1] - y_coords[i]) 
                      * (x_coords[i+2] - x_coords[i]) for i in range(len(x_coords) - 2) 
                      if sqrt((x_coords[i+1] - x_coords[i])**2 + (y_coords[i+1] - y_coords[i])**2) > distance_threshold] 
    # 如果大多数叉积为正，则点是逆时针移动的；否则，点是顺时针移动的 
    return 'Clockwise' if sum(cp > 0 for cp in cross_products) > sum(cp < 0 for cp in cross_products) else 'Counter-Clockwise' 
def is_mild_curvature(coords, curvature_threshold=7.0): 
    # 将输入列表拆分为 x 坐标和 y 坐标 
    x_coords = coords[::2] 
    y_coords = coords[1::2] 
    # 计算所有相邻三个点的曲率 
    curvatures = [] 
    for i in range(len(x_coords) - 2): 
        dx1, dx2 = x_coords[i+1] - x_coords[i], x_coords[i+2] - x_coords[i+1] 
        dy1, dy2 = y_coords[i+1] - y_coords[i], y_coords[i+2] - y_coords[i+1] 
        curvature = abs(dx1*dy2 - dx2*dy1) / ((dx1**2 + dy1**2) * sqrt(dx2**2 + dy2**2) + 1e-10) 
        curvatures.append(curvature) 
    # 判断大部分曲率是否小于阈值 
    return sum(curvature < curvature_threshold for curvature in curvatures) / len(curvatures) > 0.4 

def dynamic_classify(coords):
    if len(coords) % 2 != 0:
        coords = coords[:-1]
    if is_line(coords, 0.01, 0.8): 
        return determine_direction(coords)  
    elif is_mild_curvature(coords): 
        return determine_clockwise(coords)

    return 'Nothing Detected'