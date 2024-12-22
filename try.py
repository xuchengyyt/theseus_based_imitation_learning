import numpy as np
import matplotlib.pyplot as plt

def bezier_curve(p0, p1, p2, p3, t):
    """
    计算三次贝塞尔曲线上的点
    p0, p1, p2, p3 是控制点
    t 是参数，范围在 [0, 1]
    """
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

def bezier_curve_points(p0, p1, p2, p3, num_points=50):
    """
    生成三次贝塞尔曲线上的多个点
    p0, p1, p2, p3 是控制点
    num_points 是生成的点的数量
    """
    t_values = np.linspace(0, 1, num_points)
    curve_points = np.zeros((num_points, 2))
    
    for i, t in enumerate(t_values):
        curve_points[i] = bezier_curve(p0, p1, p2, p3, t)
    
    return curve_points

# 定义控制点
p0 = np.array([0, 0])
p1 = np.array([10, 0])
p2 = np.array([10, 3])
p3 = np.array([20, 3])

# 生成贝塞尔曲线上的点
curve_points = bezier_curve_points(p0, p1, p2, p3)
for i in range(49):
    dx = (curve_points[i + 1, 0] - curve_points[i , 0] ) * (curve_points[i + 1, 0] - curve_points[i , 0] ) + (curve_points[i + 1, 1] - curve_points[i, 1] ) * (curve_points[i + 1, 1] - curve_points[i, 1] ) 
    print(np.sqrt(dx))
# 可视化
plt.figure(figsize=(8, 6))
plt.plot(curve_points[:, 0], curve_points[:, 1], label='Bezier Curve')
plt.scatter(curve_points[:, 0], curve_points[:, 1])
# plt.scatter([p0[0], p1[0], p2[0], p3[0]], [p0[1], p1[1], p2[1], p3[1]], color='red', label='Control Points')
plt.title('Bezier Curve Path Transition')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()