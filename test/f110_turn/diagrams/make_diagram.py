import matplotlib.pyplot as plt
import numpy as np

SQRT3 = np.sqrt(3)

def sharp_right_plot():
    fig, ax = plt.subplots()
    ax.plot(
        [-1.5, -1.5, 3 - 1.5 * SQRT3, 2],
        [-6.5, 1.5, 1.5, 1 / SQRT3],
        color='black',
    )
    ax.plot(
        [1.5, 1.5, 2,],
        [-6.5, -1.5 * SQRT3, -5 / SQRT3],
        color='black',
    )
    ax.fill(
        [1.5, 1.5, 2, 2],
        [-1.5 * SQRT3, 0.5 * SQRT3, 1 / SQRT3, -5 / SQRT3],
        color='green',
    )
    ax.fill(
        [-0.2, -0.2, 0.2, 0.2],
        [-6.2, -5.8, -5.8, -6.2],
        color='blue',
    )
    ax.arrow(
        1.75, -1.0,
        0.5, 0.5,
        head_width=0.1,
        color='red',
    )
    ax.arrow(
        1.75, -1.0,
        0.5, -0.5,
        head_width=0.1,
        color='black',
    )
    ax.set_aspect('equal')
    ax.set_title('sharp right')
    fig.savefig('sharp_right.pdf')

if __name__ == '__main__':
    sharp_right_plot()
