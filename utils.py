import numpy as np
import matplotlib.pyplot as plt


def plotNodeTopo(sList, gwList):
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title("Node topology")
    sx = [s.x for s in sList]
    sy = [s.y for s in sList]
    ax.scatter(sx, sy, c='r', label='node')
    gwx = []
    gwy = []
    for gw in gwList:
        gwx.append(gw.x)
        gwy.append(gw.y)
        radius = plt.Circle((gw.x, gw.y), gw.R, color='g', fill=False)
        ax.add_patch(radius)
    ax.scatter(gwx, gwy, c='b', label='gw')
    ax.legend()
    plt.xlim([-40, 40])
    plt.ylim([-40, 40])
    plt.show()