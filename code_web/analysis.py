import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def graph(time_seconds, areas, smoke_dir, points):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.3, wspace=0.3)

    # First subplot: Cumulative Area over Time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_seconds, areas, label='Burned Area', color='firebrick')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Cumulative Area (pixel units)')
    ax1.set_title('Cumulative Burned Area Over Time')
    ax1.grid(True)
    ax1.legend()

    # Second subplot: Change in Area over Time
    area_growth = [areas[i+2] - areas[i+1] for i in range(len(areas) - 2)]
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_seconds[2:], area_growth, label='Area Change', color='orangered')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Change in Area (pixel units)')
    ax2.set_title('Rate of Area Change Over Time')
    ax2.grid(True)
    ax2.legend()

    # Third subplot: Smoke Direction
    ax3 = fig.add_subplot(gs[0, 1], polar=True)
    ax3.hist(np.radians(smoke_dir), bins=30, color='steelblue', alpha=0.7)
    ax3.set_title('Smoke Dispersion Direction')
    ax3.set_theta_direction(-1)
    ax3.set_rticks([]) 

    # Fourth subplot: Fire Spread Path
    ax4 = fig.add_subplot(gs[1, 1])
    x, y = zip(*points)
    ax4.plot(x, y, marker='o', color='darkorange', markersize=5, linestyle='-', label='Fire Path')
    ax4.invert_yaxis()

    # Add star markers for the start and end points
    ax4.plot(x[0], y[0], marker='*', color='limegreen', markersize=12, label='Ignition Point')
    ax4.plot(x[-1], y[-1], marker='*', color='red', markersize=12, label='Current Fire Front')

    ax4.set_xlabel('X Coordinate')
    ax4.set_ylabel('Y Coordinate')
    ax4.set_title('Fire Spread Path')
    ax4.grid(True)
    ax4.legend()

    # plt.tight_layout()
    # plt.savefig('forest_fire_analysis1.png', dpi=300, bbox_inches='tight')

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)

    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

    return img_str
