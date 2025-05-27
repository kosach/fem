import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def draw_mesh(ax, AKT, NT, a_x, a_y, a_z, show_nodes=True, show_node_labels=True, show_outline=True):
    """Малює 3D сітку на наданій осі matplotlib."""
    # ... (код функції як у попередній відповіді) ...
    ax.clear()
    if AKT is None or NT is None or AKT.size == 0 or NT.size == 0:
        ax.text(0.5, 0.5, 0.5, "Сітку не згенеровано", ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("Скінченно-елементна сітка")
        return
    nqp = AKT.shape[1]
    if show_nodes: ax.scatter(AKT[0,:], AKT[1,:], AKT[2,:], c='red', s=30, depthshade=True)
    if show_node_labels:
        for i in range(nqp): ax.text(AKT[0,i], AKT[1,i], AKT[2,i], str(i+1), size=8, color='blue', zorder=10)
    if show_outline:
        corners = np.array([[0,0,0],[a_x,0,0],[a_x,a_y,0],[0,a_y,0],[0,0,a_z],[a_x,0,a_z],[a_x,a_y,a_z],[0,a_y,a_z]])
        faces = [[corners[0],corners[1],corners[2],corners[3]], [corners[4],corners[5],corners[6],corners[7]],
                 [corners[0],corners[1],corners[5],corners[4]], [corners[2],corners[3],corners[7],corners[6]],
                 [corners[1],corners[2],corners[6],corners[5]], [corners[0],corners[3],corners[7],corners[4]]]
        ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', edgecolors='darkgrey', linewidths=0.5, alpha=0.1))
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z"); ax.set_title("Скінченно-елементна сітка")
    all_coords = AKT if AKT is not None else np.array([[0,a_x],[0,a_y],[0,a_z]])
    min_c = all_coords.min(axis=1); max_c = all_coords.max(axis=1)
    center = (max_c + min_c) / 2
    auto_scale_range = (max_c - min_c).max() * 1.1 if (max_c - min_c).max() > 0 else 1.1
    ax.set_xlim(center[0] - auto_scale_range/2, center[0] + auto_scale_range/2)
    ax.set_ylim(center[1] - auto_scale_range/2, center[1] + auto_scale_range/2)
    ax.set_zlim(center[2] - auto_scale_range/2, center[2] + auto_scale_range/2)
    try: ax.set_aspect('equal', adjustable='box')
    except NotImplementedError: ax.set_box_aspect([1,1,1]) # Fallback
    ax.view_init(elev=20, azim=-65)