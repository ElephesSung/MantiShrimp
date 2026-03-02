import matplotlib as mpl
from IPython.display import HTML, display
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
from IPython.display import HTML
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase

def draw_box(ax, box, **kwargs):
    """Draw rectangle boundaries (dashed box) on ax."""
    xmin, xmax, ymin, ymax = box
    ax.plot([xmin, xmax], [ymin, ymin], **kwargs)
    ax.plot([xmin, xmax], [ymax, ymax], **kwargs)
    ax.plot([xmin, xmin], [ymin, ymax], **kwargs)
    ax.plot([xmax, xmax], [ymin, ymax], **kwargs)

def periodic_traj_segment(traj, box):
    """
    For a 2D trajectory (N,2), returns a list of line segments
    that do NOT cross the periodic boundary visually.
    """
    xmin, xmax, ymin, ymax = box
    dx = xmax - xmin
    dy = ymax - ymin
    pts = np.asarray(traj)
    if len(pts) < 2:
        return [pts]
    segments = []
    seg = [pts[0]]
    for i in range(1, len(pts)):
        prev = pts[i-1]
        curr = pts[i]
        # If a jump is large (crossed box), break the segment
        if np.abs(curr[0]-prev[0]) > 0.5*dx or np.abs(curr[1]-prev[1]) > 0.5*dy:
            segments.append(np.array(seg))
            seg = [curr]
        else:
            seg.append(curr)
    if len(seg):
        segments.append(np.array(seg))
    return segments

def animate_KT(
    cell_history_df,
    killer_positions,
    target_positions,
    killer_radius=10.0,
    target_radius=12.0,
    dead_target_radius=4.0,
    *,
    killer_colour="navy",
    live_target_colour="green",
    dead_target_colour="salmon",
    traj_steps=10,
    figsize=(5, 5),
    box=(-600, 600, -600, 600),
    interval=10,
    boundary_condition='confined',
    ax_box=None,
    add_swatch_legend=True, legend_labels=None, fig_dpi=None, save_gif_path=None,
    frame_step=1, gif_fps=None, render_html=True
):
    """
    Animate Killer-Target cell simulation, showing box and periodic/reflecting trajectories.
    box: simulation box [xmin, xmax, ymin, ymax]
    ax_box: axis/plot box [xmin, xmax, ymin, ymax] (optional, can be bigger than sim box)
    """
    if ax_box is None:
        ax_box = box

    steps = int(cell_history_df['step'].max()) + 1
    frame_step = max(1, int(frame_step))
    interval = max(1, int(round(interval)))
    N_KILLER = (cell_history_df['cell_type'] == 'killer').sum() // steps
    N_TARGET = (cell_history_df['cell_type'] == 'target').sum() // steps

    killer_traj = np.array(killer_positions)
    target_traj = np.array(target_positions)

    legend_elements = [
        Patch(facecolor=killer_colour, edgecolor='black', label='Killer'),
        Patch(facecolor=live_target_colour, edgecolor='black', label='Target (Alive)'),
        Patch(facecolor=dead_target_colour, edgecolor='black', label='Target (Dead)')
    ]

    fig, ax = plt.subplots(figsize=figsize, dpi = fig_dpi)
    ax.set_xlim(ax_box[0], ax_box[1])
    ax.set_ylim(ax_box[2], ax_box[3])
    ax.set_aspect('equal')

    # --- Draw dashed simulation box boundary ---
    draw_box(ax, box, color='gray', linestyle='dashed', linewidth=1)
    
    # ---- Swatch bar legend ----
    if add_swatch_legend:
        # Use default legend_labels if not given
        if legend_labels is None:
            legend_labels = [
                'Killer',
                'Target (Alive)',
                'Target (Dead)'
            ]
        # Colors for the legend (order must match the labels)
        colors = [killer_colour, live_target_colour, dead_target_colour]
        cmap = ListedColormap(colors)
        bounds = [0, 1, 2, 3]  # n_colors+1
        norm = plt.Normalize(0, 3)
        
        # Use axes_grid1 to add colorbar axis
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = ColorbarBase(
            cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0.5, 1.5, 2.5], spacing='proportional', orientation='vertical'
        )
        cb.ax.set_yticklabels(legend_labels)
        cb.set_label("Cell Type")
        cb.ax.tick_params(size=0)

    # --- Cell circles ---
    killer_circles = [Circle((0,0), radius=killer_radius, color=killer_colour, ec='black', lw=1, alpha=1) for _ in range(N_KILLER)]
    target_circles = [Circle((0,0), radius=target_radius, color=live_target_colour, ec='black', lw=1, alpha=1) for _ in range(N_TARGET)]
    dead_target_circles = [Circle((0,0), radius=dead_target_radius, color=dead_target_colour, ec='black', lw=1, alpha=1, visible=False) for _ in range(N_TARGET)]
    for c in killer_circles + target_circles + dead_target_circles:
        ax.add_patch(c)

    # --- Trajectory lines (only last N steps) ---
    killer_lines = [ [] for _ in range(N_KILLER) ]
    target_lines = [ [] for _ in range(N_TARGET) ]
    for i in range(N_KILLER):
        l, = ax.plot([], [], color=killer_colour, lw=0.7, alpha=0.7)
        killer_lines[i].append(l)
    for j in range(N_TARGET):
        l, = ax.plot([], [], color=live_target_colour, lw=0.7, alpha=0.7)
        target_lines[j].append(l)

    # ax.legend(handles=legend_elements, loc="upper right")

    def animate(step):
        traj_start = max(0, step - traj_steps + 1)
        # Remove previous lines (only for periodic, to allow segments)
        if boundary_condition == "periodic":
            for group in killer_lines + target_lines:
                for l in group[1:]:
                    l.remove()
                group[:] = group[:1]

        # Trajectories
        for i in range(N_KILLER):
            xy = killer_traj[traj_start:step+1, i, :]
            if boundary_condition == "periodic":
                for seg in periodic_traj_segment(xy, box):
                    line, = ax.plot(seg[:, 0], seg[:, 1], color=killer_colour, lw=0.7, alpha=0.7)
                    killer_lines[i].append(line)
                killer_lines[i][0].set_data([], [])
            else:
                killer_lines[i][0].set_data(xy[:, 0], xy[:, 1])
        for j in range(N_TARGET):
            xy = target_traj[traj_start:step+1, j, :]
            if boundary_condition == "periodic":
                for seg in periodic_traj_segment(xy, box):
                    line, = ax.plot(seg[:, 0], seg[:, 1], color=live_target_colour, lw=0.7, alpha=0.7)
                    target_lines[j].append(line)
                target_lines[j][0].set_data([], [])
            else:
                target_lines[j][0].set_data(xy[:, 0], xy[:, 1])

        # Circles (alive/dead)
        this_df = cell_history_df[cell_history_df['step'] == step]
        for i, row in this_df[this_df['cell_type']=='killer'].iterrows():
            killer_circles[int(row['cell_id'])].center = (row['x'], row['y'])
        for j, row in this_df[this_df['cell_type']=='target'].iterrows():
            idx = int(row['cell_id'])
            if row['alive_status']:
                target_circles[idx].center = (row['x'], row['y'])
                target_circles[idx].set_visible(True)
                dead_target_circles[idx].set_visible(False)
            else:
                dead_target_circles[idx].center = (row['x'], row['y'])
                dead_target_circles[idx].set_visible(True)
                target_circles[idx].set_visible(False)
        ax.set_title(f"Step {step}  |  Time {this_df['time'].iloc[0]:.2f}")
        return (sum(killer_lines, []) + sum(target_lines, []) +
                killer_circles + target_circles + dead_target_circles)

    ani = animation.FuncAnimation(
        fig, animate, frames=range(0, steps, frame_step), interval=interval, blit=True
    )

    if save_gif_path is not None:
        try:
            from matplotlib.animation import PillowWriter
            fps = int(gif_fps) if gif_fps is not None else max(1, int(round(1000 / interval)))
            ani.save(save_gif_path, writer=PillowWriter(fps=fps))
            print(f"Animation saved as GIF: {save_gif_path}")
        except Exception as e:
            print(f"Failed to save GIF: {e}")

    plt.close(fig)
    if render_html:
        display(HTML(ani.to_jshtml()))
    return ani