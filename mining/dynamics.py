import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def flow_pairplot(timestamps, means, opinion_names, do_pairplot=True):
    num_users, num_timesteps, num_opinions = means.shape
    min_timestamp = np.min(num_timesteps)
    max_timestamp = np.max(num_timesteps)
    if do_pairplot:
        figs, axes = plt.subplots(nrows=num_opinions, ncols=num_opinions, figsize=(2.5 * num_opinions, 2.5 * num_opinions))
        figs.tight_layout()
    else:
        figs = np.zeros((num_opinions, num_opinions), dtype=object)
        axes = np.zeros((num_opinions, num_opinions), dtype=object)
        for i in range(num_opinions):
            for j in range(num_opinions):
                figs[i, j], axes[i, j] = plt.subplots(figsize=(4, 4))
                figs[i, j].tight_layout()
    
    for i in range(num_opinions):
        for j in range(num_opinions):
            if do_pairplot:
                fig = figs
            else:
                fig = figs[i, j]

            ax = axes[i, j]
            if i == j:
                ax.set_xlim([min_timestamp, max_timestamp])
                ax.set_ylim([-1, 1])
                for k in range(num_users):
                    ax.plot(timestamps, means[k,:,i], alpha=0.2)

            if i < j:
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                for k in range(num_users):
                    ax.plot(means[k,:,i], means[k,:,j])

            if i > j:
                # Implement vector field analysis
                # Create a grid
                grid_step = 0.1
                x = np.linspace(-1, 1, 20)
                y = np.linspace(-1, 1, 20)
                X, Y = np.meshgrid(x, y)
                
                # Calculate the average change in x and y directions
                dx = np.zeros_like(X)
                dy = np.zeros_like(Y)
                
                for k in range(num_users):
                    x_data = means[k, :-1, i]
                    y_data = means[k, :-1, j]
                    x_next = means[k, 1:, i]
                    y_next = means[k, 1:, j]
                    
                    for m in range(X.shape[0]):
                        for n in range(X.shape[1]):
                            mask = ((x_data >= X[m,n]-grid_step) & (x_data < X[m,n]+grid_step) & 
                                    (y_data >= Y[m,n]-grid_step) & (y_data < Y[m,n]+grid_step) &
                                    ~np.isnan(x_next) & ~np.isnan(y_next))
                            if np.sum(mask) > 0:
                                dx[m,n] += np.mean(x_next[mask] - x_data[mask])
                                dy[m,n] += np.mean(y_next[mask] - y_data[mask])
                
                # Normalize by max vector
                magnitude = np.sqrt(dx**2 + dy**2)
                max_magnitude = np.max(magnitude)
                dx /= max_magnitude
                dy /= max_magnitude

                norm = matplotlib.colors.Normalize()
                norm.autoscale(magnitude)
                cm = matplotlib.cm.viridis

                sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
                sm.set_array([])
                
                # Plot vector field
                ax.quiver(X, Y, dx, dy, magnitude, scale=25, width=0.004)
                fig.colorbar(sm, ax=ax)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

            if j == 0:
                ax.set_ylabel(opinion_names[i])

            if i == num_opinions - 1:
                ax.set_xlabel(opinion_names[j])

    return figs, axes
