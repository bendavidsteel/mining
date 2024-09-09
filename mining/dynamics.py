import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def flow_pairplot(timestamps, means, separate=False):
    num_users, num_timesteps, num_opinions = means.shape
    if separate:
        figs, axes = np.zeros((num_opinions, num_opinions), dtype=object), np.zeros((num_opinions, num_opinions), dtype=object)
        for i in range(num_opinions):
            for j in range(num_opinions):
                figs[i, j], axes[i, j] = plt.subplots(figsize=(6, 6))
    else:
        figs, axes = plt.subplots(nrows=num_opinions, ncols=num_opinions, figsize=(12, 12))
    for i in range(num_opinions):
        for j in range(num_opinions):
            ax = axes[i, j]
            if separate:
                fig = figs[i, j]
            else:
                fig = figs
            if i == j:
                for k in range(num_users):
                    ax.plot(timestamps, means[k,:,i], alpha=0.2)

            if i < j:
                for k in range(num_users):
                    ax.plot(means[k,:,i], means[k,:,j])

            if i > j:
                # Implement vector field analysis
                # Create a grid
                x = np.linspace(-3, 3, 20)
                y = np.linspace(-3, 3, 20)
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
                            mask = ((x_data >= X[m,n]-0.3) & (x_data < X[m,n]+0.3) & 
                                    (y_data >= Y[m,n]-0.3) & (y_data < Y[m,n]+0.3) &
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
                ax.set_xlim(-3, 3)
                ax.set_ylim(-3, 3)

            if separate:
                if i == j:
                    ax.set_ylabel('User')
                    ax.set_xlabel('Time')
                    ax.set_title(f'Opinion {i+1}')

                else:
                    ax.set_ylabel(f'Opinion {j+1}')
                    ax.set_xlabel(f'Opinion {i+1}')
            else:
                if j == 0:
                    ax.set_ylabel(f'Opinion {i+1}')

                if i == num_opinions - 1:
                    ax.set_xlabel(f'Opinion {i+1}')

    return figs, axes
