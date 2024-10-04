from datetime import datetime
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

def flow_pairplot(timestamps, means, confidence_intervals, opinion_names, do_pairplot=True, save=False, save_path=None, significance_threshold=0.01):
    num_users, num_timesteps, num_opinions = means.shape
    
    if do_pairplot:
        figs, axes = plt.subplots(nrows=num_opinions, ncols=num_opinions, figsize=(2.5 * num_opinions, 2.5 * num_opinions))
    else:
        figs = np.empty((num_opinions, num_opinions), dtype=object)
        axes = np.empty((num_opinions, num_opinions), dtype=object)
        for i in range(num_opinions):
            for j in range(num_opinions):
                figs[i, j], axes[i, j] = plt.subplots(figsize=(4, 4))
    
    # Create separate figures for 1D flow field plots
    flow_figs = np.empty(num_opinions, dtype=object)
    flow_axes = np.empty(num_opinions, dtype=object)
    for i in range(num_opinions):
        flow_figs[i], flow_axes[i] = plt.subplots(figsize=(6, 4))
    
    cmap = plt.get_cmap('viridis')

    for i in range(num_opinions):
        for j in range(num_opinions):
            print(f'Plotting {opinion_names[i]} vs {opinion_names[j]}')
            ax = axes[i, j] if do_pairplot else axes[i, j]
            fig = figs if do_pairplot else figs[i, j]

            if i == j:
                ax.set_ylim([-1, 1])
                
                # Initialize arrays for 1D flow field
                x = np.linspace(-1, 1, 20)
                dx = np.zeros_like(x)
                counts = np.zeros_like(x)
                all_diffs = [[] for _ in range(len(x))]
                all_weights = [[] for _ in range(len(x))]
                
                for k in range(num_users):
                    user_means = means[k, :, i]
                    user_ci_lower = confidence_intervals[k, :, i, 0]
                    user_ci_upper = confidence_intervals[k, :, i, 1]
                    user_times = timestamps[k, :, i]
                    if np.isnan(user_means).all():
                        continue
                    
                    times = [datetime.fromtimestamp(t) for t in user_times]
                    
                    # Calculate time differences
                    time_diffs = np.diff(user_times)
                    time_diffs_months = time_diffs / (30 * 24 * 60 * 60)
                    
                    # Normalize mean differences by time differences
                    mean_diffs = np.diff(user_means) / time_diffs_months
                    
                    # Plot means with confidence intervals
                    alpha = 0.001 + 0.02 * np.std(user_means)
                    color = 'blue'
                    ax.plot(times, user_means, color=color, alpha=alpha)
                    ax.fill_between(times, user_ci_lower, user_ci_upper, color=color, alpha=alpha/4)

                    # Accumulate data for 1D flow field
                    for l, val in enumerate(x):
                        mask = (user_means[:-1] >= val - 0.1) & (user_means[:-1] < val + 0.1)
                        if np.sum(mask) > 0:
                            dx[l] += np.nansum(mean_diffs[mask])
                            counts[l] += np.sum(mask)
                            all_diffs[l].extend(mean_diffs[mask])
                            ci_widths = user_ci_upper[:-1][mask] - user_ci_lower[:-1][mask]
                            all_weights[l].extend(1 / (ci_widths ** 2))  # Use inverse variance as weight

                # Calculate weighted average and statistical significance for 1D flow field
                mask = counts > 0
                dx[mask] /= counts[mask]
                
                # Perform weighted t-test for each bin
                p_values = np.ones_like(x)
                for l in range(len(x)):
                    if len(all_diffs[l]) > 1:
                        weights = np.array(all_weights[l])
                        weighted_mean = np.average(all_diffs[l], weights=weights)
                        weighted_variance = np.average((np.array(all_diffs[l]) - weighted_mean)**2, weights=weights)
                        weighted_std = np.sqrt(weighted_variance)
                        t_statistic = weighted_mean / (weighted_std / np.sqrt(len(all_diffs[l])))
                        p_values[l] = 2 * (1 - stats.t.cdf(abs(t_statistic), df=len(all_diffs[l])-1))
                
                # Normalize and plot 1D flow field on separate figure
                flow_ax = flow_axes[i]
                max_dx = np.max(np.abs(dx))
                if max_dx > 0:
                    dx /= max_dx
                
                # Create a norm for this specific 1D flow field
                norm = colors.Normalize(vmin=-1, vmax=1)
                
                # Plot arrows with color based on magnitude and direction
                for l in range(len(x)):
                    magnitude = dx[l]
                    color = cmap(norm(magnitude))
                    alpha = 0.7 if p_values[l] < significance_threshold else 0.2
                    flow_ax.quiver(x[l], x[l], magnitude, 0, scale=2, width=0.004, color=color, alpha=alpha)
                
                flow_ax.set_xlim(-1, 1)
                flow_ax.set_ylim(-1, 1)
                flow_ax.set_title(f'1D Flow Field for {opinion_names[i]}')
                flow_ax.set_xlabel('Opinion Value')
                # remove y axis
                # flow_ax.get_yaxis().set_visible(False)
                # remove y axis ticks
                # flow_ax.get_yaxis().set_ticks([])
                
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                flow_figs[i].colorbar(sm, ax=flow_ax, label='Change rate (normalized)')
                
                flow_figs[i].tight_layout()
                if save:
                    flow_figs[i].savefig(os.path.join(save_path, f'flow_field_{i}.png'))

                # Format x-axis for datetime
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                fig.autofmt_xdate()  # Rotate and align the tick labels

            elif i < j:
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                for k in range(num_users):
                    if np.isnan(means[k,:,i]).all() or np.isnan(means[k,:,j]).all():
                        continue
                    
                    # Get timestamps for both opinions
                    times_i = timestamps[k, :, i]
                    times_j = timestamps[k, :, j]
                    
                    # Create a common time range
                    common_times = np.sort(np.unique(np.concatenate([times_i, times_j])))
                    
                    # Interpolate means for both opinions
                    interp_i = interp1d(times_i, means[k,:,i], bounds_error=False, kind='linear', fill_value=(means[k,0,i], means[k,-1,i]))
                    interp_j = interp1d(times_j, means[k,:,j], bounds_error=False, kind='linear', fill_value=(means[k,0,j], means[k,-1,j]))
                    
                    aligned_means_i = interp_i(common_times)
                    aligned_means_j = interp_j(common_times)
                    
                    # Plot aligned means
                    alpha = min(0.001 + 0.01 * np.std(aligned_means_i) + 0.01 * np.std(aligned_means_j), 1.)
                    color = 'blue'
                    ax.plot(aligned_means_i, aligned_means_j, color=color, alpha=alpha)
                    
                    # Add arrows to show direction of time
                    num_arrows = min(5, len(common_times) - 1)
                    arrow_indices = np.linspace(0, len(common_times) - 2, num_arrows, dtype=int)
                    for idx in arrow_indices:
                        ax.annotate('', xy=(aligned_means_i[idx+1], aligned_means_j[idx+1]),
                                    xytext=(aligned_means_i[idx], aligned_means_j[idx]),
                                    arrowprops=dict(arrowstyle='->', color=color, alpha=alpha))

            else:  # i > j
                # Implement vector field analysis
                grid_step = 0.1
                x = np.linspace(-1, 1, 20)
                y = np.linspace(-1, 1, 20)
                X, Y = np.meshgrid(x, y)
                
                dx = np.zeros_like(X)
                dy = np.zeros_like(Y)
                counts = np.zeros_like(X)
                all_x_diffs = [[[] for _ in range(X.shape[1])] for _ in range(X.shape[0])]
                all_y_diffs = [[[] for _ in range(Y.shape[1])] for _ in range(Y.shape[0])]
                all_weights = [[[] for _ in range(X.shape[1])] for _ in range(X.shape[0])]
                
                for k in range(num_users):
                    times_i = timestamps[k, :, i]
                    times_j = timestamps[k, :, j]
                    
                    # Create a common time range
                    common_times = np.sort(np.unique(np.concatenate([times_i, times_j])))
                    
                    # Interpolate means for both opinions
                    interp_i = interp1d(times_i, means[k,:,i], bounds_error=False, kind='linear', fill_value=(means[k,0,i], means[k,-1,i]))
                    interp_j = interp1d(times_j, means[k,:,j], bounds_error=False, kind='linear', fill_value=(means[k,0,j], means[k,-1,j]))
                    
                    aligned_means_i = interp_i(common_times)
                    aligned_means_j = interp_j(common_times)
                    
                    time_diffs_months = np.diff(common_times) / (30 * 24 * 60 * 60)
                    
                    x_data = aligned_means_i[:-1]
                    y_data = aligned_means_j[:-1]
                    x_next = aligned_means_i[1:]
                    y_next = aligned_means_j[1:]
                    
                    x_diffs = (x_next - x_data) / time_diffs_months
                    y_diffs = (y_next - y_data) / time_diffs_months
                    
                    # Interpolate confidence intervals
                    interp_ci_i_lower = interp1d(times_i, confidence_intervals[k, :, i, 0], kind='linear', fill_value='extrapolate')
                    interp_ci_i_upper = interp1d(times_i, confidence_intervals[k, :, i, 1], kind='linear', fill_value='extrapolate')
                    interp_ci_j_lower = interp1d(times_j, confidence_intervals[k, :, j, 0], kind='linear', fill_value='extrapolate')
                    interp_ci_j_upper = interp1d(times_j, confidence_intervals[k, :, j, 1], kind='linear', fill_value='extrapolate')
                    
                    x_ci_lower = interp_ci_i_lower(common_times)[:-1]
                    x_ci_upper = interp_ci_i_upper(common_times)[:-1]
                    y_ci_lower = interp_ci_j_lower(common_times)[:-1]
                    y_ci_upper = interp_ci_j_upper(common_times)[:-1]
                    
                    for m in range(X.shape[0]):
                        for n in range(X.shape[1]):
                            mask = ((x_data >= X[m,n]-grid_step) & (x_data < X[m,n]+grid_step) & 
                                    (y_data >= Y[m,n]-grid_step) & (y_data < Y[m,n]+grid_step) &
                                    ~np.isnan(x_diffs) & ~np.isnan(y_diffs))
                            if np.sum(mask) > 0:
                                dx[m,n] += np.sum(x_diffs[mask])
                                dy[m,n] += np.sum(y_diffs[mask])
                                counts[m,n] += np.sum(mask)
                                all_x_diffs[m][n].extend(x_diffs[mask])
                                all_y_diffs[m][n].extend(y_diffs[mask])
                                x_ci_widths = x_ci_upper[mask] - x_ci_lower[mask]
                                y_ci_widths = y_ci_upper[mask] - y_ci_lower[mask]
                                all_weights[m][n].extend(1 / (x_ci_widths**2 + y_ci_widths**2))  # Use inverse total variance as weight
                
                # Calculate weighted average
                mask = counts > 0
                dx[mask] /= counts[mask]
                dy[mask] /= counts[mask]
                
                # Calculate statistical significance
                p_values = np.ones_like(X)
                for m in range(X.shape[0]):
                    for n in range(X.shape[1]):
                        if len(all_x_diffs[m][n]) > 1 and len(all_y_diffs[m][n]) > 1:
                            weights = np.array(all_weights[m][n])
                            
                            # Weighted t-test for x component
                            x_weighted_mean = np.average(all_x_diffs[m][n], weights=weights)
                            x_weighted_variance = np.average((np.array(all_x_diffs[m][n]) - x_weighted_mean)**2, weights=weights)
                            x_weighted_std = np.sqrt(x_weighted_variance)
                            x_t_statistic = x_weighted_mean / (x_weighted_std / np.sqrt(len(all_x_diffs[m][n])))
                            p_x = 2 * (1 - stats.t.cdf(abs(x_t_statistic), df=len(all_x_diffs[m][n])-1))
                            
                            # Weighted t-test for y component
                            y_weighted_mean = np.average(all_y_diffs[m][n], weights=weights)
                            y_weighted_variance = np.average((np.array(all_y_diffs[m][n]) - y_weighted_mean)**2, weights=weights)
                            y_weighted_std = np.sqrt(y_weighted_variance)
                            y_t_statistic = y_weighted_mean / (y_weighted_std / np.sqrt(len(all_y_diffs[m][n])))
                            p_y = 2 * (1 - stats.t.cdf(abs(y_t_statistic), df=len(all_y_diffs[m][n])-1))
                            
                            p_values[m,n] = min(p_x, p_y)
                
                # Normalize by max vector
                magnitude = np.sqrt(dx**2 + dy**2)
                max_magnitude = np.max(magnitude)
                if max_magnitude > 0:
                    dx /= max_magnitude
                    dy /= max_magnitude

                # Create a norm for this specific 2D flow field
                norm = colors.Normalize(vmin=0, vmax=1)

                # Plot vector field
                significant = p_values < significance_threshold
                ax.quiver(X[significant], Y[significant], dx[significant], dy[significant], 
                          magnitude[significant], scale=25, width=0.004, cmap=cmap, norm=norm)
                ax.quiver(X[~significant], Y[~significant], dx[~significant], dy[~significant], 
                          color='gray', scale=25, width=0.004, alpha=0.2)
                
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax)
                cbar.set_label('Change rate (normalized)')
                
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

            if do_pairplot:
                if j == 0:
                    ax.set_ylabel(opinion_names[i])
                if i == num_opinions - 1:
                    ax.set_xlabel(opinion_names[j])
            else:
                if i == j:
                    ax.set_ylabel('Stance')
                    ax.set_xlabel('Time')
                    ax.set_title(opinion_names[i])
                else:
                    ax.set_ylabel(opinion_names[j])
                    ax.set_xlabel(opinion_names[i])
                fig.tight_layout()
                if save:
                    fig.savefig(os.path.join(save_path, f'flow_field_{i}_{j}.png'))

    if do_pairplot:
        figs.tight_layout()
        if save:
            figs.savefig(os.path.join(save_path, 'flow_pairplot.png'))

    return figs, axes, flow_figs, flow_axes