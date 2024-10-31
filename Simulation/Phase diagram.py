import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as cl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage as ndimage
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = 'CMU Serif Roman'
plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.figsize"] = [10, 10]

file_name = "final_data/42_0.05.npz"
sigma = file_name[-8:-4]

# class IndexTracker:
#     def __init__(self, ax, X, slice_titles, slice_norms, slice_cmaps, slice_patches, vmax_slices):
#         self.ax = ax
#
#         self.X = X
#         rows, cols, self.slices = X.shape
#         self.ind = 0
#
#         self.divider = make_axes_locatable(self.ax)
#         self.cax = self.divider.append_axes("right", size="5%", pad=0.05)
#         self.slice_norms = slice_norms
#         self.slice_cmaps = slice_cmaps
#         self.slice_titles = slice_titles
#         self.im = ax.imshow(self.X[:, :, self.ind], cmap=self.slice_cmaps[self.ind], origin="lower", norm=self.slice_norms[self.ind], vmin=0)
#
#         self.slice_patches = slice_patches
#
#         self.update()
#
#     def on_scroll(self, event):
#         print("%s %s" % (event.button, event.step))
#         increment = 1 if event.button == 'up' else -1
#         max_index = self.X.shape[-1] - 1
#         self.ind = np.clip(self.ind + increment, 0, max_index)
#         self.update()
#
#     def update(self):
#         self.im.set_data(self.X[:, :, self.ind])
#         cmap = self.slice_cmaps[self.ind]
#         cmap.set_bad(color='w')
#         # cmap.set_under(color='w')
#         self.im = ax.imshow(self.X[:, :, self.ind], cmap=cmap, origin="lower", norm=self.slice_norms[self.ind], vmin=0.0, vmax=vmax_slices[self.ind])  # norm=cl.TwoSlopeNorm(np.median(mean_variance[:, :, 0])))
#         self.ax.set_xlabel(fr"$\Delta v_\mathsf{{{'dec'}}}$ / $\Delta v_\mathsf{{{'acc'}}}$")
#         self.ax.set_ylabel(fr"$L$ / $(N \cdot \Delta x_\mathsf{{{'min'}}})$")
#         fig.suptitle(self.slice_titles[self.ind])
#         plt.xticks(ax.get_xticks(), np.round_(((ax.get_xticks() * x_axis_scale) + x_axis_trans), decimals=3))
#         plt.yticks(ax.get_yticks(), np.round_((ax.get_yticks() * y_axis_scale) + y_axis_trans, decimals=3))
#         ax.set_xlim(left=0, right=X_pixels)
#         ax.set_ylim(top=Y_pixels, bottom=0)
#
#         if self.slice_patches[self.ind] != []:
#             self.ax.legend(handles=self.slice_patches[self.ind],  loc='upper left')
#
#         fig.colorbar(self.im, cax=self.cax)
#         self.im.axes.figure.canvas.draw()
#
#         # plt.savefig(f"phase_diagrams/{file_name[:-4]}/{int(self.ind)}.png", dpi=300)
#
#         if self.slice_patches[self.ind] != []:
#             self.ax.get_legend().remove()

def fmt(x):
    s = f"{x:.2f}"
    if s.endswith("0"):
        s = f"{x:.1f}"
    return f"{s}"

file = np.load(file_name)
initial_parameters = file['init_car_params'].copy()
final_state_properties = file['final_param'].copy()
system_parameters = file['sys_param'].copy()
final_state_information = file['final_info'].copy()
active_interactions_per_tick = file['interactions_count'].copy()
mean_interaction_length = file['interaction_length'].copy()

# system_parameters[i, j] = [N, width, acc, dec, min_delta]
# print(np.shape(system_parameters))

number_of_cars = int(system_parameters[0, 0, 0])
num_ticks = system_parameters
min_delta = system_parameters[0, 0, 4]

num_I, num_J, num_sys_params = np.shape(system_parameters)
num_runs = np.shape(final_state_properties)[2]

width_N_vals = np.empty(num_I)
dec_acc_vals = np.empty(num_J)

for i, fixed_width in enumerate(system_parameters):
    for j, system_par in enumerate(fixed_width):
        # print(system_par)
        width_N_vals[i] = system_par[1] / system_par[0]
        dec_acc_vals[j] = system_par[3] / system_par[2]

X_param_range = dec_acc_vals[-1] - dec_acc_vals[0]
X_pixels = len(dec_acc_vals)
Y_param_range = width_N_vals[-1] - width_N_vals[0]
Y_pixels = len(width_N_vals)

print(X_param_range)
print(X_pixels)
print(Y_param_range)
print(Y_pixels)

mean_variance = np.empty((num_I, num_J, num_runs))
mean_speed = np.empty((num_I, num_J, num_runs))
number_of_solitons = np.empty((num_I, num_J, num_runs))
final_state = np.empty((num_I, num_J, num_runs))
final_state_time = np.empty((num_I, num_J, num_runs))
final_soliton_period = np.empty((num_I, num_J, num_runs))
total_ticks_ran = np.empty((num_I, num_J, num_runs))

average_interactions_per_tick = np.empty((num_I, num_J, num_runs))

for i, fixed_width in enumerate(final_state_properties):
    for j, final_props in enumerate(fixed_width):
        for run in range(num_runs):
            mean_speed[i, j, run] = final_props[run, 0]
            mean_variance[i, j, run] = final_props[run, 1]
            total_ticks_ran[i, j, run] = final_props[run, 10]


for i, fixed_width in enumerate(final_state_information):
    for j, final_info in enumerate(fixed_width):
        for run in range(num_runs):

            # if final_state_time[i, j, run] == 0.0:  # Whoops! Incorrectly saved time to reached final state as zero
            #     final_state_time[i, j, run] = total_ticks_ran[i, j, run]

            if final_info[run, 2] == 2.0:  # If unknown state, set values negative, so they will be blacked out
                final_state_time[i, j, run] = np.nan
                number_of_solitons[i, j, run] = np.nan
                final_soliton_period[i, j, run] = np.nan
                final_state[i, j, run] = np.nan
            else:
                final_state_time[i, j, run] = final_info[run, 1]
                number_of_solitons[i, j, run] = final_info[run, 3]
                final_soliton_period[i, j, run] = final_info[run, 5]
                final_state[i, j, run] = final_info[run, 2] * 0.5
            if final_info[run, 2] == 0.0:
                final_soliton_period[i, j, run] = np.nan
                number_of_solitons[i, j, run] = np.nan

            if final_info[run, 2] == 0.0:
                average_interactions_per_tick[i, j, run] = np.nan
                mean_interaction_length[i,  j, run] = np.nan
            else:
                average_interactions_per_tick[i, j, run] = np.dot(active_interactions_per_tick[i, j], np.arange(0, 100)) / total_ticks_ran[i, j, run]

# mean_speed[:, :, 0] = np.nan_to_num(mean_speed[:, :, 0], nan=0.0)  # Now fixed and not needed (avg_v was undefined for some states)
# mean_variance[:, :, 0] = np.nan_to_num(mean_variance[:, :, 0], nan=1000)  # Now fixed and not needed (var_v was undefined for some states)

mean_speed_smooth = ndimage.gaussian_filter(mean_speed[:, :, 0], sigma=1, order=0, mode='reflect')

# plots = np.dstack((mean_speed[:, :, 0], mean_variance[:, :, 0], mean_speed_smooth, number_of_solitons[:, :, 0],
#                    final_state[:, :, 0], final_state_time[:, :, 0], final_soliton_period[:, :, 0], total_ticks_ran[:, :, 0]))


cmaps = [plt.get_cmap('viridis').copy(),
         plt.get_cmap('viridis').copy(),
         plt.get_cmap('viridis').copy(),
         plt.get_cmap('viridis').copy(),
         plt.get_cmap('viridis_r').copy(),
         plt.get_cmap('viridis_r').copy(),
         plt.get_cmap('viridis').copy(),
         plt.get_cmap('viridis').copy(),
         plt.get_cmap('viridis').copy(),
         plt.get_cmap('viridis').copy()]

norms = [None, None, None, None, cl.Normalize(vmin=0, vmax=1), None, None, None,
             cl.LogNorm(vmin=0.001, vmax=50), cl.Normalize(vmin=1, vmax=10)]  # cl.Normalize(vmax=20., vmin=0.1, clip=True)

colors = [[get_cmap('Greys')(0)],
          [get_cmap('Greys')(0)],
          [get_cmap('Greys')(0)],
          [get_cmap('Greys')(0)],
          [get_cmap('viridis_r')(0), get_cmap('viridis_r')(0.5)],  #, get_cmap('Greys')(0)],
          [get_cmap('Greys')(0)],
          [get_cmap('Greys')(0)],
          [get_cmap('Greys')(0)],
          [get_cmap('Greys')(0)],
          [get_cmap('Greys')(0)]]

legend_labels = [["Unknown"],
                 ["Unknown"],
                 ["Unknown"],
                 ["None (free state)"],
                 ["Free", "Soliton"],  # "Unknown"],
                 ["Unknown"],
                 ["Not applicable"],
                 ["Unknown"],
                 ["None"],
                 ["None"]]

patches = [[],  # [mpatches.Patch(color=colors[0][i], label=legend_labels[0][i], edgecolor='black') for i in range(1)],
           [],  # [mpatches.Patch(color=colors[1][i], label=legend_labels[1][i], edgecolor='black') for i in range(1)],
           [],  # [mpatches.Patch(color=colors[2][i], label=legend_labels[2][i], edgecolor='black') for i in range(1)],
           [mpatches.Patch(color=colors[3][i], label=legend_labels[3][i], edgecolor='black') for i in range(1)],
           [mpatches.Patch(color=colors[4][i], label=legend_labels[4][i], edgecolor='black') for i in range(2)],  # range(3)],
           [],  # [mpatches.Patch(color=colors[5][i], label=legend_labels[5][i], edgecolor='black') for i in range(1)],
           [mpatches.Patch(color=colors[6][i], label=legend_labels[6][i], edgecolor='black') for i in range(1)],
           [],  # [mpatches.Patch(color=colors[7][i], label=legend_labels[7][i], edgecolor='black') for i in range(1)],
           [mpatches.Patch(color=colors[8][i], label=legend_labels[8][i], edgecolor='black') for i in range(1)],
           [mpatches.Patch(color=colors[8][i], label=legend_labels[8][i], edgecolor='black') for i in range(1)]]

vmax_slices = [None, None, None, None, 1, None, None, None, None, None]

formats = [None, None, None, None, None, '%.1e', '%.1e', '%.1e', '%.1e', None]

titles = [
        f'Final mean speed \n'
        #fr'fixed initial $v_0$, equally-spaced $x_0$' f'\n'
        fr'N = {number_of_cars}, $\sigma_\mathsf{{{"v"}}} = {sigma}$, $T_\mathsf{{{"max"}}}$ = 250,000',

        f'Final speed variance \n'
        #fr'fixed initial $v_0$, equally-spaced $x_0$' f'\n'
        fr'N = {number_of_cars}, $\sigma_\mathsf{{{"v"}}} = {sigma}$, $T_\mathsf{{{"max"}}}$ = 250,000',

        f'Final mean speed (smoothed) \n'
        #fr'fixed initial $v_0$, equally-spaced $x_0$' f'\n'
        fr'N = {number_of_cars}, $\sigma_\mathsf{{{"v"}}} = {sigma}$, $T_\mathsf{{{"max"}}}$ = 250,000',

        f'Final number of solitons \n'
        #fr'fixed initial $v_0$, equally-spaced $x_0$' f'\n'
        fr'N = {number_of_cars}, $\sigma_\mathsf{{{"v"}}} = {sigma}$, $T_\mathsf{{{"max"}}}$ = 250,000',

        f'Final state type \n'
        #fr'fixed initial $v_0$, equally-spaced $x_0$' f'\n'
        fr'N = {number_of_cars}, $\sigma_\mathsf{{{"v"}}} = {sigma}$, $T_\mathsf{{{"max"}}}$ = 250,000',

        f'Final state \n'
        #fr'fixed initial $v_0$, equally-spaced $x_0$' f'\n'
        fr'N = {number_of_cars}, $\sigma_\mathsf{{{"v"}}} = {sigma}$, $T_\mathsf{{{"max"}}}$ = 250,000',

        f'Soliton period \n'
        #fr'fixed initial $v_0$, equally-spaced $x_0$' f'\n'
        fr'N = {number_of_cars}, $\sigma_\mathsf{{{"v"}}} = {sigma}$, $T_\mathsf{{{"max"}}}$ = 250,000',

        f'Ticks ran / saved \n '
        #fr'fixed initial $v_0$, equally-spaced $x_0$' f'\n'
        fr'N = {number_of_cars}, $\sigma_\mathsf{{{"v"}}} = {sigma}$, $T_\mathsf{{{"max"}}}$ = 250,000',

        f'Average concurrent interactions \n'
        #fr'fixed initial $v_0$, equally-spaced $x_0$' f'\n'
        fr'N = {number_of_cars}, $\sigma_\mathsf{{{"v"}}} = {sigma}$, $T_\mathsf{{{"max"}}}$ = 250,000',

        f'Average interaction length \n'
        #fr'fixed initial $v_0$, equally-spaced $x_0$' f'\n'
        fr'N = {number_of_cars}, $\sigma_\mathsf{{{"v"}}} = {sigma}$, $T_\mathsf{{{"max"}}}$ = 250,000'
        ]


print(fr"$\Delta v_\mathsf{{{'dec'}}} / \Delta v_\mathsf{{{'acc'}}}$")

x_axis_scale = dec_acc_vals[1] - dec_acc_vals[0]
y_axis_scale = (width_N_vals[1] - width_N_vals[0]) / min_delta
x_axis_trans = dec_acc_vals[0]
y_axis_trans = (width_N_vals[0]) / min_delta

# fig, ax = plt.subplots(1, 1)
# tracker = IndexTracker(ax, plots, titles, norms, cmaps, patches, vmax_slices)
# fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
# plt.sca(ax)

colors = ["k", "k", "w", "w", "w", "w", "w", "w", "w", "w"]
# CS = ax.contour(mean_speed_smooth, levels=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .95], linewidths=1., colors="k")
# colors = ["w", "w", "k", "k", "k", "k", "k", "k", "k", "k"]
# ax.scatter(*zip(*manual_locations))


# ax.plot([10, 160], [20, 20], '-o', color='red', linestyle='dashed', alpha=0.6)
# manual_locations = [(100, 10), (120, 20), (140, 45), (160, 60), (170, 90), (180, 125), (190, 150),
#                     (175, 175), (170, 185), (150, 185)]
# ax.clabel(CS, levels=CS.levels, fontsize=13, inline=False, inline_spacing=0., fmt=fmt, manual=manual_locations,
#           colors="k")


#plt.yticks(ax.get_yticks(), np.linspace(width_N_vals[0], width_N_vals[-1], num=len(ax.get_yticks())))

# ax.set_xlim(left=0, right=X_pixels)
# ax.set_ylim(top=Y_pixels, bottom=0)

plt.ion()
plt.tight_layout()
plt.show()

print(width_N_vals[0])

for index, pd in enumerate([mean_speed[:, :, 0], mean_variance[:, :, 0], mean_speed_smooth, number_of_solitons[:, :, 0],
                            final_state[:, :, 0], final_state_time[:, :, 0], final_soliton_period[:, :, 0],
                            total_ticks_ran[:, :, 0], average_interactions_per_tick[:, :, 0], mean_interaction_length[:, :, 0]]):

    # cbar_labels = [fr"Final $\overline v$", r"Final ${\sigma_v}^2$", r"Final $\overline v$", r"Final $S_n$",
    #                'State type', fr'$t_\mathsf{{{"final"}}}$', r'$S_T$', 'Ticks saved', 'Avg. conc. \n interactions']

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot()
    plt.tight_layout(pad=4)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cmap = cmaps[index]
    cmap.set_bad(color='w')
    if index == 9:
        cmap.set_over(color=get_cmap('viridis_r')(0))
    # cmap.set_under(color='w')
    im = ax.imshow(pd, cmap=cmap, origin="lower", norm=norms[index], extent=[0, 400, 1, 21], aspect='auto')  # norm=cl.TwoSlopeNorm(np.median(mean_variance[:, :, 0])))
    ax.set_xlabel(fr"$\Delta v_\mathsf{{{'dec'}}}$ / $\Delta v_\mathsf{{{'acc'}}}$")
    ax.set_ylabel(fr"$L$ / $(N \cdot \Delta x_\mathsf{{{'min'}}})$")
    fig.suptitle(titles[index])
    # plt.xticks(ax.get_xticks(), np.round_(((ax.get_xticks() * x_axis_scale) + x_axis_trans), decimals=3))
    # plt.yticks(ax.get_yticks(), np.round_((ax.get_yticks() * y_axis_scale) + y_axis_trans, decimals=3))
    # ax.set_xlim(left=0, right=X_pixels)
    # ax.set_ylim(top=Y_pixels, bottom=0)

    if patches[index] != []:
        legend = ax.legend(handles=patches[index],  loc='upper left', facecolor='gray')
        frame = legend.get_frame()
        frame.set_facecolor('0.90')

    if formats[index] != None:
        cb = fig.colorbar(im, cax=cax, format=formats[index])
        # cb.ax.get_yaxis().labelpad = 18
        # cb.ax.set_ylabel(cbar_labels[index], rotation=270)
    else:
        cb = fig.colorbar(im, cax=cax)
    # im.axes.figure.canvas.draw()

    plt.savefig(f"phase_diagrams/{file_name[:-4]}/{int(index)}.png", dpi=300)

    if patches[index] != []:
        ax.get_legend().remove()

# plt.close('all')
print(f"All phase diagrams saved in phase_diagrams/{file_name[:-4]}")