import numpy as np
import pylab as pl

pl.rcParams['font.family'] = 'serif'
pl.rcParams['font.sans-serif'] = 'CMU Serif Roman'
pl.rcParams.update({'font.size': 18})

a = np.array([[0, 1]])
pl.figure(figsize=(9, 2))
img = pl.imshow(a, cmap="Greens")
pl.gca().set_visible(False)
cax = pl.axes([0.1, 0.2, 0.8, 0.6])
cbar = pl.colorbar(orientation="horizontal", cax=cax)
pl.tight_layout()

cbar_labels = [item.get_text() for item in cbar.ax.get_xticklabels()]
cbar_labels = [fr'$t_\mathsf{{{"final"}}}$',
               fr'$t_\mathsf{{{"final"}}} + S_\mathsf{{{"t"}}}$',
               fr'$t_\mathsf{{{"final"}}} + 2 S_\mathsf{{{"t"}}}$',
               fr'$t_\mathsf{{{"final"}}} + 3 S_\mathsf{{{"t"}}}$',
               fr'$t_\mathsf{{{"final"}}} + 4 S_\mathsf{{{"t"}}}$',
               fr'$t_\mathsf{{{"final"}}} + 5 S_\mathsf{{{"t"}}}$']
# cbar_labels[-1] = f'{final_state_tick}'  # horizontal colorbar
# cbar_labels[0] = fr'$t_\mathsf{{{"final"}}}$'
# cbar_labels[1] = fr'$S_\mathsf{{{"period"}}}$'
# cbar_labels[2] = fr'$2S_\mathsf{{{"period"}}}$'
# cbar_labels[3] = fr'$3S_\mathsf{{{"period"}}}$'
# cbar_labels[4] = fr'$4S_\mathsf{{{"period"}}}$'
# cbar_labels[5] = fr'$5S_\mathsf{{{"period"}}}$'
cbar.ax.set_xticklabels(cbar_labels)
cbar.ax.set_xlabel(fr"Ticks $t$")

pl.savefig("colorbar.png", dpi=300)

