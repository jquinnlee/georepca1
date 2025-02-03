from utils import *

def plot_maps(maps, animal, path, SFPs_traced=False, example_cells_idx=False, map_type=False, unsmoothed=False, norm_days=True, make_dir=True,
              cmap='viridis', cpf = 10, fig_height=8, fig_width=10):
    '''
    Plot aligned rate maps for registered cells across days
    :arg path: defines where plots of aligned rate maps will be saved
    :arg unsmoothed: whether to plot smoothed or unsmoothed ratemaps
    :arg make_dir: indicate whether new folder needs to be created for saving images
    '''
    os.chdir(path)
    if make_dir:
        ppath = os.path.join(path, 'rate_maps')
        if not glob(ppath):
            os.mkdir(ppath)
        os.chdir(ppath)
        if map_type:
            spath = os.path.join(ppath, map_type)
            if not glob(spath):
                os.mkdir(spath)
            os.chdir(spath)
    else:
        os.chdir(os.path.join(path, 'rate_maps'))

    if unsmoothed:
        maps = maps['unsmoothed']
    else:
        maps = maps['smoothed']

    n_cells = maps.shape[2]
    n_days = maps.shape[3]
    # iterate over cells
    # for cell in np.arange(n_cells):
    for cell in np.arange(n_cells):
        # generate new figure if cpf is satisfied or cell idx is 0
        if cell % cpf == 0:
            plt.figure(figsize=(fig_width, fig_height))
            count = 1
        # first plot the traced SFP of the target cell
        if np.any(SFPs_traced):
            if (count - 1) % (n_days + 1) == 0:
                ax = plt.subplot(cpf, n_days + 1, count)
                traced_projection = np.nansum(SFPs_traced[:, :, cell, :], axis=2).astype(float)
                traced_projection /= np.nanmax(traced_projection)
                ax.imshow(traced_projection[5:-5, 5:-5], cmap='binary', vmax=.33)
                ax.set_xticks([])
                ax.set_yticks([])
                if cell == 0:
                    ax.set_title('SFP', weight='bold', size=16, pad=10)
                if np.any(example_cells_idx):
                    ax.set_ylabel(f'C{example_cells_idx[cell] + 1}', fontsize=12, weight='bold')
                else:
                    ax.set_ylabel(f'C{cell + 1}', fontsize=12, weight='bold')
                count += 1
        # grab maximum event rate of cell across days
        cell_maxs = np.zeros(n_days)
        for day in np.arange(n_days):
            cell_maxs[day] = np.nanmax(maps[:, :, cell, day].T)
        cell_max = np.nanmax(cell_maxs)
        for day in np.arange(n_days):
            # if target cell is registered on session plot rate map
            if ~np.all(np.isnan(maps[:, :, cell, day])):
                if np.any(SFPs_traced):
                    ax = plt.subplot(cpf, n_days + 1, count)
                else:
                    ax = plt.subplot(cpf, n_days, count)
                # plot rate map. Note the '-1' since alignMap is 1 indexed, but rate_maps are 0 indexed
                if norm_days:
                    ax.imshow(maps[:, :, cell, day].T, cmap=cmap, vmax=cell_max, vmin=0)
                else:
                    ax.imshow(maps[:, :, cell, day].T, cmap=cmap, vmin=0)
                ax.set_xticks([])
                ax.set_yticks([])
            # if not registered on session plot nans size of rate map
            else:
                if np.any(SFPs_traced):
                    ax = plt.subplot(cpf, n_days + 1, count)
                else:
                    ax = plt.subplot(cpf, n_days, count)
                # plot nans size of rate map. Note the '-1' since alignMap is 1 indexed, but rate_maps are 0 indexed
                ax.imshow(np.zeros_like(maps[:, :, cell, day].T) * np.nan, cmap='jet', vmax=cell_max, vmin=0)
                ax.set_xticks([])
                ax.set_yticks([])
            plt.setp(ax.spines.values(), color='k', linewidth=1.5)
            count += 1
        if cell % cpf == (cpf - 1) or (day == n_days - 1 and cell == n_cells - 1):
            # plt.suptitle(f'{animal}')
            if map_type:
                plt.savefig(f'{animal}_{map_type}_c{cell + 2 - cpf}-{cell + 1}.svg', format='svg')
            else:
                plt.savefig(f'{animal}_c{cell + 2 - cpf}-{cell + 1}.svg', format='svg')
            plt.show()