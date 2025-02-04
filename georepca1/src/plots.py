from utils import *


def plot_maps(maps, animal, path, SFPs_traced=False, example_cells_idx=False, map_type=False, unsmoothed=False,
              norm_days=True, make_dir=True, cmap='viridis', cpf = 10, fig_height=8, fig_width=10):
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


def plot_shr_pvals(df_pvals):
    sns.set(style='dark', font_scale=1.75)
    alpha = [0.05, 0.01, 0.001]
    plt.figure(figsize=(4, 5))
    df_pvals_a, df_pvals_b, df_pvals_c = deepcopy(df_pvals), deepcopy(df_pvals), deepcopy(df_pvals)
    ax = plt.subplot()
    df_pvals_a['SHR'] = df_pvals_a['SHR'] < alpha[0]
    sns.lineplot(data=df_pvals_a.groupby(by=['Animal', 'Day']).mean(), x='Day', y='SHR', units='Animal', linewidth=1,
                 estimator=None, alpha=.25, c='b')
    sns.lineplot(data=df_pvals_a.groupby(by=['Animal', 'Day']).mean(), x='Day', y='SHR', linewidth=4,
                 estimator='mean',
                 errorbar='se', c='b')

    df_pvals_b['SHR'] = df_pvals_b['SHR'] < alpha[1]
    sns.lineplot(data=df_pvals_b.groupby(by=['Animal', 'Day']).mean(), x='Day', y='SHR', units='Animal',
                 linewidth=1,
                 estimator=None, alpha=.25, c='orchid')
    sns.lineplot(data=df_pvals_b.groupby(by=['Animal', 'Day']).mean(), x='Day', y='SHR', linewidth=4,
                 estimator='mean',
                 errorbar='se', c='orchid')

    df_pvals_c['SHR'] = df_pvals_c['SHR'] < alpha[2]
    sns.lineplot(data=df_pvals_c.groupby(by=['Animal', 'Day']).mean(), x='Day', y='SHR', units='Animal',
                 linewidth=1,
                 estimator=None, alpha=.25, c="coral")
    sns.lineplot(data=df_pvals_c.groupby(by=['Animal', 'Day']).mean(), x='Day', y='SHR', linewidth=4,
                 estimator='mean',
                 errorbar='se', c="coral")

    ax.set_xticks(np.arange(0, df_pvals['Day'].max() + 1, 10).astype(int))
    ax.set_xticklabels(np.arange(1, df_pvals['Day'].max() + 2, 10).astype(int))
    ax.set_ylim([0, 1.])
    ax.set_yticks(np.linspace(0, 1., 6))
    ax.set_ylabel(f'Proportion place cells', weight='bold')
    ax.text(x=0, y=1.1, s="p < 0.05", c="b", size=18, weight="bold")
    ax.text(x=12, y=1.1, s="< 0.01", c="orchid", size=18, weight="bold")
    ax.text(x=21, y=1.1, s="< 0.001", c="coral", size=18, weight="bold")
    ax.set_xlabel('Day', weight='bold')
    plt.show()


def plot_decoding_within_days(df_decoding, residual=False, vmax=40, plot_min_line=True):
    sns.set(style='dark', font_scale=1.75)
    plt.figure(figsize=(4, 5))
    ax = plt.subplot()
    if plot_min_line:
        ax.axhline(10, linewidth=3, linestyle="--", c="gray")  # set line for theoretical max
    sns.lineplot(data=df_decoding, x='Day', y='Error', units='Animal', linewidth=1, estimator=None, alpha=.25)
    sns.lineplot(data=df_decoding, x='Day', y='Error', linewidth=4, estimator='mean', color='b')
    ax.set_xticks(np.arange(0, df_decoding['Day'].max() + 1, 10).astype(int))
    ax.set_xticklabels(np.arange(1, df_decoding['Day'].max() + 2, 10).astype(int))
    if residual:
        ax.set_ylim([-10, vmax - 10])
        ax.set_yticks(np.linspace(-10, vmax, 6))
        ax.axhline(0, linewidth=3, linestyle="--", c="gray")
    else:
        ax.set_ylim([0, vmax])
        ax.set_yticks(np.linspace(0, vmax, 6))
    ax.set_ylabel('Mean Decoding Error (cm)', weight='bold')
    ax.set_xlabel('Day', weight='bold')
    plt.show()


def plot_map_corr_mds(mean_map_corr, labels, vmax=.5, cbar_title="Ratemap correlation"):

    mds_ = MDS(n_components=2, metric=False, n_jobs=9, n_init=1, max_iter=int(1e12), eps=1e-12, random_state=9003,
               dissimilarity='precomputed')
    mds_.fit(1-mean_map_corr)
    embeddings = deepcopy(mds_.embedding_)
    # for square shape take the average of the two at ends of sequence to account for drift
    embeddings[0] = np.array([embeddings[0], embeddings[-1]]).mean(0)
    embeddings -= np.mean(embeddings, axis=0)
    embeddings += np.array([.15, -.15])
    e_polys = [get_environment_label(env)[1] for env in labels[:-1]]
    sns.set(style="dark", font_scale=2.)
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    # add custom, colored patches of shapes within plot for legend
    label_scale = np.array([0.005, 0.005])
    cmap = matplotlib.cm.get_cmap('cool')
    for i, poly in enumerate(e_polys):
        ax.add_patch(patches.PathPatch(mpath.Path(poly * label_scale + np.array([embeddings[i, 0],
                                                                                 embeddings[i, 1]])),
                                       clip_on=False, facecolor=cmap(i/(len(labels)-1)), edgecolor=cmap(i/(len(labels)-1))))
    plt.legend([], frameon=False)
    ax.set_ylim([-.75, .75])
    ax.set_xlim([-.75, .75])
    ax.set_xticks(np.linspace(-.75, .75, 2))
    ax.set_xlabel('nMDS Dim1 ($a.u.$)', weight="bold", labelpad=-15)
    ax.set_yticks(np.linspace(-.75, .75, 2))
    ax.set_ylabel('nMDS Dim2 ($a.u.$)', weight="bold", labelpad=-45)
    plt.text(-.65, -.65, f"$stress1$={mds_.stress_:.3f}", weight="bold", fontsize=18)
    plt.tight_layout()
    plt.setp(ax.spines.values(), linewidth=4., color="k")
    plt.show()

    return fig


def plot_map_corr_rsm_ordered(mean_map_corr, labels, vmax=.5, vmin=.0, cbar_title="Ratemap correlation"):
    # plot map correlation rsm defined by hard-coded order (clustered and determined visually from dendrogram)
    temp_corr = deepcopy(mean_map_corr)
    reorder_list = np.array(["+", "rectangle", "glenn", "square", "l", "bit donut", "o", "u", "i", "t"])
    reorder_idx = np.array([np.where(labels[:-1] == label)[0] for label in reorder_list]).ravel()
    e_polys = [get_environment_label(env, flipud=True)[1] for env in labels[:-1][reorder_idx]]
    sns.set(style="dark", font_scale=2.)
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    temp_corr[np.eye(temp_corr.shape[0]).astype(bool)] = np.nan
    mappable = ax.imshow(temp_corr[:-1, :-1][reorder_idx, :][:, reorder_idx],
                         vmin=vmin, vmax=vmax, cmap="inferno", origin="upper")
    cbar = plt.colorbar(mappable, shrink=.7)
    cbar.ax.set_ylabel(cbar_title, weight="bold", labelpad=15, rotation=270)
    # cbar.ax.set_title("$R$", weight="bold")
    cbar.ax.set_yticks(np.linspace(vmin, vmax, 2))
    label_scale = np.array([0.02, 0.02])
    tickypos = -1.5
    tickxpos = 0
    cmap = matplotlib.cm.get_cmap('cool')
    for j, poly in enumerate(e_polys):
        ax.add_patch(patches.PathPatch(mpath.Path(poly * label_scale + np.array([tickxpos, tickypos])),
                                                   clip_on=False, facecolor="b",
                                                   edgecolor="b"))
        tickxpos += 1.
    tickypos = 0
    tickxpos = -1.5
    cmap = matplotlib.cm.get_cmap('cool')
    for j, poly in enumerate(e_polys):
        ax.add_patch(patches.PathPatch(mpath.Path(poly * label_scale + np.array([tickxpos, tickypos])),
                                                   clip_on=False, facecolor="b",
                                                   edgecolor="b"))
        tickypos += 1.
    ax.set_xticks([])
    ax.set_yticks([])
    plt.setp(cbar.ax.spines.values(), color="k", linewidth=4.)
    plt.setp(ax.spines.values(), color="k", linewidth=4.)
    plt.tight_layout()
    plt.show()
    return fig


def plot_map_corr_dendrogram(mean_map_corr, labels):
    # make a temporary copy of the mean map correlation
    temp_corr = deepcopy(mean_map_corr)
    temp_corr[:, 0] = np.vstack((temp_corr[:, 0], np.flip(temp_corr[:, -1]))).mean(0)
    temp_corr[0, :] = temp_corr[:, 0]
    # Calculate distance matrix
    distance_matrix = 1-temp_corr[:-1, :-1]
    Z = linkage(distance_matrix, method='single')
    # plot the dedrogram of similarities
    sns.set(style="dark", font_scale=2.)
    fig = plt.figure(figsize=(7, 6))
    ax = plt.subplot()
    D = dendrogram(Z, labels=labels[:-1], ax=ax)
    e_polys = [get_environment_label(env)[1] for env in D["ivl"]]
    ax.set_ylim([0.7, 1.05])
    tickxpos = 5.
    tickypos = .68
    label_scale = np.array([0.15, 0.00075])
    cmap = matplotlib.cm.get_cmap('cool')
    for j, poly in enumerate(e_polys):
        ax.add_patch(patches.PathPatch(mpath.Path(poly * label_scale + np.array([tickxpos, tickypos])),
                                                   clip_on=False, facecolor="b",
                                                   edgecolor="b"))
        tickxpos += 10
    plt.xticks([])
    plt.ylabel('Distance', weight="bold")
    plt.tight_layout()
    plt.setp(ax.spines.values(), linewidth=4., color="k")
    plt.show()
    return fig


def plot_across_animal_similarity(df_map_corr_animals_sequences):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    sns.barplot(data=df_map_corr_animals_sequences, x="Sequence", y="Fit", hue="Shuffle", legend=False,
                width=.5, edgecolor="k", linewidth=4., errcolor="k", errwidth=4., palette="muted", saturation=1.)
    ax.set_ylim([0.0, .8])
    ax.set_xlabel("Sequence")
    ax.set_xticklabels(["1", "2", "3"])
    ax.set_xlabel("Sequence", weight="bold")
    ax.set_ylabel(f"Across-animal Similarity ($Tau$)", weight="bold")
    plt.setp(ax.spines.values(), color="k", linewidth=4.)
    plt.tight_layout()
    plt.show()
    return fig


def plot_rsm_parts_averaged(rsm_parts_averaged, vmax=1., vmin=-.1, cmap='inferno'):
    sns.set(style='dark', font_scale=2)
    rsm_copy = deepcopy(rsm_parts_averaged)
    nan_mask = ~np.isnan(rsm_copy[np.eye(rsm_copy.shape[0]).astype(bool)])
    rsm_copy = rsm_copy[nan_mask, :][:, nan_mask]
    np.fill_diagonal(rsm_copy, np.nan)
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    mappable = ax.imshow(rsm_copy, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(mappable, shrink=.8)
    cbar.ax.set_title('R', weight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.setp(ax.spines.values(), color='k', linewidth=4)
    plt.tight_layout()
    plt.show()
    return fig


def plot_rsm_parts_mds(rsm_parts_averaged, precomputed=False):
    rsm = deepcopy(rsm_parts_averaged)
    n_parts = 9
    n_envs = rsm.shape[0] // n_parts
    c_labels = np.tile(np.array([0, 3, 6, 1, 4, 7, 2, 5, 8]), n_envs)
    p_labels = np.tile(np.arange(n_parts), n_envs)
    env_labels = np.tile(np.arange(n_envs)[np.newaxis].T, n_parts).ravel()
    nan_mask = ~np.isnan(rsm[np.eye(rsm.shape[0]).astype(bool)])
    # np.random.seed(1984)
    if precomputed:
        mds_ = MDS(n_components=2, metric=False, n_jobs=9, n_init=1, max_iter=int(1e12), eps=1e-12, random_state=9003,
                   dissimilarity='precomputed')
    else:
        mds_ = MDS(n_components=2, metric=False, n_jobs=9, n_init=1, max_iter=int(1e12), eps=1e-12, random_state=9003)
    mds_.fit(1 - (rsm[nan_mask, :][:, nan_mask]))
    square_average = np.vstack((np.vstack((mds_.embedding_[:n_parts, 0], mds_.embedding_[-n_parts:, 0])).mean(0),
                                np.vstack((mds_.embedding_[:n_parts, 1], mds_.embedding_[-n_parts:, 1])).mean(0))).T
    # measure angle of MDS embedding offset, assuming the lower-left and top-right corners of square should be 45 deg
    theta = .25 * np.pi - np.arctan2(square_average[2, 1] - square_average[6, 1],
                                     square_average[2, 0] - square_average[6, 0]) + np.pi
    rotation_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    mds_.embedding_ = (rotation_mat@mds_.embedding_.T).T

    sns.set(style='dark', font_scale=2.1)

    # plot individual embeddings in a single figure panel
    fig = plt.figure(figsize=(24, 10))
    c = 1
    for i in range(1, n_envs):
        x1, y1 = np.vstack((mds_.embedding_[:n_parts, :][p_labels[nan_mask][env_labels[nan_mask] == i]][:, 0],
                            mds_.embedding_[-n_parts:, :][p_labels[nan_mask][env_labels[nan_mask] == i]][:, 0])).mean(0), \
                 np.vstack((mds_.embedding_[:n_parts, :][p_labels[nan_mask][env_labels[nan_mask] == i]][:, 1],
                            mds_.embedding_[-n_parts:, :][p_labels[nan_mask][env_labels[nan_mask] == i]][:, 1])).mean(0)
        x2, y2 = mds_.embedding_[env_labels[nan_mask] == i][:, 0], mds_.embedding_[env_labels[nan_mask] == i][:, 1]

        ax = plt.subplot(2, n_envs//2, c)
        c += 1

        ax.scatter(np.vstack((mds_.embedding_[:n_parts, 0], mds_.embedding_[-n_parts:, 0])).mean(0),
                   np.vstack((mds_.embedding_[:n_parts, 1], mds_.embedding_[-n_parts:, 1])).mean(0), marker="+",
                   c="white", edgecolor="k", s=300, linewidth=3, zorder=2)
        # ax.scatter(mds_.embedding_[env_labels[nan_mask] == i, 0], mds_.embedding_[env_labels[nan_mask] == i, 1],
        #            s=650, c=c_labels[nan_mask][env_labels[nan_mask] == i], vmin=0, vmax=n_parts-1, alpha=1.,
        #            cmap="cool", edgecolor='k', linewidth=0., marker='s')
        ax.scatter(mds_.embedding_[env_labels[nan_mask] == i, 0], mds_.embedding_[env_labels[nan_mask] == i, 1],
                   s=400, c=c_labels[nan_mask][env_labels[nan_mask] == i], vmin=0, vmax=n_parts - 1, alpha=1.,
                   cmap="cool", edgecolor='k', linewidth=0., marker="s")
        ax.set_facecolor('black')

        for j in range(x1.shape[0]):
            ax.plot(np.hstack((x1[j], x2[j])), np.hstack((y1[j], y2[j])), c="white", linewidth=3,
                    linestyle=":")

        ax.set_ylim([-.8, .8])
        ax.set_yticks(np.linspace(-.8, .8, 2))
        ax.set_xlim([-.8, .8])
        ax.set_xticks(np.linspace(-.8, .8, 2))
        ax.set_aspect('equal')
        ax.set_ylabel('nMDS dim 1\n($abu$)', weight='bold', labelpad=-40)
        ax.set_xlabel('nMDS dim 2\n($abu$)', weight='bold', labelpad=-10)
        plt.setp(ax.spines.values(), color="gray", linewidth=4)

    plt.tight_layout()
    plt.show()
    return fig


def plot_similarity_parts_matrix(animals, p):
    rsm_parts_animals = get_rsm_partitioned_sequences(animals, p)
    rsm_parts_ordered, rsm_parts_averaged = get_rsm_partitioned_similarity(rsm_parts_animals, animals, False, False)

    labels = rsm_parts_animals['cannon_labels'][9:]

    fig = plt.figure(figsize=(12, 5))
    for i, e in enumerate(np.unique(labels[:, 0])):
        mask = labels[:, 0] == e
        sub_mat = rsm_parts_averaged[9:, :9][mask]
        sub_mat = sub_mat[np.eye(sub_mat.shape[0]).astype(bool)][np.newaxis].reshape(3,3)
        ax = plt.subplot(2, 5, i+1)
        ax.imshow(sub_mat.T, vmin=0., vmax=.45, cmap='plasma')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.setp(ax.spines.values(), color='k', linewidth=4)
    plt.show()
    return fig


def plot_rsm_parts_examples(rsm_parts_ordered, a1=1, a2=2, vmax=.6):
    target_rsms = np.array([[a1, 0], [a1, 1], [a1, 2], [a2, 0], [a2, 1], [a2, 2]])
    sns.set(style='dark', font_scale=2)
    fig = plt.figure(figsize=(16, 12))
    for i, (t_animal, t_sequence) in enumerate(target_rsms):
        ax = plt.subplot(target_rsms.shape[1], target_rsms.shape[0] // target_rsms.shape[1], i+1)

        rsm_copy = deepcopy(rsm_parts_ordered[t_sequence, t_animal])
        nan_mask = ~np.isnan(rsm_copy[np.eye(rsm_copy.shape[0]).astype(bool)])
        rsm_copy = rsm_copy[nan_mask, :][:, nan_mask]
        np.fill_diagonal(rsm_copy, np.nan)
        ax.imshow(rsm_copy, cmap='inferno', vmin=-.2, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        if i == 0:
            ax.set_ylabel("Mouse A", weight="bold")
        elif i == 3:
            ax.set_ylabel("Mouse B", weight="bold")
        ax.set_title(f'Sequence {t_sequence+1}', weight='bold', pad=10)
        plt.setp(ax.spines.values(), color='k', linewidth=4)
    plt.tight_layout()
    plt.show()
    return fig


def plot_partitioned_rsm_similarity(df_animal_similarity):
    sns.set(font_scale=2.5, palette='muted', style='dark')
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot()
    sns.barplot(data=df_animal_similarity, x='Sequence', y='Fit', hue='Data', edgecolor='k', width=.5,
                        saturation=1.,
                        linewidth=4, errcolor='k', errwidth=4, capsize=.05, palette="muted")
    ax.set_ylabel('Across-animal\nsimilarity ($Tau$)', weight='bold')
    ax.set_xlabel('Sequence', weight='bold')
    ax.set_xticklabels(['1', '2', '3'])
    ax.set_ylim([-.1, .6])
    plt.setp(ax.spines.values(), color='k', linewidth=4)
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()
    return fig


def plot_partitioned_rsm_predictions(df_animals):
    sns.set(font_scale=2.5, palette='muted', style='dark')
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot()
    sns.barplot(data=df_animals, x='Sequence', y='Decoding Accuracy', hue='Data', edgecolor='k', width=.5, saturation=1.,
                    linewidth=4, errcolor='k', errwidth=4, capsize=.05, palette="muted")
    ax.set_ylabel('Across-animal\nRSM decoding ($R^2$)', weight='bold')
    ax.set_ylim([-.25, .75])
    ax.set_xlabel('Sequence', weight='bold')
    ax.set_xticklabels(['1', '2', '3'])
    plt.setp(ax.spines.values(), color='k', linewidth=4)
    plt.legend(bbox_to_anchor=(1,1))
    plt.show()
    return fig


def plot_animal_id_predictions(df_animal_ID):
    sns.set(font_scale=2.5, palette='muted', style='dark')
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot()
    sns.barplot(data=df_animal_ID, x='Sequence', y='Correct', hue='Data', edgecolor='k', width=.5, saturation=1.,
                    linewidth=4, errcolor='k', errwidth=4, capsize=.05, errorbar=None, palette="muted")
    ax.set_ylabel('Animal decoding (prob)', weight='bold')
    ax.set_ylim([0, .75])
    ax.set_xlabel('Sequence', weight='bold')
    ax.set_xticklabels(['1', '2', '3'])
    plt.setp(ax.spines.values(), color='k', linewidth=4)
    plt.legend(bbox_to_anchor=(1.65,1.))
    plt.show()
    return fig


def plot_rsm_partitioned_similarity_resampled(df, n_samples):
    sns.set(style="dark", font_scale=2.25)
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    # sns.barplot(data=df, x="N cells", y="R", estimator="mean", errorbar="se")
    sns.lineplot(data=df.groupby("N cells").mean(), x="N cells", y="R",
                 color="gray", linestyle="--", linewidth=4., zorder=0)
    sns.scatterplot(data=df.groupby("N cells").mean(), x="N cells", y="R",
                    s=150, c=df.groupby("N cells").mean()["R"], edgecolor="k",
                    linewidth=4., zorder=1)
    ax.set_ylim([0, 1.])
    ax.set_xticks(n_samples[::4] + [1000])
    ax.set_xticklabels(n_samples[::4] + [1000], rotation=90)
    ax.set_ylabel("Across-animal\nsimilarity ($R$)", weight="bold")
    ax.set_xlabel("Sub-sampled cells", weight="bold")
    plt.setp(ax.spines.values(), color="k", linewidth=4.)
    plt.tight_layout()
    plt.show()
    return fig