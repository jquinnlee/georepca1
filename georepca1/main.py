# Imports and data loading #####################################################################################

from utils import *
from plots import *
p = r'/Users/jquinnlee/Desktop/georepca1/georepca1'
os.chdir(p)

# Define animal names in dataset
animals = ['QLAK-CA1-08', 'QLAK-CA1-30', 'QLAK-CA1-50', 'QLAK-CA1-51', 'QLAK-CA1-56', 'QLAK-CA1-74', 'QLAK-CA1-75']
animal = animals[2] # select example animal from list

# Convert original matlab files to joblib files (large) that can be loaded efficiently in Python
for animal in tqdm(animals):
    mat2joblib(animal, p)

# Generate and save dictionary containing behavior and environment labels for all animals (lighter-weight dataset)
generate_behav_dict(animals, p, format="joblib")

# Load third animal in dataset as example
animal = animals[2]
dat = load_dat(animal, p, format="joblib")

# FIGURE 1 #############################################################################################################

# Figure 1C
# Use example cells and sessions from paper
example_cells_idx = np.array([17, 18, 28, 36, 62, 69, 91, 255, 347, 497])
example_days_idx = np.arange(20, 31)
# Index example maps and create contours from spatial footprints ("SFPs")
example_maps = {"smoothed": dat[animal]["maps"]["smoothed"][:, :, example_cells_idx, :][:, :, :, example_days_idx]}
example_sfps = trace_sfps(dat[animal]['SFPs'])[:, :, example_cells_idx, :][:, :, :, example_days_idx]
plot_maps(example_maps, animal,  p, example_sfps, example_cells_idx, unsmoothed=False, cmap='viridis')

# Figure 1E-F
# Calculate split-half spatial reliability for all recorded cells within each session
for animal in animals:
    dat = load_dat(animal, p, format="joblib")
    p_vals, _ = get_shr_within(dat, animal, nsims=10)
    joblib.dump(p_vals, os.path.join(p, "results", f"{animal}_shr"))

# fit bayesian decoder with flat priors to each day position within day (cross-validated)
# return decoding error (mean each day) for all animals and save to results folder
within_decoding = {}
for animal in animals:
    within_decoding[animal] = {}
    dat = load_dat(animal, p, format="joblib")
    within_decoding[animal]['envs'] = dat[animal]['envs'].squeeze()
    within_decoding[animal]['decoding_error'] = \
        decode_position_within(dat[animal]['position'].T, dat[animal]['trace'].T, dat[animal]['maps']['smoothed'])[0]
joblib.dump(within_decoding, os.path.join(p, "results", "within_decoding"))

# Load split-half reliability p values from all animals and all sessions into single dataframe
# Plot SHR and decoding error across all recordings
df_pvals = get_all_shr_pvals(animals, p)
# STATS One-way ANOVA
plot_shr_pvals(df_pvals)
formula = 'SHR ~ C(Day)'
lm = ols(formula, df_pvals).fit()
print(f"One-way ANOVA for SHR across days: \n{anova_lm(lm)}")
df_decoding = get_all_decoding_within(animals, p)
plot_decoding_within_days(df_decoding)
formula = 'Error ~ C(Day)'
lm = ols(formula, df_decoding).fit()
print(f"One-way ANOVA for decoding error across days: \n{anova_lm(lm)}")

# Figure 1H-I
# calculate the average map correlation across geometries, and plot as RSM and with non-metric MDS and dendrogram
mean_map_corr, _, labels, map_corr_animals_sequences = get_mean_map_corr(animals, p)
map_corr_dict = {"mean_map_corr": mean_map_corr,
                 "labels": labels,
                 "map_corr_animals_sequences": map_corr_animals_sequences}
joblib.dump(map_corr_dict, os.path.join(p, "results", "map_corr_envs"))
map_corr_dict = joblib.load(os.path.join(p, "results", "map_corr_envs"))
mean_map_corr, labels, map_corr_animals_sequences = (map_corr_dict["mean_map_corr"], map_corr_dict["labels"],
                                                     map_corr_dict["map_corr_animals_sequences"])

map_corr_envs = joblib.load(os.path.join(p, "results", "map_corr_envs"))
mean_map_corr, labels, map_corr_animals_sequences = (map_corr_envs["mean_map_corr"], map_corr_envs["labels"],
                                                     map_corr_envs["map_corr_animals_sequences"])

fig_h = plot_map_corr_rsm_ordered(mean_map_corr, labels)
fig_i = plot_map_corr_mds(mean_map_corr, labels)
fig_j = plot_map_corr_dendrogram(mean_map_corr, labels)

# Figure 1K
# Calculate similarity of representations across animals
df_map_corr_animals_sequences = get_rsm_similarity_animals_sequences(animals, p)
fig_k = plot_across_animal_similarity(df_map_corr_animals_sequences)

# two-way ANOVA for effect of sequence and shuffle on across-animal remapping similarity (whole rate map)
formula = 'Fit ~ C(Sequence) + C(Shuffle) + C(Sequence):C(Shuffle)'
lm = ols(formula, df_map_corr_animals_sequences).fit()
print(f"Two-way ANOVA for across-animal RSM similarity based on sequnece: \n{anova_lm(lm)}")

# FIGURE 2 #############################################################################################################

# Figure 2B
# Calculate and save cell-wise, partition-wise rsm for all animals from rate map data
for animal in animals:
    dat = load_dat(animal, p, format="joblib")
    rsm_parts, rsm_labels, rsm_cell_idx = get_cell_rsm_partitioned(dat[animal]['maps'], d_thresh=0)
    rsm_dict = {'RSM': rsm_parts, 'd_labels': rsm_labels[:, 0], 'p_labels': rsm_labels[:, 1],
                'cell_idx': rsm_cell_idx, 'envs': dat[animal]['envs']}
    joblib.dump(rsm_dict, os.path.join(p, "results", f'{animal}_rsm_partitioned'))
    del dat

# Combine averaged partition-wise RSM for all animals and sequences
rsm_parts_animals = get_rsm_partitioned_sequences(animals, p)

# Order partitioned rsms across animals
rsm_parts_ordered, rsm_parts_averaged = get_rsm_partitioned_similarity(rsm_parts_animals, animals,
                                                                       False, False)
fig_b = plot_rsm_parts_averaged(rsm_parts_averaged, vmax=0.5)

# Figure 2C
# Embed partitioned RSM in 2d with non-metric multidimensional scaling
fig_c = plot_rsm_parts_mds(rsm_parts_averaged)

# Figure 2D
# Plot the average similarity of each partition in each shape to the partition in square
fig_d = plot_similarity_parts_matrix(animals, p)

# Figure 2E
# Plot partitioned RSMs for two animals in each sequence
fig_e = plot_rsm_parts_examples(rsm_parts_ordered, 0, 2)

# Figure 2F,H-I
# Measure the similarity of partitioned RSM across animals
# Decode RSM values across animals
# Decode animal identity from individual RSM
df_animal_similarity, df_animal_ID, df_animals, df_sequences = predict_rsm_animals(animals, rsm_parts_animals)

fig_g = plot_partitioned_rsm_similarity(df_animal_similarity)
fig_h = plot_partitioned_rsm_predictions(df_animals)
fig_i = plot_animal_id_predictions(df_animal_ID)

# Figure 2G
# Measure similarity of partitioned RSM across animals, resampling different numbers of cells
rsm_parts_animals = get_rsm_partitioned_sequences(animals, p)
n_samples = [10, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000]
df = get_partitioned_rsm_similarity_resampled(animals, rsm_parts_animals, n_samples)
fig_g = plot_rsm_partitioned_similarity_resampled(df, n_samples)

# FIGURE 3 #############################################################################################################

# Load behavioral dataset and environment names of first animal (cannon order)
behav_dict = joblib.load(os.path.join(p, "data", "behav_dict"))
envs = behav_dict[animals[0]]["envs"][:11]

# Simulate simple model predictions with RSA method using heuristic models of cognitive mapping
# Three heuristic models will be based on similarity in Euclidean space, local boundaries, and animal trajectory
euc_similarity = get_euclidean_similarity_partitioned(envs)
bound_similarity = get_boundary_similarity_partitioned(envs)
transition_similarity = get_traj_similarity_partitioned(animals, envs, p)

# Figure 3D
# Plot partitioned RSM results for Euclidean, local boundary, and animal trajectory heuristic models
fig_d1 = plot_rsm_parts_averaged(euc_similarity, vmin=0, vmax=1.)
fig_d2 = plot_rsm_parts_averaged(bound_similarity, vmin=0, vmax=1.)
fig_d3 = plot_rsm_parts_averaged(transition_similarity, vmin=0., vmax=1.)

# Figure 3E
# Calculate similarity to true CA1 RSM result for each model, along with noise ceiling minmax
noise_margin_agg, rsm_mask_agg = get_noise_margin(rsm_parts_ordered)
# get bootstrap fits for each heuristic model
euc_fit, euc_se, euc_p_val = get_rsm_fit_bootstrap(rsm_parts_averaged, euc_similarity)
bound_fit, bound_se, bound_p_val = get_rsm_fit_bootstrap(rsm_parts_averaged, bound_similarity)
traj_fit, traj_se, traj_p_val = get_rsm_fit_bootstrap(rsm_parts_averaged, transition_similarity)
fig_e = plot_heuristic_model_fits(euc_fit, euc_se, bound_fit, bound_se, traj_fit, traj_se,
                                  noise_margin_agg)

# FIGURE 4 #############################################################################################################

# Simulate Grid cells, Place cells (naive model), and Boundary vector cells with RatInABox following animal trajectories
# First illustrate with an example of riab application
plot_riab_example(animal, p)

# Simulate grid cells, boundary vector cells, and naive place cells across sessions from animal trajectories
n_features = 200
# simulate and save agent and basis set using ratinabox for target animals
bases = ["GC", "BVC", "PC"]
for basis in bases:
    simulate_bases(animals, p, n_features, bases=[basis])

# build rate maps from simulated bases
for basis in bases:
    get_model_maps(animals, os.path.join(p), feature_types=[basis], compute_rsm=False)

# simulate place cell
pc_rate_maps = get_solstad_pc_population(n_pcs=1000, threshold=True)
joblib.dump(pc_rate_maps, os.path.join(p, "results", "riab", "solstad_gc2pc_receptive_fields_th"))

# grid-to-place cell models with and without boundary-tethering method (Solstad et al. (2006) + Keinath et al. (2018))
np.random.seed(2023)
pc_receptive_fields = joblib.load(os.path.join(p, "results", "riab", "solstad_gc2pc_receptive_fields_th"))
get_gc2pc_maps(animals, p, pc_receptive_fields, threshold=True, compute_rsm=True)
get_bt_gc2pc_maps(animals, p, n_pc=200, threshold=True, compute_rsm=True)

# boundary vector-to-place cell model (Barry et al., (2006) + Grieves et al., (2018))
for animal in animals:
    get_bvc2pc_maps(animal, p, nPCs=200, compute_rsm=True)

# Plot example maps
model_PCs = ["GC2PC_th", "bt_GC2PC_th", "BVC2PC"]
for model in model_PCs:
    model_maps = joblib.load(os.path.join(p, "results", "riab", f"{animal}_{model}_maps"))
    example_maps = deepcopy(model_maps)
    example_cells_idx = np.arange(1, 200, 25)
    example_days_idx = np.arange(20, 31)
    example_maps["smoothed"] = example_maps["smoothed"][:, :, example_cells_idx, :][:, :, :, example_days_idx]
    _ = plot_maps(example_maps, animal,  p, False, example_cells_idx,
                  unsmoothed=False, make_dir=False, cmap='viridis')

# Build dictionary for model RSM with average results, and plot resulting RSM
get_rsm_model_dict([animal], model_PCs, p_models=os.path.join(p, "results", "riab"))
rsm_models = joblib.load(os.path.join(p, "results", "riab", "rsm_models"))
for model in model_PCs:
    _ = plot_rsm_parts_averaged(rsm_models[model]["averaged"], vmin=-.1, vmax=1., cmap="inferno")

# Learn successor features from PC or BVC model bases with temporal difference learning rule (de Cothi et al., 2019)
bases = ["PC", "BVC"]
sr_gamma = 0.999
sr_alpha = (50./30.)*10**(-3)
for basis in bases:
    simulate_basis2sf([animal], p, basis, sr_gamma, sr_alpha)

# Build rate maps and RSMs from model successor features
for basis in bases:
    feature_type = f"{basis}2SF_{sr_gamma:.5f}gamma_{sr_alpha:.5f}alpha"
    get_model_maps([animal], p, feature_types=[feature_type], compute_rsm=True)

# Plot example successor features
for basis in bases:
    feature_type = f"{basis}2SF_{sr_gamma:.5f}gamma_{sr_alpha:.5f}alpha"
    model_maps = joblib.load(os.path.join(p, "results", "riab", f"{animal}_{feature_type}_maps"))
    example_maps = deepcopy(model_maps)
    example_cells_idx = np.arange(1, 200, 25)
    example_days_idx = np.arange(20, 31)
    example_maps["smoothed"] = example_maps["smoothed"][:, :, example_cells_idx, :][:, :, :, example_days_idx]
    _ = plot_maps(example_maps, animal,  p, False, example_cells_idx,
              unsmoothed=False, make_dir=False, cmap='viridis')

# Build dictionary for model RSM with average results, and plot resulting RSM
feature_types = [f"{basis}2SF_{sr_gamma:.5f}gamma_{sr_alpha:.5f}alpha" for basis in bases]
get_rsm_model_dict([animal], feature_types, p_models=os.path.join(p, "results", "riab"))
rsm_models = joblib.load(os.path.join(p, "results", "riab", "rsm_models"))
for feature_type in feature_types:
    _ = plot_rsm_parts_averaged(rsm_models[feature_type]["averaged"], vmin=-.1, vmax=1., cmap="inferno")

# Figure 4D
# Measure and plot fit (Kendall's Tau) between model RSM results and actual CA1 data
feature_types = ["PC", "GC2PC_th", f"PC2SF_{sr_gamma:.5f}gamma_{sr_alpha:.5f}alpha", "bt_GC2PC_th", "BVC2PC",
                 f"BVC2SF_{sr_gamma:.5f}gamma_{sr_alpha:.5f}alpha"]
feature_names = ["PC", "GC2PC", "PC2SF", "bt-GC2PC", "BVC2PC", "BVC2SF"]

get_rsm_model_dict(animals, feature_types, p_models=os.path.join(p, "results", "riab"))
rsm_models = joblib.load(os.path.join(p, "results", "riab", "rsm_models"))

# Load the rsm computed from actual data
rsm_parts_animals = get_rsm_partitioned_sequences(animals, p)
rsm_parts_ordered, rsm_parts_averaged = get_rsm_partitioned_similarity(rsm_parts_animals, animals,
                                                                       False, False)
# Compute upper and lower bound of noise ceiling for each animal, use mask is to drop triangle and nans from rsm
noise_margin_agg, rsm_mask_agg = get_noise_margin(rsm_parts_ordered)

# Compute model fits to ca1 data and organize into dataframe for stats and plotting
df_agg_bootstrap = get_ca1_model_fits(rsm_parts_averaged, rsm_models, feature_types)

# Plot model fits
fig_d = plot_model2ca1_similarities(df_agg_bootstrap, noise_margin_agg, feature_names)

# Figure 4E
# Measure model fits to CA1 RSM across repetitions of the geometric sequence
df_agg_bootstrap_sequences, n_seq = get_ca1_model_fits_sequences(rsm_parts_ordered, rsm_models, feature_types,
                                                                 feature_names)
fig_e = plot_ca1_model_fits_sequences(df_agg_bootstrap_sequences, n_seq, noise_margin_agg)

# Figure 4F
# Measure model fits for specific subsets of comparisons: same environment/different partitions; different environment
# same partitions; different environment/different partitions
df_hypo_comps = get_ca1_model_fit_subsets(rsm_parts_animals, rsm_parts_averaged)
fig_f = plot_ca1_model_fit_subsets(df_hypo_comps, feature_names)

# FIGURE 5 #############################################################################################################

# Figure 5A
# plot example BVC tunings ratinabox example receptive fields for 10 BVCs
for c in [1, 8, 16, 32]:
    _, fig = plot_riab_example(animal, p, const=c*10**(-2)) # const is 1e-2 of paper values

for s in [1/24, 1/12, 1/6, 1/3]:
    _, fig = plot_riab_example(animal, p, scalar=s**-1) # scalar is inverse of paper values

# Figure 5B
# Plot relationship between preferred bvc tuning distances with manipulation of alpha and beta parameter in distribution
fig_b1, fig_b2 = plot_bvc_beta_distribution(vals=[0.25, 0.5, 0.75, 1., 2.5, 5.])

# FIGURE 6 #############################################################################################################

