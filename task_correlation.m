%% TASK_CORRELATION
% Task-dependent correlation analysis between fNIRS short channels and multimodal physiology.
%
% Operations performed:
%   1. Loads preprocessed fNIRS and extracts Short-Separation (SC) PCA components for HbO/HbR.
%   2. Aligns auxiliary physiology (ECG HR, PPG HR, Respiration, EDA Tonic/Phasic, SpO2).
%   3. Segments signals across experimental states: Baseline (30s), FT (210s), BH (105s), COMB (105s).
%   4. Computes 8x8 correlation matrices using both raw concatenated and block-detrended approaches.
%   5. Saves <subject>_TaskCorrelation.mat containing correlation structures and regressor matrices.

clearvars -except subj_dir; close all; clc;

%% 1. Load fNIRS data and valid triggers
if ~exist('subj_dir', 'var') || isempty(subj_dir)
    default_data_dir = fullfile(pwd, 'DATASET');
    script_dir = fileparts(mfilename('fullpath'));
    candidates = {fullfile(pwd, 'DATASET'), fullfile(script_dir, '..', '..', 'DATASET'), fullfile(script_dir, '..', 'DATASET'), pwd};
    for i = 1:numel(candidates)
        if exist(candidates{i}, 'dir') && ~isempty(dir(fullfile(candidates{i}, '20*')))
            default_data_dir = candidates{i};
            break;
        end
    end
    subj_dir = uigetdir(default_data_dir, 'Select Subject Folder');
    if isequal(subj_dir, 0), disp('Operation canceled by user.'); return; end
end

[~, subject_name] = fileparts(subj_dir);
file_snirf     = fullfile(subj_dir, [subject_name, '_Satori2.snirf']);
file_snirf_raw = fullfile(subj_dir, [subject_name, '.snirf']);
file_csv       = fullfile(subj_dir, [subject_name, '_ProcessedPhysio.csv']);
file_valid     = fullfile(subj_dir, [subject_name, '_ValidTrials.mat']);

assert(exist(file_snirf, 'file') == 2, 'Satori SNIRF not found: %s', file_snirf);
assert(exist(file_snirf_raw, 'file') == 2, 'Raw SNIRF not found: %s', file_snirf_raw);
assert(exist(file_csv, 'file') == 2, 'Physiology CSV not found: %s', file_csv);
assert(exist(file_valid, 'file') == 2, 'ValidTrials MAT not found: %s. Run physio_general.m first.', file_valid);

fprintf('Processing task correlations for: %s\n', subject_name);

data_time_series = h5read(file_snirf, '/nirs/data1/dataTimeSeries')'; 
time_vector = h5read(file_snirf, '/nirs/data1/time');
num_time_points = length(time_vector);
fs_fnirs = 1 / mean(diff(time_vector));

src_pos = h5read(file_snirf_raw, '/nirs/probe/sourcePos3D')';
det_pos = h5read(file_snirf_raw, '/nirs/probe/detectorPos3D')';

load(file_valid, 'valid_onsets_FT', 'valid_onsets_BH', 'valid_onsets_COMB');
onsets_stim1 = valid_onsets_FT;
onsets_stim2 = valid_onsets_BH;
onsets_stim3 = valid_onsets_COMB;

total_columns = size(data_time_series, 2);
num_channels = floor(total_columns / 2);

ml_src = zeros(num_channels, 1);
ml_det = zeros(num_channels, 1);
for ch = 1:num_channels
    path_hbo = sprintf('/nirs/data1/measurementList%d', ch);
    ml_src(ch) = double(h5read(file_snirf, [path_hbo, '/sourceIndex']));
    ml_det(ch) = double(h5read(file_snirf, [path_hbo, '/detectorIndex']));
end

scale_factor_satori = 10.0;
y = zeros(num_time_points, 3, num_channels);
for ch = 1:num_channels
    y(:, 1, ch) = data_time_series(:, ch) / scale_factor_satori;              % HbO (uM)
    y(:, 2, ch) = data_time_series(:, ch + num_channels) / scale_factor_satori; % HbR (uM)
    y(:, 3, ch) = y(:, 1, ch) + y(:, 2, ch);                                  % HbT
end

%% 2. Short-channel PCA (SC PCA)
sd_dist = zeros(num_channels, 1);
for ch = 1:num_channels
    sd_dist(ch) = norm(src_pos(ml_src(ch), :) - det_pos(ml_det(ch), :));
end
short_channels = find(sd_dist < 15)';
assert(~isempty(short_channels), 'No short channels (< 15 mm) identified.');

% SC PCA HbO
sc_data_hbo = squeeze(y(:, 1, short_channels));
R_sc_hbo = corr(sc_data_hbo, 'Rows', 'complete');
mean_corr_hbo = zeros(length(short_channels), 1);
for ch = 1:length(short_channels)
    other_idx = setdiff(1:length(short_channels), ch);
    mean_corr_hbo(ch) = mean(R_sc_hbo(ch, other_idx), 'omitnan');
end
sc_clean_hbo = short_channels(mean_corr_hbo >= 0.1);
[~, score_hbo] = pca(zscore(squeeze(y(:, 1, sc_clean_hbo))));
sc_pca1_hbo = zscore(score_hbo(:, 1));
if mean(corr(sc_pca1_hbo, squeeze(y(:, 1, sc_clean_hbo)))) < 0
    sc_pca1_hbo = -sc_pca1_hbo; % Align sign with scalp channels
end

% SC PCA HbR
sc_data_hbr = squeeze(y(:, 2, short_channels));
R_sc_hbr = corr(sc_data_hbr, 'Rows', 'complete');
mean_corr_hbr = zeros(length(short_channels), 1);
for ch = 1:length(short_channels)
    other_idx = setdiff(1:length(short_channels), ch);
    mean_corr_hbr(ch) = mean(R_sc_hbr(ch, other_idx), 'omitnan');
end
sc_clean_hbr = short_channels(mean_corr_hbr >= 0.1);
[~, score_hbr] = pca(zscore(squeeze(y(:, 2, sc_clean_hbr))));
sc_pca1_hbr = zscore(score_hbr(:, 1));
if mean(corr(sc_pca1_hbr, squeeze(y(:, 2, sc_clean_hbr)))) < 0
    sc_pca1_hbr = -sc_pca1_hbr; % Align sign with scalp channels
end

%% 3. Load and align physiological regressors
physio_table = readtable(file_csv); 
t_phys = physio_table.Time_s;

resp_aligned       = interp1(t_phys, physio_table.Resp_Clean, time_vector, 'linear', 'extrap');
eda_tonic_aligned  = interp1(t_phys, physio_table.EDA_Tonic, time_vector, 'linear', 'extrap');
eda_phasic_aligned = interp1(t_phys, physio_table.EDA_Phasic, time_vector, 'linear', 'extrap');
hr_ppg_aligned     = interp1(t_phys, physio_table.HR_PPG, time_vector, 'linear', 'extrap');
hr_ecg_aligned     = interp1(t_phys, physio_table.HR_ECG, time_vector, 'linear', 'extrap');
spo2_aligned       = interp1(t_phys, physio_table.SpO2_Clean, time_vector, 'linear', 'extrap');

regressor_names = { ...
    'HR (ECG)', 'HR (PPG)', 'Resp', 'EDA Tonic', ...
    'EDA Phasic', 'SpO2', 'SC PCA HbO', 'SC PCA HbR' ...
};

raw_regressors = [ ...
    hr_ecg_aligned(:), hr_ppg_aligned(:), resp_aligned(:), ...
    eda_tonic_aligned(:), eda_phasic_aligned(:), spo2_aligned(:), ...
    sc_pca1_hbo(:), sc_pca1_hbr(:) ...
];
regressor_matrix = zscore(raw_regressors);

%% 4. Define state masks
first_onset = min([onsets_stim1; onsets_stim2; onsets_stim3]);
idx_baseline = time_vector >= (first_onset - 30) & time_vector < first_onset;

task_duration = 21; % seconds
idx_ft = false(num_time_points, 1);
for i = 1:length(onsets_stim1)
    idx_ft = idx_ft | (time_vector >= onsets_stim1(i) & time_vector <= (onsets_stim1(i) + task_duration));
end

idx_bh = false(num_time_points, 1);
for i = 1:length(onsets_stim2)
    idx_bh = idx_bh | (time_vector >= onsets_stim2(i) & time_vector <= (onsets_stim2(i) + task_duration));
end

idx_comb = false(num_time_points, 1);
for i = 1:length(onsets_stim3)
    idx_comb = idx_comb | (time_vector >= onsets_stim3(i) & time_vector <= (onsets_stim3(i) + task_duration));
end

%% 5. Compute correlation matrices (Raw vs Detrended)
% Raw concatenated blocks
R_baseline_raw = corr(regressor_matrix(idx_baseline, :), 'Rows', 'complete');
R_ft_raw       = corr(regressor_matrix(idx_ft, :),       'Rows', 'complete');
R_bh_raw       = corr(regressor_matrix(idx_bh, :),       'Rows', 'complete');
R_comb_raw     = corr(regressor_matrix(idx_comb, :),     'Rows', 'complete');

% Block-by-block detrended blocks
R_baseline_det = corr(detrend(regressor_matrix(idx_baseline, :)), 'Rows', 'complete');

ft_reg_det = [];
for i = 1:length(onsets_stim1)
    idx = time_vector >= onsets_stim1(i) & time_vector <= (onsets_stim1(i) + task_duration);
    ft_reg_det = [ft_reg_det; detrend(regressor_matrix(idx, :))]; %#ok<AGROW>
end
R_ft_det = corr(ft_reg_det, 'Rows', 'complete');

bh_reg_det = [];
for i = 1:length(onsets_stim2)
    idx = time_vector >= onsets_stim2(i) & time_vector <= (onsets_stim2(i) + task_duration);
    bh_reg_det = [bh_reg_det; detrend(regressor_matrix(idx, :))]; %#ok<AGROW>
end
R_bh_det = corr(bh_reg_det, 'Rows', 'complete');

comb_reg_det = [];
for i = 1:length(onsets_stim3)
    idx = time_vector >= onsets_stim3(i) & time_vector <= (onsets_stim3(i) + task_duration);
    comb_reg_det = [comb_reg_det; detrend(regressor_matrix(idx, :))]; %#ok<AGROW>
end
R_comb_det = corr(comb_reg_det, 'Rows', 'complete');

% Full session correlations
R_raw_full       = corr(regressor_matrix, 'Rows', 'complete');
R_detrended_full = corr(detrend(regressor_matrix), 'Rows', 'complete');

%% 6. Plot correlation heatmaps
n_col = 128;
r_map = [linspace(0.2, 1, n_col), linspace(1, 0.85, n_col)]';
g_map = [linspace(0.45, 1, n_col), linspace(1, 0.25, n_col)]';
b_map = [linspace(0.75, 1, n_col), linspace(1, 0.25, n_col)]';
rwb_cmap = [r_map, g_map, b_map];

cond_titles_det = {'1. Baseline (Detrended)', '2. Fingertapping (Detrended)', ...
                   '3. Breath Holding (Detrended)', '4. Combined (Detrended)'};
R_det_all = {R_baseline_det, R_ft_det, R_bh_det, R_comb_det};

fig_det = figure('Name', 'Task Correlation: Detrended', 'Color', 'w', 'Position', [100 100 950 850]);
for c = 1:4
    subplot(2, 2, c);
    plot_correlation_heatmap(R_det_all{c}, regressor_names, cond_titles_det{c}, rwb_cmap);
end
sgtitle(sprintf('Task-Based Correlation (Detrended) - %s', subject_name), 'FontWeight', 'bold');

fig_det_path = fullfile(subj_dir, [subject_name, '_TaskCorrelation_Detrended_2x2.png']);
try exportgraphics(fig_det, fig_det_path, 'Resolution', 300); catch; saveas(fig_det, fig_det_path); end

%% 7. Save task correlation MAT file
file_task_corr_mat = fullfile(subj_dir, [subject_name, '_TaskCorrelation.mat']);
save(file_task_corr_mat, 'R_detrended_full', 'R_raw_full', 'R_baseline_det', ...
    'R_ft_det', 'R_bh_det', 'R_comb_det', 'regressor_names', 'regressor_matrix');
fprintf('Saved task correlation MAT file: %s\n', file_task_corr_mat);

%% Auxiliary function: heatmap display
function plot_correlation_heatmap(R, names, title_str, cmap)
    imagesc(R);
    colorbar;
    clim([-1 1]); 
    colormap(cmap);
    axis square;
    
    n_vars = length(names);
    set(gca, 'XTick', 1:n_vars, 'XTickLabel', names, ...
             'YTick', 1:n_vars, 'YTickLabel', names, ...
             'FontSize', 9, 'FontWeight', 'bold');
    xtickangle(45);
    title(title_str, 'FontSize', 11, 'FontWeight', 'bold');
    
    for i = 1:n_vars
        for j = 1:n_vars
            val = R(i, j);
            if isnan(val), txt = 'NaN';
            else, txt = sprintf('%+.2f', val); end
            
            if abs(val) > 0.45, col = 'w'; else, col = 'k'; end
            text(j, i, txt, 'HorizontalAlignment', 'center', ...
                 'VerticalAlignment', 'middle', 'Color', col, ...
                 'FontSize', 8, 'FontWeight', 'bold');
        end
    end
end
