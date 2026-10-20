function results = integrated_spatial_agreement_validation(subject_dir, output_dir, show_figures)
% INTEGRATED_SPATIAL_AGREEMENT_VALIDATION
%
% Integrated, leakage-resistant validation for an FT/BH/COMB fNIRS dataset.
%
% The analysis deliberately distinguishes three roles:
%   1. A simultaneous Homer GLM (FT + BH + COMB) with short-channel PCA
%      provides an empirical, SC-corrected FT reference.
%   2. Ridge mappings are fitted and selected using BH blocks only, with
%      nested leave-one-BH-block-out cross-validation.
%   3. Frozen mappings are transferred to COMB and evaluated against the FT
%      reference using continuous spatial metrics and FDR-controlled maps.
%
% Usage:
%   results = integrated_spatial_agreement_validation();
%   results = integrated_spatial_agreement_validation(subject_dir);
%   results = integrated_spatial_agreement_validation(subject_dir, output_dir);
%
% Required subject files:
%   <subject>_Satori2.snirf
%   <subject>.snirf
%   <subject>_ProcessedPhysio.csv
%   <subject>_ValidTrials.mat
%   <subject>_TaskCorrelation.mat
%   hmrDeconvHRF_DriftSS.m
%
% Outputs are descriptive at the subject level. Model conclusions should be
% based on paired subject-level results across the complete dataset.

%% 1. Load data, files and parameter selection
% Base dataset directory
default_dataset_dir = fullfile(pwd, 'DATASET');
if ~exist(default_dataset_dir, 'dir')
    default_dataset_dir = pwd;
end

% Interactive subject selection
if nargin < 1 || isempty(subject_dir)
    subject_dir = uigetdir(default_dataset_dir, 'Select the Subject Folder');
    if isequal(subject_dir, 0)
        disp('Operation canceled by the user.');
        results = struct();
        return;
    end
end
if nargin < 2 || isempty(output_dir)
    output_dir = fullfile(subject_dir, 'outputs');
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

[~, subject_id] = fileparts(subject_dir);
expected_satori = fullfile(subject_dir, [subject_id '_Satori2.snirf']);
if exist(expected_satori, 'file') ~= 2
    candidates = dir(fullfile(subject_dir, '*_Satori2.snirf'));
    assert(isscalar(candidates), ...
        'Folder name does not match the subject prefix and exactly one *_Satori2.snirf file was not found.');
    subject_id = erase(candidates(1).name, '_Satori2.snirf');
end
addpath(subject_dir);

% Configuration
scale_factor_satori = 10.0;
alpha = 0.05;
epoch_range = [-5, 40];
baseline_range = [-5, 0];
task_range = [2, 21];
evaluation_range = [0, 40];
homer_trange = [-5, 50];
lags_sec = [0, 2, 4, 6, 8, 10, 12];
lambda_grid = [0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000];
num_bootstrap = 2000;
random_seed = 20260911;
if nargin < 3 || isempty(show_figures)
    show_figures = true; % Set to true to show figures interactively on screen, false for headless mode
end

% Files
file_snirf = fullfile(subject_dir, [subject_id '_Satori2.snirf']);
file_raw = fullfile(subject_dir, [subject_id '.snirf']);
file_phys = fullfile(subject_dir, [subject_id '_ProcessedPhysio.csv']);
file_valid = fullfile(subject_dir, [subject_id '_ValidTrials.mat']);
file_corr = fullfile(subject_dir, [subject_id '_TaskCorrelation.mat']);

% Locate Homer deconvolution function (check search path, subject_dir, or local/parent homer2 folder)
if exist('hmrDeconvHRF_DriftSS.m', 'file') == 2
    file_homer = which('hmrDeconvHRF_DriftSS.m');
elseif exist(fullfile(subject_dir, 'hmrDeconvHRF_DriftSS.m'), 'file') == 2
    file_homer = fullfile(subject_dir, 'hmrDeconvHRF_DriftSS.m');
elseif exist(fullfile(pwd, 'homer2', 'hmrDeconvHRF_DriftSS.m'), 'file') == 2
    file_homer = fullfile(pwd, 'homer2', 'hmrDeconvHRF_DriftSS.m');
    addpath(fullfile(pwd, 'homer2'));
elseif exist(fullfile(fileparts(mfilename('fullpath')), '..', 'homer2', 'hmrDeconvHRF_DriftSS.m'), 'file') == 2
    file_homer = fullfile(fileparts(mfilename('fullpath')), '..', 'homer2', 'hmrDeconvHRF_DriftSS.m');
    addpath(fileparts(file_homer));
else
    error('Cannot find hmrDeconvHRF_DriftSS.m. Please ensure Homer2 is added to the MATLAB search path.');
end

required_files = {file_snirf, file_raw, file_phys, file_valid, file_corr, file_homer};
for k = 1:numel(required_files)
    assert(exist(required_files{k}, 'file') == 2, 'Missing required file: %s', required_files{k});
end

%% 2. Load fNIRS, geometry, events and aligned physiology
V = load(file_valid);
R = load(file_corr, 'regressor_matrix', 'regressor_names');
phys_table = readtable(file_phys);

time = double(h5read(file_snirf, '/nirs/data1/time'));
time = time(:);
data_ts = double(h5read(file_snirf, '/nirs/data1/dataTimeSeries'))';
num_time = numel(time);
fs = 1 / mean(diff(time));

% Determine number of channels dynamically: Satori exports HbO (1..N) and HbR (N+1..2N)
% (In the study's reference motor montage: 54 channels, 1..54 = HbO, 55..108 = HbR)
total_columns = size(data_ts, 2);
num_channels = floor(total_columns / 2);

assert(size(data_ts, 1) == num_time && size(data_ts, 2) >= 2 * num_channels, ...
    'Unexpected Satori SNIRF dimensions.');
assert(size(R.regressor_matrix, 1) == num_time, ...
    'Physiological regressor and fNIRS lengths differ.');

expected_names = ["HR (ECG)"; "HR (PPG)"; "Resp"; "EDA Tonic"; ...
                  "EDA Phasic"; "SpO2"; "SC PCA HbO"; "SC PCA HbR"];
actual_names = string(R.regressor_names(:));
assert(numel(actual_names) == numel(expected_names) && all(actual_names == expected_names), ...
    'Unexpected physiological regressor order.');

y_hbo = data_ts(:, 1:num_channels) / scale_factor_satori;
y_hbr = data_ts(:, num_channels + (1:num_channels)) / scale_factor_satori;
y_homer = zeros(num_time, 3, num_channels);
y_homer(:, 1, :) = reshape(y_hbo, num_time, 1, num_channels);
y_homer(:, 2, :) = reshape(y_hbr, num_time, 1, num_channels);
y_homer(:, 3, :) = y_homer(:, 1, :) + y_homer(:, 2, :);

src_pos = double(h5read(file_raw, '/nirs/probe/sourcePos3D'))';
det_pos = double(h5read(file_raw, '/nirs/probe/detectorPos3D'))';
ml_src = zeros(num_channels, 1);
ml_det = zeros(num_channels, 1);
sd_dist = zeros(num_channels, 1);
for ch = 1:num_channels
    group = sprintf('/nirs/data1/measurementList%d', ch);
    ml_src(ch) = double(h5read(file_snirf, [group '/sourceIndex']));
    ml_det(ch) = double(h5read(file_snirf, [group '/detectorIndex']));
    sd_dist(ch) = norm(src_pos(ml_src(ch), :) - det_pos(ml_det(ch), :));
end

short_channels = find(sd_dist < 15)';
long_channels = setdiff(1:num_channels, short_channels);
num_long = numel(long_channels);
assert(~isempty(short_channels), 'No short channels were identified with the 15-mm threshold.');

% Geometric C3 channel selection based on proximity to Detector 4.
if size(det_pos, 1) >= 4
    det4_pos = det_pos(4, :);
    dist_to_d4 = zeros(num_long, 1);
    for k = 1:num_long
        ch = long_channels(k);
        midpoint = (src_pos(ml_src(ch), :) + det_pos(ml_det(ch), :)) / 2;
        dist_to_d4(k) = norm(midpoint - det4_pos);
    end
    [~, order_d4] = sort(dist_to_d4, 'ascend');
    c3_channels = long_channels(order_d4(1:min(12, num_long)));
else
    c3_channels = long_channels(1:min(12, num_long));
end
[~, c3_idx] = ismember(c3_channels, long_channels);
c3_idx = c3_idx(c3_idx > 0);

onsets = {V.valid_onsets_FT(:), V.valid_onsets_BH(:), V.valid_onsets_COMB(:)};
assert(numel(onsets{1}) >= 2 && numel(onsets{2}) >= 3 && numel(onsets{3}) >= 2, ...
    'Insufficient valid FT, BH or COMB trials.');

t_inc = interp1(phys_table.Time_s, double(V.tInc_physio(:)), time, 'nearest', 'extrap');
t_inc = double(t_inc(:) > 0.5);

fprintf('\nSubject %s: %.4f Hz, %d long channels, %d short channels.\n', ...
    subject_id, fs, num_long, numel(short_channels));
fprintf('Valid trials: FT=%d, BH=%d, COMB=%d.\n', ...
    numel(onsets{1}), numel(onsets{2}), numel(onsets{3}));

%% 3. Empirical FT reference: simultaneous Homer GLM with SC PCA
stim = zeros(num_time, 3);
for cond = 1:3
    for k = 1:numel(onsets{cond})
        [~, idx] = min(abs(time - onsets{cond}(k)));
        stim(idx, cond) = 1;
    end
end

SD = struct();
SD.SrcPos = src_pos;
SD.DetPos = det_pos;
SD.MeasList = [ml_src, ml_det, ones(num_channels, 1), ones(num_channels, 1)];
SD.MeasListAct = ones(num_channels, 1);

% homer2 function: mean HRF for each channel and condition
reg = double(R.regressor_matrix);
fprintf('Estimating SC-corrected FT reference with simultaneous Homer GLM ...\n');
[homer_avg_hbo, ~, t_hrf, n_homer_trials, homer_new_hbo] = ...
    hmrDeconvHRF_DriftSS(y_homer, stim, time, SD, reg(:, 7), t_inc, ...
    homer_trange, 1, 1, [0.5 0.5], 0, 0, 3, 0);
[homer_avg_hbr, ~, t_hrf_hbr, ~, homer_new_hbr] = ...
    hmrDeconvHRF_DriftSS(y_homer, stim, time, SD, reg(:, 8), t_inc, ...
    homer_trange, 1, 1, [0.5 0.5], 0, 0, 3, 0);
assert(max(abs(t_hrf - t_hrf_hbr)) < 1e-9, 'Homer HbO and HbR time vectors differ.');

% amplitude estimates trial-by-trial
ft_reference_hrf_hbo = squeeze(homer_avg_hbo(:, 1, long_channels, 1));
ft_reference_hrf_hbr = squeeze(homer_avg_hbr(:, 2, long_channels, 1));
ft_reference_trials_hbo = trial_amplitudes_continuous(time, ...
    squeeze(homer_new_hbo(:, 1, long_channels)), onsets{1}, baseline_range, task_range);
ft_reference_trials_hbr = trial_amplitudes_continuous(time, ...
    squeeze(homer_new_hbr(:, 2, long_channels)), onsets{1}, baseline_range, task_range);

% t-test and FDR correction
ft_stats_hbo = activation_statistics(ft_reference_trials_hbo, 'right'); % right tail: HbO increase
ft_stats_hbr = activation_statistics(ft_reference_trials_hbr, 'left'); % left tail: HbR decrease
ft_mask_hbo = ft_stats_hbo.q < alpha & ft_stats_hbo.mean > 0;
ft_mask_hbr = ft_stats_hbr.q < alpha & ft_stats_hbr.mean < 0;
ft_mask_joint = ft_mask_hbo & ft_mask_hbr; % simultaneous HbO increase and HbR decrease

%% 4. Fixed, nested physiological feature bank
resp = reg(:, 3);
resp_activity = movstd(resp, max(5, round(4 * fs)), 0, 'omitnan');
phys_signals = [resp, resp_activity, reg(:, 1), reg(:, 6), reg(:, 5), reg(:, 4)];
phys_names = ["Resp raw", "Resp activity", "ECG-HR", "SpO2", "EDA phasic", "EDA tonic"];
[phys_lagged, phys_lag_names, phys_groups] = make_lag_bank(time, phys_signals, phys_names, lags_sec);

feature_hbo = [reg(:, 7), phys_lagged];
feature_hbr = [reg(:, 8), phys_lagged];
feature_names_hbo = ["SC PCA HbO", phys_lag_names];
feature_names_hbr = ["SC PCA HbR", phys_lag_names];

idx_sc = 1;
idx_resp = 1 + [phys_groups{1}, phys_groups{2}];
idx_hr = 1 + phys_groups{3};
idx_spo2 = 1 + phys_groups{4};
idx_eda_phasic = 1 + phys_groups{5};
idx_eda_tonic = 1 + phys_groups{6};

model_names = [ ...
    "M0 No correction";
    "M1 SC only";
    "M2 SC + respiration";
    "M3 SC + respiration + ECG-HR";
    "M4 SC + respiration + ECG-HR + SpO2";
    "M5 M4 + phasic EDA";
    "M6 M5 + tonic EDA"];
model_features = { ...
    [], ...
    idx_sc, ...
    [idx_sc, idx_resp], ...
    [idx_sc, idx_resp, idx_hr], ...
    [idx_sc, idx_resp, idx_hr, idx_spo2], ...
    [idx_sc, idx_resp, idx_hr, idx_spo2, idx_eda_phasic], ...
    [idx_sc, idx_resp, idx_hr, idx_spo2, idx_eda_phasic, idx_eda_tonic]};
num_models = numel(model_names);
num_features = cellfun(@numel, model_features(:));

%% 5. Baseline-corrected epochs
n_epoch = round(diff(epoch_range) * fs) + 1;
t_epoch = (0:n_epoch-1)' / fs + epoch_range(1);
idx_base = t_epoch >= baseline_range(1) & t_epoch < baseline_range(2);
idx_task = t_epoch >= task_range(1) & t_epoch <= task_range(2);
idx_eval = t_epoch >= evaluation_range(1) & t_epoch <= evaluation_range(2);

y_hbo_long = y_hbo(:, long_channels);
y_hbr_long = y_hbr(:, long_channels);
Ybh_hbo = extract_epochs(time, y_hbo_long, onsets{2}, t_epoch, idx_base);
Ybh_hbr = extract_epochs(time, y_hbr_long, onsets{2}, t_epoch, idx_base);
Yft_hbo = extract_epochs(time, y_hbo_long, onsets{1}, t_epoch, idx_base);
Yft_hbr = extract_epochs(time, y_hbr_long, onsets{1}, t_epoch, idx_base);
Ycomb_hbo = extract_epochs(time, y_hbo_long, onsets{3}, t_epoch, idx_base);
Ycomb_hbr = extract_epochs(time, y_hbr_long, onsets{3}, t_epoch, idx_base);

Xbh_hbo = extract_epochs(time, feature_hbo, onsets{2}, t_epoch, idx_base);
Xbh_hbr = extract_epochs(time, feature_hbr, onsets{2}, t_epoch, idx_base);
Xft_hbo = extract_epochs(time, feature_hbo, onsets{1}, t_epoch, idx_base);
Xft_hbr = extract_epochs(time, feature_hbr, onsets{1}, t_epoch, idx_base);
Xcomb_hbo = extract_epochs(time, feature_hbo, onsets{3}, t_epoch, idx_base);
Xcomb_hbr = extract_epochs(time, feature_hbr, onsets{3}, t_epoch, idx_base);

%% 6. BH-only cross-validation and one-standard-error model selection
n_bh = numel(onsets{2});
cv_hbo = repmat(empty_cv_result(n_bh), num_models, 1);
cv_hbr = repmat(empty_cv_result(n_bh), num_models, 1);
final_fit_hbo = cell(num_models, 1);
final_fit_hbr = cell(num_models, 1);

for m = 1:num_models
    feat = model_features{m};
    fprintf('BH validation: %s (%d predictors) ...\n', model_names(m), numel(feat));
    cv_hbo(m) = nested_block_cv(Xbh_hbo(:, feat, :), Ybh_hbo, idx_base, idx_eval, lambda_grid);
    cv_hbr(m) = nested_block_cv(Xbh_hbr(:, feat, :), Ybh_hbr, idx_base, idx_eval, lambda_grid);
    final_fit_hbo{m} = fit_final_with_lobo_lambda( ...
        Xbh_hbo(:, feat, :), Ybh_hbo, idx_base, idx_eval, lambda_grid);
    final_fit_hbr{m} = fit_final_with_lobo_lambda( ...
        Xbh_hbr(:, feat, :), Ybh_hbr, idx_base, idx_eval, lambda_grid);
end

% nRMSE error (lower is better)
cv_fold_combined = zeros(n_bh, num_models);
for m = 1:num_models
    cv_fold_combined(:, m) = (cv_hbo(m).fold_nrmse + cv_hbr(m).fold_nrmse) / 2;
end
cv_mean_combined = mean(cv_fold_combined, 1, 'omitnan');
cv_se_combined = std(cv_fold_combined, 0, 1, 'omitnan') / sqrt(n_bh);
[~, minimum_cv_model] = min(cv_mean_combined);
one_se_limit = cv_mean_combined(minimum_cv_model) + cv_se_combined(minimum_cv_model);
selected_model = find(cv_mean_combined <= one_se_limit, 1, 'first');

fprintf('Minimum BH-CV error: %s.\n', model_names(minimum_cv_model));
fprintf('One-standard-error selection: %s.\n', model_names(selected_model));

%% 7. Freeze mappings, transfer to COMB/FT, and compute trial amplitudes
comb_trials_hbo = cell(num_models, 1);
comb_trials_hbr = cell(num_models, 1);
comb_stats_hbo = cell(num_models, 1);
comb_stats_hbr = cell(num_models, 1);
comb_mask_hbo = false(num_models, num_long);
comb_mask_hbr = false(num_models, num_long);
comb_mask_joint = false(num_models, num_long);
mean_comb_hbo = cell(num_models, 1);
mean_comb_hbr = cell(num_models, 1);
mean_ft_corrected_hbo = cell(num_models, 1);
mean_ft_corrected_hbr = cell(num_models, 1);

map_r_hbo = nan(num_models, 1);
map_r_hbr = nan(num_models, 1);
map_rmse_hbo = nan(num_models, 1);
map_rmse_hbr = nan(num_models, 1);
bootstrap_r_hbo = nan(num_models, 2);
bootstrap_r_hbr = nan(num_models, 2);
bootstrap_rmse_hbo = nan(num_models, 2);
bootstrap_rmse_hbr = nan(num_models, 2);

active_hbo = zeros(num_models, 1);
active_hbr = zeros(num_models, 1);
active_joint = zeros(num_models, 1);
f1_hbo = nan(num_models, 1);
f1_hbr = nan(num_models, 1);
f1_joint = nan(num_models, 1);
nonreference_rate_hbo = nan(num_models, 1);
nonreference_rate_hbr = nan(num_models, 1);
nonreference_rate_joint = nan(num_models, 1);
ft_correction_c3_hbo = nan(num_models, 1);
ft_correction_c3_hbr = nan(num_models, 1);
comb_to_ft_hrf_c3_hbo = nan(num_models, 1);
comb_to_ft_hrf_c3_hbr = nan(num_models, 1);

rng(random_seed, 'twister');
boot_ft = randi(numel(onsets{1}), numel(onsets{1}), num_bootstrap);
boot_comb = randi(numel(onsets{3}), numel(onsets{3}), num_bootstrap);

ft_map_hbo = ft_stats_hbo.mean;
ft_map_hbr = ft_stats_hbr.mean;
mean_raw_ft_hbo = mean(Yft_hbo, 3, 'omitnan');
mean_raw_ft_hbr = mean(Yft_hbr, 3, 'omitnan');

for m = 1:num_models
    feat = model_features{m};
    pred_comb_hbo = predict_epochs(final_fit_hbo{m}, Xcomb_hbo(:, feat, :), idx_base);
    pred_comb_hbr = predict_epochs(final_fit_hbr{m}, Xcomb_hbr(:, feat, :), idx_base);
    pred_ft_hbo = predict_epochs(final_fit_hbo{m}, Xft_hbo(:, feat, :), idx_base);
    pred_ft_hbr = predict_epochs(final_fit_hbr{m}, Xft_hbr(:, feat, :), idx_base);
    
    clean_comb_hbo = Ycomb_hbo - pred_comb_hbo;
    clean_comb_hbr = Ycomb_hbr - pred_comb_hbr;
    clean_ft_hbo = Yft_hbo - pred_ft_hbo;
    clean_ft_hbr = Yft_hbr - pred_ft_hbr;

    mean_comb_hbo{m} = mean(clean_comb_hbo, 3, 'omitnan');
    mean_comb_hbr{m} = mean(clean_comb_hbr, 3, 'omitnan');
    mean_ft_corrected_hbo{m} = mean(clean_ft_hbo, 3, 'omitnan');
    mean_ft_corrected_hbr{m} = mean(clean_ft_hbr, 3, 'omitnan');

    comb_trials_hbo{m} = compute_trial_amplitudes(clean_comb_hbo, idx_task, idx_base);
    comb_trials_hbr{m} = compute_trial_amplitudes(clean_comb_hbr, idx_task, idx_base);
    comb_stats_hbo{m} = activation_statistics(comb_trials_hbo{m}, 'right');
    comb_stats_hbr{m} = activation_statistics(comb_trials_hbr{m}, 'left');

    comb_mask_hbo(m, :) = comb_stats_hbo{m}.q < alpha & comb_stats_hbo{m}.mean > 0;
    comb_mask_hbr(m, :) = comb_stats_hbr{m}.q < alpha & comb_stats_hbr{m}.mean < 0;
    comb_mask_joint(m, :) = comb_mask_hbo(m, :) & comb_mask_hbr(m, :);

    hbo_binary = binary_agreement(comb_mask_hbo(m, :), ft_mask_hbo);
    hbr_binary = binary_agreement(comb_mask_hbr(m, :), ft_mask_hbr);
    joint_binary = binary_agreement(comb_mask_joint(m, :), ft_mask_joint);
    active_hbo(m) = sum(comb_mask_hbo(m, :));
    active_hbr(m) = sum(comb_mask_hbr(m, :));
    active_joint(m) = sum(comb_mask_joint(m, :));
    
    f1_hbo(m) = hbo_binary.f1;
    f1_hbr(m) = hbr_binary.f1;
    f1_joint(m) = joint_binary.f1;
    
    nonreference_rate_hbo(m) = hbo_binary.nonreference_rate;
    nonreference_rate_hbr(m) = hbr_binary.nonreference_rate;
    nonreference_rate_joint(m) = joint_binary.nonreference_rate;

    comb_map_hbo = comb_stats_hbo{m}.mean;
    comb_map_hbr = comb_stats_hbr{m}.mean;
    
    map_r_hbo(m) = map_correlation(comb_map_hbo, ft_map_hbo);
    map_r_hbr(m) = map_correlation(comb_map_hbr, ft_map_hbr);
    map_rmse_hbo(m) = vector_rmse(comb_map_hbo, ft_map_hbo);
    map_rmse_hbr(m) = vector_rmse(comb_map_hbr, ft_map_hbr);

    [boot_r, boot_rmse] = bootstrap_map_metrics( ...
        ft_reference_trials_hbo, comb_trials_hbo{m}, boot_ft, boot_comb);
    bootstrap_r_hbo(m, :) = prctile(boot_r, [2.5 97.5]);
    bootstrap_rmse_hbo(m, :) = prctile(boot_rmse, [2.5 97.5]);
    [boot_r, boot_rmse] = bootstrap_map_metrics( ...
        ft_reference_trials_hbr, comb_trials_hbr{m}, boot_ft, boot_comb);
    bootstrap_r_hbr(m, :) = prctile(boot_r, [2.5 97.5]);
    bootstrap_rmse_hbr(m, :) = prctile(boot_rmse, [2.5 97.5]);

    ft_correction_c3_hbo(m) = matrix_rmse( ...
        mean_ft_corrected_hbo{m}(idx_eval, c3_idx), mean_raw_ft_hbo(idx_eval, c3_idx));
    ft_correction_c3_hbr(m) = matrix_rmse( ...
        mean_ft_corrected_hbr{m}(idx_eval, c3_idx), mean_raw_ft_hbr(idx_eval, c3_idx));

    comb_interp_hbo = interp1(t_epoch, mean_comb_hbo{m}, t_hrf, 'linear', NaN);
    comb_interp_hbr = interp1(t_epoch, mean_comb_hbr{m}, t_hrf, 'linear', NaN);
    hrf_eval = t_hrf >= evaluation_range(1) & t_hrf <= evaluation_range(2);
    
    comb_to_ft_hrf_c3_hbo(m) = matrix_rmse( ...
        comb_interp_hbo(hrf_eval, c3_idx), ft_reference_hrf_hbo(hrf_eval, c3_idx));
    comb_to_ft_hrf_c3_hbr(m) = matrix_rmse( ...
        comb_interp_hbr(hrf_eval, c3_idx), ft_reference_hrf_hbr(hrf_eval, c3_idx));
end

%% Numerical output tables
cv_nrmse_hbo = reshape(arrayfun(@(x) x.mean_nrmse, cv_hbo), [], 1);
cv_nrmse_hbr = reshape(arrayfun(@(x) x.mean_nrmse, cv_hbr), [], 1);
cv_r_hbo = reshape(arrayfun(@(x) x.mean_r, cv_hbo), [], 1);
cv_r_hbr = reshape(arrayfun(@(x) x.mean_r, cv_hbr), [], 1);
lambda_hbo = cellfun(@(x) x.lambda, final_fit_hbo);
lambda_hbr = cellfun(@(x) x.lambda, final_fit_hbr);
is_minimum_cv = (1:num_models)' == minimum_cv_model;
is_selected_one_se = (1:num_models)' == selected_model;

model_table = table((0:num_models-1)', cellstr(model_names), num_features, ...
    cv_nrmse_hbo, cv_nrmse_hbr, cv_mean_combined', cv_se_combined', ...
    cv_r_hbo, cv_r_hbr, lambda_hbo, lambda_hbr, ...
    map_r_hbo, bootstrap_r_hbo(:, 1), bootstrap_r_hbo(:, 2), ...
    map_r_hbr, bootstrap_r_hbr(:, 1), bootstrap_r_hbr(:, 2), ...
    map_rmse_hbo, bootstrap_rmse_hbo(:, 1), bootstrap_rmse_hbo(:, 2), ...
    map_rmse_hbr, bootstrap_rmse_hbr(:, 1), bootstrap_rmse_hbr(:, 2), ...
    active_hbo, active_hbr, active_joint, f1_hbo, f1_hbr, f1_joint, ...
    nonreference_rate_hbo, nonreference_rate_hbr, nonreference_rate_joint, ...
    ft_correction_c3_hbo, ft_correction_c3_hbr, ...
    comb_to_ft_hrf_c3_hbo, comb_to_ft_hrf_c3_hbr, ...
    is_minimum_cv, is_selected_one_se, ...
    'VariableNames', { ...
    'ModelID','Model','NumPredictors', ...
    'BH_CV_nRMSE_HbO','BH_CV_nRMSE_HbR','BH_CV_nRMSE_Combined','BH_CV_SE_Combined', ...
    'BH_CV_R_HbO','BH_CV_R_HbR','Lambda_HbO','Lambda_HbR', ...
    'SpatialMap_R_HbO','SpatialMap_R_HbO_CI_Low','SpatialMap_R_HbO_CI_High', ...
    'SpatialMap_R_HbR','SpatialMap_R_HbR_CI_Low','SpatialMap_R_HbR_CI_High', ...
    'SpatialMap_RMSE_HbO_uM','SpatialMap_RMSE_HbO_CI_Low','SpatialMap_RMSE_HbO_CI_High', ...
    'SpatialMap_RMSE_HbR_uM','SpatialMap_RMSE_HbR_CI_Low','SpatialMap_RMSE_HbR_CI_High', ...
    'Active_HbO_FDR','Active_HbR_FDR','Active_Joint_FDR', ...
    'F1_HbO_FDR','F1_HbR_FDR','F1_Joint_FDR', ...
    'NonReferenceRate_HbO','NonReferenceRate_HbR','NonReferenceRate_Joint', ...
    'FT_CorrectionMagnitude_C3_HbO_uM','FT_CorrectionMagnitude_C3_HbR_uM', ...
    'COMB_to_HomerFT_HRF_RMSE_C3_HbO_uM','COMB_to_HomerFT_HRF_RMSE_C3_HbR_uM', ...
    'MinimumBHCVError','SelectedByOneSE'});

activation_table = build_activation_table(subject_id, long_channels, ml_src, ml_det, ...
    c3_channels, model_names, ft_stats_hbo, ft_stats_hbr, ft_mask_hbo, ft_mask_hbr, ...
    comb_stats_hbo, comb_stats_hbr, comb_mask_hbo, comb_mask_hbr);

model_csv = fullfile(output_dir, [subject_id '_IntegratedSpatial_ModelSummary.csv']);
activation_csv = fullfile(output_dir, [subject_id '_IntegratedSpatial_ActivationAudit.csv']);
writetable(model_table, model_csv);
writetable(activation_table, activation_csv);

%% Neutral probe-layout figures (no unverified L/R orientation labels)
midpoints = (src_pos(ml_src(long_channels), 1:2) + det_pos(ml_det(long_channels), 1:2)) / 2;
layout_center = mean([min(midpoints, [], 1); max(midpoints, [], 1)], 1);
layout_xy = midpoints - layout_center;
if size(det_pos, 1) >= 4
    d4_xy = det_pos(4, 1:2) - layout_center;
else
    d4_xy = [];
end
if size(det_pos, 1) >= 12
    d12_xy = det_pos(12, 1:2) - layout_center;
else
    d12_xy = [];
end

comparison_model = selected_model;
comparison_role = "one-SE selected";
if comparison_model == 2
    if minimum_cv_model ~= 2
        comparison_model = minimum_cv_model;
        comparison_role = "minimum BH-CV error";
    else
        comparison_model = min(5, num_models);
        comparison_role = "prespecified multimodal comparator";
    end
end

figure_hbo = fullfile(output_dir, [subject_id '_IntegratedSpatial_HbO_ProbeMaps.png']);
figure_hbr = fullfile(output_dir, [subject_id '_IntegratedSpatial_HbR_ProbeMaps.png']);
plot_four_probe_maps(layout_xy, d4_xy, d12_xy, long_channels, ...
    ft_stats_hbo, comb_stats_hbo, ft_mask_hbo, comb_mask_hbo, ...
    model_names, comparison_model, comparison_role, 'HbO', figure_hbo, show_figures);
plot_four_probe_maps(layout_xy, d4_xy, d12_xy, long_channels, ...
    ft_stats_hbr, comb_stats_hbr, ft_mask_hbr, comb_mask_hbr, ...
    model_names, comparison_model, comparison_role, 'HbR', figure_hbr, show_figures);

summary_figure = fullfile(output_dir, [subject_id '_IntegratedSpatial_MetricSummary.png']);
plot_metric_summary(model_names, cv_mean_combined, cv_se_combined, ...
    selected_model, minimum_cv_model, map_r_hbo, map_r_hbr, ...
    bootstrap_r_hbo, bootstrap_r_hbr, f1_hbo, f1_hbr, f1_joint, ...
    nonreference_rate_hbo, nonreference_rate_hbr, summary_figure, subject_id, show_figures);

figure_temporal = fullfile(output_dir, [subject_id '_IntegratedSpatial_TemporalHRF_C3.png']);
plot_temporal_hrf_c3(t_epoch, t_hrf, ft_reference_hrf_hbo, ft_reference_hrf_hbr, ...
    mean_comb_hbo, mean_comb_hbr, c3_idx, long_channels, ml_src, ml_det, ...
    task_range, figure_temporal, subject_id, show_figures);

%% Reproducibility object and plain-language summary
results = struct();
results.subject_id = subject_id;
results.fs = fs;
results.long_channels = long_channels;
results.short_channels = short_channels;
results.provisional_c3_channels = c3_channels;
results.model_names = model_names;
results.model_features = model_features;
results.feature_names_hbo = feature_names_hbo;
results.feature_names_hbr = feature_names_hbr;
results.lags_sec = lags_sec;
results.lambda_grid = lambda_grid;
results.homer_time = t_hrf;
results.homer_trial_count = n_homer_trials;
results.ft_reference_hrf_hbo = ft_reference_hrf_hbo;
results.ft_reference_hrf_hbr = ft_reference_hrf_hbr;
results.ft_reference_trials_hbo = ft_reference_trials_hbo;
results.ft_reference_trials_hbr = ft_reference_trials_hbr;
results.ft_stats_hbo = ft_stats_hbo;
results.ft_stats_hbr = ft_stats_hbr;
results.cv_hbo = cv_hbo;
results.cv_hbr = cv_hbr;
results.final_fit_hbo = final_fit_hbo;
results.final_fit_hbr = final_fit_hbr;
results.selected_model_index = selected_model;
results.minimum_cv_model_index = minimum_cv_model;
results.one_se_limit = one_se_limit;
results.model_table = model_table;
results.activation_table = activation_table;
results.comb_trials_hbo = comb_trials_hbo;
results.comb_trials_hbr = comb_trials_hbr;
results.mean_comb_hbo = mean_comb_hbo;
results.mean_comb_hbr = mean_comb_hbr;

mat_out = fullfile(output_dir, [subject_id '_IntegratedSpatial_Results.mat']);
save(mat_out, 'results', '-v7.3');

readme_out = fullfile(output_dir, [subject_id '_IntegratedSpatial_Readme.txt']);
fid = fopen(readme_out, 'w');
assert(fid >= 0, 'Could not create readme file.');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'Integrated spatial agreement and neurofunctional validation\n');
fprintf(fid, 'Subject: %s\n\n', subject_id);
fprintf(fid, 'Empirical FT reference: simultaneous FT+BH+COMB Homer GLM with SC PCA.\n');
fprintf(fid, 'FT active after FDR: HbO %d, HbR %d, joint %d of %d long channels.\n', ...
    sum(ft_mask_hbo), sum(ft_mask_hbr), sum(ft_mask_joint), num_long);
fprintf(fid, 'Minimum BH-CV-error model: %s.\n', model_names(minimum_cv_model));
fprintf(fid, 'One-standard-error selected model: %s.\n', model_names(selected_model));
fprintf(fid, 'The one-SE rule is heuristic with only %d BH blocks.\n\n', n_bh);
fprintf(fid, 'Terminology: non-reference activation is not asserted to be a known biological false positive.\n');
fprintf(fid, 'The 12-channel C3 definition is geometric and must be verified against the montage.\n');
fprintf(fid, 'HbO and HbR FDR maps are primary; their conjunction is a sensitivity analysis.\n');
fprintf(fid, 'Bootstrap intervals quantify trial-sampling instability but remain limited by the small trial count.\n');
clear cleanup;

disp(model_table);
fprintf('\nSaved integrated analysis:\n  %s\n  %s\n  %s\n  %s\n  %s\n  %s\n', ...
    model_csv, activation_csv, figure_hbo, figure_hbr, summary_figure, mat_out);
end

%% Local functions
function amplitudes = trial_amplitudes_continuous(time, data, onsets, baseline_window, task_window)
amplitudes = nan(numel(onsets), size(data, 2));
for k = 1:numel(onsets)
    idx_base = time >= onsets(k) + baseline_window(1) & time < onsets(k) + baseline_window(2);
    idx_task = time >= onsets(k) + task_window(1) & time <= onsets(k) + task_window(2);
    amplitudes(k, :) = mean(data(idx_task, :), 1, 'omitnan') - ...
        mean(data(idx_base, :), 1, 'omitnan');
end
end

function amplitudes = compute_trial_amplitudes(epochs, idx_task, idx_base)
n_trials = size(epochs, 3);
n_channels = size(epochs, 2);
amplitudes = nan(n_trials, n_channels);
for k = 1:n_trials
    amplitudes(k, :) = mean(epochs(idx_task, :, k), 1, 'omitnan') - ...
        mean(epochs(idx_base, :, k), 1, 'omitnan');
end
end

function stats = activation_statistics(amplitudes, tail)
n_channels = size(amplitudes, 2);
stats.mean = mean(amplitudes, 1, 'omitnan');
stats.p = nan(1, n_channels);
stats.t = nan(1, n_channels);
stats.df = nan(1, n_channels);
for ch = 1:n_channels
    values = amplitudes(:, ch);
    values = values(isfinite(values));
    if numel(values) >= 2 && std(values) > 0
        [~, stats.p(ch), ~, test_stats] = ttest(values, 0, 'Tail', tail);
        stats.t(ch) = test_stats.tstat;
        stats.df(ch) = test_stats.df;
    end
end
stats.q = bh_adjust(stats.p);
end

function q = bh_adjust(p)
original_size = size(p);
p = p(:);
q = nan(size(p));
valid = isfinite(p);
pv = p(valid);
[sorted_p, order] = sort(pv);
n = numel(sorted_p);
if n > 0
    adjusted = sorted_p .* n ./ (1:n)';
    adjusted = flipud(cummin(flipud(adjusted)));
    adjusted = min(adjusted, 1);
    unsorted = nan(size(pv));
    unsorted(order) = adjusted;
    q(valid) = unsorted;
end
q = reshape(q, original_size);
end

function agreement = binary_agreement(predicted, reference)
predicted = logical(predicted(:));
reference = logical(reference(:));
agreement.tp = sum(predicted & reference);
agreement.fp = sum(predicted & ~reference);
agreement.fn = sum(~predicted & reference);
agreement.tn = sum(~predicted & ~reference);
agreement.precision = agreement.tp / max(1, agreement.tp + agreement.fp);
agreement.sensitivity = agreement.tp / max(1, agreement.tp + agreement.fn);
agreement.nonreference_rate = agreement.fp / max(1, agreement.fp + agreement.tn);
denominator = 2 * agreement.tp + agreement.fp + agreement.fn;
if denominator == 0
    agreement.f1 = NaN;
else
    agreement.f1 = 2 * agreement.tp / denominator;
end
end

function r = map_correlation(a, b)
a = a(:);
b = b(:);
valid = isfinite(a) & isfinite(b);
if sum(valid) < 3 || std(a(valid)) < 1e-12 || std(b(valid)) < 1e-12
    r = NaN;
else
    r = corr(a(valid), b(valid));
end
end

function value = vector_rmse(a, b)
d = a(:) - b(:);
value = sqrt(mean(d.^2, 'omitnan'));
end

function [boot_r, boot_rmse] = bootstrap_map_metrics(reference_trials, test_trials, ref_indices, test_indices)
n_boot = size(ref_indices, 2);
boot_r = nan(n_boot, 1);
boot_rmse = nan(n_boot, 1);
for b = 1:n_boot
    ref_map = mean(reference_trials(ref_indices(:, b), :), 1, 'omitnan');
    test_map = mean(test_trials(test_indices(:, b), :), 1, 'omitnan');
    boot_r(b) = map_correlation(test_map, ref_map);
    boot_rmse(b) = vector_rmse(test_map, ref_map);
end
end

function value = matrix_rmse(a, b)
d = a - b;
value = sqrt(mean(d.^2, 'all', 'omitnan'));
end

function [bank, names_out, groups] = make_lag_bank(time, signals, names_in, lags)
n_signals = size(signals, 2);
n_lags = numel(lags);
bank = zeros(numel(time), n_signals * n_lags);
names_out = strings(1, n_signals * n_lags);
groups = cell(1, n_signals);
column = 0;
for s = 1:n_signals
    first_column = column + 1;
    for k = 1:n_lags
        column = column + 1;
        bank(:, column) = interp1(time, signals(:, s), time - lags(k), 'linear', 'extrap');
        names_out(column) = sprintf('%s lag %gs', names_in(s), lags(k));
    end
    groups{s} = first_column:column;
end
end

function epochs = extract_epochs(time, data, onsets, t_epoch, idx_base)
epochs = nan(numel(t_epoch), size(data, 2), numel(onsets));
for b = 1:numel(onsets)
    epochs(:, :, b) = interp1(time, data, onsets(b) + t_epoch, 'linear', NaN);
    baseline = mean(epochs(idx_base, :, b), 1, 'omitnan');
    epochs(:, :, b) = epochs(:, :, b) - baseline;
end
end

function out = empty_cv_result(n_blocks)
out = struct('fold_nrmse', nan(n_blocks, 1), 'fold_rmse', nan(n_blocks, 1), ...
    'fold_r', nan(n_blocks, 1), 'outer_lambda', nan(n_blocks, 1), ...
    'mean_nrmse', NaN, 'mean_rmse', NaN, 'mean_r', NaN);
end

function out = nested_block_cv(X, Y, idx_base, idx_eval, lambda_grid)
n_blocks = size(Y, 3);
out = empty_cv_result(n_blocks);
all_blocks = 1:n_blocks;
no_features = size(X, 2) == 0;
for outer = 1:n_blocks
    train_blocks = setdiff(all_blocks, outer);
    if no_features
        prediction = zeros(size(Y(:, :, outer)));
        chosen_lambda = NaN;
    else
        inner_error = nan(numel(lambda_grid), numel(train_blocks));
        for li = 1:numel(lambda_grid)
            for iv = 1:numel(train_blocks)
                validation_block = train_blocks(iv);
                inner_train = setdiff(train_blocks, validation_block);
                fit = fit_epoch_blocks(X, Y, inner_train, lambda_grid(li), idx_eval);
                prediction_inner = apply_ridge(fit, X(:, :, validation_block));
                prediction_inner = baseline_prediction(prediction_inner, idx_base);
                inner_error(li, iv) = normalized_error( ...
                    prediction_inner(idx_eval, :), Y(idx_eval, :, validation_block));
            end
        end
        [~, best_lambda_index] = min(mean(inner_error, 2, 'omitnan'));
        chosen_lambda = lambda_grid(best_lambda_index);
        fit = fit_epoch_blocks(X, Y, train_blocks, chosen_lambda, idx_eval);
        prediction = apply_ridge(fit, X(:, :, outer));
        prediction = baseline_prediction(prediction, idx_base);
    end
    truth = Y(:, :, outer);
    out.fold_nrmse(outer) = normalized_error(prediction(idx_eval, :), truth(idx_eval, :));
    out.fold_rmse(outer) = matrix_rmse(prediction(idx_eval, :), truth(idx_eval, :));
    out.fold_r(outer) = mean_channel_correlation(prediction(idx_eval, :), truth(idx_eval, :));
    out.outer_lambda(outer) = chosen_lambda;
end
out.mean_nrmse = mean(out.fold_nrmse, 'omitnan');
out.mean_rmse = mean(out.fold_rmse, 'omitnan');
out.mean_r = mean(out.fold_r, 'omitnan');
end

function final_fit = fit_final_with_lobo_lambda(X, Y, idx_base, idx_eval, lambda_grid)
n_blocks = size(Y, 3);
if size(X, 2) == 0
    final_fit = struct('is_null', true, 'lambda', NaN, 'num_targets', size(Y, 2));
    return;
end
all_blocks = 1:n_blocks;
cv_error = nan(numel(lambda_grid), n_blocks);
for li = 1:numel(lambda_grid)
    for b = 1:n_blocks
        fit = fit_epoch_blocks(X, Y, setdiff(all_blocks, b), lambda_grid(li), idx_eval);
        prediction = apply_ridge(fit, X(:, :, b));
        prediction = baseline_prediction(prediction, idx_base);
        cv_error(li, b) = normalized_error(prediction(idx_eval, :), Y(idx_eval, :, b));
    end
end
[~, best_lambda_index] = min(mean(cv_error, 2, 'omitnan'));
final_fit = fit_epoch_blocks(X, Y, all_blocks, lambda_grid(best_lambda_index), idx_eval);
final_fit.lambda = lambda_grid(best_lambda_index);
final_fit.is_null = false;
end

function fit = fit_epoch_blocks(X, Y, block_ids, lambda, idx_fit)
n_rows = sum(idx_fit) * numel(block_ids);
Xtrain = nan(n_rows, size(X, 2));
Ytrain = nan(n_rows, size(Y, 2));
cursor = 0;
for b = block_ids
    rows = cursor + (1:sum(idx_fit));
    Xtrain(rows, :) = X(idx_fit, :, b);
    Ytrain(rows, :) = Y(idx_fit, :, b);
    cursor = cursor + sum(idx_fit);
end
valid_rows = all(isfinite(Xtrain), 2) & all(isfinite(Ytrain), 2);
Xtrain = Xtrain(valid_rows, :);
Ytrain = Ytrain(valid_rows, :);
fit.mu_x = mean(Xtrain, 1);
fit.sd_x = std(Xtrain, 0, 1);
fit.sd_x(fit.sd_x < 1e-10 | ~isfinite(fit.sd_x)) = 1;
Xz = (Xtrain - fit.mu_x) ./ fit.sd_x;
fit.mu_y = mean(Ytrain, 1);
Yc = Ytrain - fit.mu_y;
fit.beta = (Xz' * Xz + lambda * eye(size(Xz, 2))) \ (Xz' * Yc);
fit.lambda = lambda;
fit.is_null = false;
end

function prediction = apply_ridge(fit, X)
Xz = (X - fit.mu_x) ./ fit.sd_x;
prediction = Xz * fit.beta + fit.mu_y;
end

function prediction = baseline_prediction(prediction, idx_base)
prediction = prediction - mean(prediction(idx_base, :), 1, 'omitnan');
end

function predictions = predict_epochs(fit, Xepochs, idx_base)
n_epoch = size(Xepochs, 1);
n_blocks = size(Xepochs, 3);
if isfield(fit, 'is_null') && fit.is_null
    predictions = zeros(n_epoch, fit.num_targets, n_blocks);
    return;
end
predictions = zeros(n_epoch, size(fit.beta, 2), n_blocks);
for b = 1:n_blocks
    prediction = apply_ridge(fit, Xepochs(:, :, b));
    predictions(:, :, b) = baseline_prediction(prediction, idx_base);
end
end

function value = normalized_error(prediction, truth)
error_energy = sum((prediction - truth).^2, 1, 'omitnan');
truth_energy = sum(truth.^2, 1, 'omitnan');
value = mean(sqrt(error_energy ./ max(truth_energy, eps)), 'omitnan');
end

function value = mean_channel_correlation(a, b)
z = nan(1, size(a, 2));
for ch = 1:size(a, 2)
    r = map_correlation(a(:, ch), b(:, ch));
    if isfinite(r)
        z(ch) = atanh(min(max(r, -0.999999), 0.999999));
    end
end
if all(isnan(z))
    value = 0;
else
    value = tanh(mean(z, 'omitnan'));
end
end

function table_out = build_activation_table(subject_id, long_channels, ml_src, ml_det, ...
    c3_channels, model_names, ft_hbo, ft_hbr, ft_mask_hbo, ft_mask_hbr, ...
    comb_hbo, comb_hbr, comb_mask_hbo, comb_mask_hbr)
n_channels = numel(long_channels);
n_models = numel(model_names);
n_rows = (n_models + 1) * n_channels;
Subject = repmat(string(subject_id), n_rows, 1);
Condition = strings(n_rows, 1);
ModelID = nan(n_rows, 1);
Model = strings(n_rows, 1);
Channel = zeros(n_rows, 1);
Source = zeros(n_rows, 1);
Detector = zeros(n_rows, 1);
InProvisionalC3 = false(n_rows, 1);
MeanAmplitude_HbO_uM = nan(n_rows, 1);
P_HbO = nan(n_rows, 1);
Q_HbO = nan(n_rows, 1);
T_HbO = nan(n_rows, 1);
Active_HbO_FDR = false(n_rows, 1);
MeanAmplitude_HbR_uM = nan(n_rows, 1);
P_HbR = nan(n_rows, 1);
Q_HbR = nan(n_rows, 1);
T_HbR = nan(n_rows, 1);
Active_HbR_FDR = false(n_rows, 1);
Active_Joint_FDR = false(n_rows, 1);

for set_index = 0:n_models
    rows = set_index * n_channels + (1:n_channels);
    Channel(rows) = long_channels(:);
    Source(rows) = ml_src(long_channels);
    Detector(rows) = ml_det(long_channels);
    InProvisionalC3(rows) = ismember(long_channels, c3_channels);
    if set_index == 0
        Condition(rows) = "FT reference";
        ModelID(rows) = -1;
        Model(rows) = "Simultaneous Homer GLM + SC PCA";
        stats_hbo = ft_hbo;
        stats_hbr = ft_hbr;
        mask_hbo = ft_mask_hbo;
        mask_hbr = ft_mask_hbr;
    else
        Condition(rows) = "COMB";
        ModelID(rows) = set_index - 1;
        Model(rows) = model_names(set_index);
        stats_hbo = comb_hbo{set_index};
        stats_hbr = comb_hbr{set_index};
        mask_hbo = comb_mask_hbo(set_index, :);
        mask_hbr = comb_mask_hbr(set_index, :);
    end
    MeanAmplitude_HbO_uM(rows) = stats_hbo.mean(:);
    P_HbO(rows) = stats_hbo.p(:);
    Q_HbO(rows) = stats_hbo.q(:);
    T_HbO(rows) = stats_hbo.t(:);
    Active_HbO_FDR(rows) = mask_hbo(:);
    MeanAmplitude_HbR_uM(rows) = stats_hbr.mean(:);
    P_HbR(rows) = stats_hbr.p(:);
    Q_HbR(rows) = stats_hbr.q(:);
    T_HbR(rows) = stats_hbr.t(:);
    Active_HbR_FDR(rows) = mask_hbr(:);
    Active_Joint_FDR(rows) = mask_hbo(:) & mask_hbr(:);
end

table_out = table(Subject, Condition, ModelID, Model, Channel, Source, Detector, ...
    InProvisionalC3, MeanAmplitude_HbO_uM, P_HbO, Q_HbO, T_HbO, Active_HbO_FDR, ...
    MeanAmplitude_HbR_uM, P_HbR, Q_HbR, T_HbR, Active_HbR_FDR, Active_Joint_FDR);
end

function plot_four_probe_maps(xy, d4_xy, d12_xy, channels, ft_stats, comb_stats, ...
    ft_mask, comb_masks, model_names, comparison_model, comparison_role, chromophore, output_file, show_figures)
if nargin < 14, show_figures = true; end
if strcmpi(chromophore, 'HbO') || strcmpi(chromophore, 'O_2Hb') || strcmpi(chromophore, 'O2Hb')
    sign_factor = 1;
    chromo_disp = 'O_2Hb';
else
    sign_factor = -1;
    chromo_disp = 'HHb';
end
strength = cell(4, 1);
masks = cell(4, 1);
titles = strings(4, 1);
strength{1} = sign_factor * ft_stats.t;
strength{2} = sign_factor * comb_stats{1}.t;
strength{3} = sign_factor * comb_stats{2}.t;
strength{4} = sign_factor * comb_stats{comparison_model}.t;
masks{1} = ft_mask;
masks{2} = comb_masks(1, :);
masks{3} = comb_masks(2, :);
masks{4} = comb_masks(comparison_model, :);
titles(1) = sprintf('A. Empirical FT reference (active %d/%d)', sum(masks{1}), numel(channels));
titles(2) = sprintf('B. COMB M0 uncorrected (active %d/%d)', sum(masks{2}), numel(channels));
titles(3) = sprintf('C. COMB M1 SC only (active %d/%d)', sum(masks{3}), numel(channels));
titles(4) = sprintf('D. COMB %s: %s (active %d/%d)', ...
    extractBefore(model_names(comparison_model), ' '), comparison_role, sum(masks{4}), numel(channels));
all_strength = [strength{1}(:); strength{2}(:); strength{3}(:); strength{4}(:)];
all_strength = all_strength(isfinite(all_strength));
color_limit = max(2.5, max([2.5; all_strength]));

if show_figures
    fig_vis = 'on';
else
    fig_vis = 'off';
end
fig = figure('Visible', fig_vis, 'Color', 'w', 'Position', [50 50 1400 1000]);
layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:4
    ax = nexttile;
    plot_probe_layout(ax, xy, d4_xy, d12_xy, max(0, strength{p}), masks{p}, ...
        titles(p), [0 color_limit]);
end
colormap(fig, parula);
cb = colorbar;
cb.Layout.Tile = 'east';
ylabel(cb, sprintf('Directional %s t-statistic', chromo_disp));
title(layout, sprintf('%s spatial agreement: FDR q < 0.05', chromo_disp), ...
    'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');
subtitle(layout, 'Black rings indicate chromophore-specific activation; probe orientation must be verified against the montage', ...
    'Color', 'k');
style_figure_for_export(fig);
exportgraphics(fig, output_file, 'Resolution', 240);
if ~show_figures
    close(fig);
else
    drawnow;
end
end

function plot_probe_layout(ax, xy, d4_xy, d12_xy, values, active_mask, title_text, limits)
scatter(ax, xy(:, 1), xy(:, 2), 125, values, 'filled', ...
    'MarkerEdgeColor', [0.35 0.35 0.35], 'LineWidth', 0.7);
hold(ax, 'on');
active = find(active_mask);
if ~isempty(active)
    scatter(ax, xy(active, 1), xy(active, 2), 210, 'o', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 2.0);
end
if ~isempty(d4_xy)
    plot(ax, d4_xy(1), d4_xy(2), 'kx', 'MarkerSize', 9, 'LineWidth', 1.8);
    text(ax, d4_xy(1), d4_xy(2), ' D4/C3 marker', 'FontSize', 9, 'VerticalAlignment', 'bottom');
end
if ~isempty(d12_xy)
    plot(ax, d12_xy(1), d12_xy(2), 'kx', 'MarkerSize', 9, 'LineWidth', 1.8);
    text(ax, d12_xy(1), d12_xy(2), ' D12/C4 marker', 'FontSize', 9, 'VerticalAlignment', 'bottom');
end
axis(ax, 'equal');
axis(ax, 'off');
clim(ax, limits);
title(ax, title_text, 'Interpreter', 'none', 'FontSize', 11);
end

function plot_metric_summary(model_names, cv_mean, cv_se, selected_model, minimum_model, ...
    map_r_hbo, map_r_hbr, ci_hbo, ci_hbr, f1_hbo, f1_hbr, f1_joint, ...
    nonref_hbo, nonref_hbr, output_file, subject_id, show_figures)
if nargin < 17, show_figures = true; end
ids = 0:numel(model_names)-1;
if show_figures
    fig_vis = 'on';
else
    fig_vis = 'off';
end
fig = figure('Visible', fig_vis, 'Color', 'w', 'Position', [70 70 1450 930]);
layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
bar(ids, cv_mean, 'FaceColor', [0.25 0.55 0.78]); hold on;
errorbar(ids, cv_mean, cv_se, 'k.', 'LineWidth', 1.2);
xline(selected_model-1, 'k:', '1-SE selected', 'HandleVisibility', 'off');
xline(minimum_model-1, 'r--', 'Minimum', 'HandleVisibility', 'off');
yline(1, 'Color', [0.35 0.35 0.35], 'LineStyle', '--', 'HandleVisibility', 'off');
xlabel('Model ID'); ylabel('BH CV nRMSE'); title('Independent BH prediction (lower is better)'); grid on;

nexttile;
b = bar(ids, [map_r_hbo map_r_hbr], 'grouped'); hold on;
for k = 1:2
    if k == 1
        values = map_r_hbo; intervals = ci_hbo;
    else
        values = map_r_hbr; intervals = ci_hbr;
    end
    lower = values - intervals(:, 1);
    upper = intervals(:, 2) - values;
    errorbar(b(k).XEndPoints, values, lower, upper, 'k.', 'LineWidth', 1.0);
end
xlabel('Model ID'); ylabel('Spatial map correlation'); title('COMB-to-FT spatial similarity with 95% trial bootstrap CI');
legend({'O_2Hb','HHb'}, 'Location', 'best'); grid on;

nexttile;
bar(ids, [f1_hbo f1_hbr f1_joint], 'grouped');
xlabel('Model ID'); ylabel('F1'); title('FDR activation agreement (secondary)');
legend({'O_2Hb','HHb','Joint sensitivity analysis'}, 'Location', 'best'); ylim([0 1]); grid on;

nexttile;
bar(ids, 100 * [nonref_hbo nonref_hbr], 'grouped');
xlabel('Model ID'); ylabel('Non-reference activation rate (%)');
title('Reference-discordant activation (descriptive)');
legend({'O_2Hb','HHb'}, 'Location', 'best'); grid on;

title(layout, sprintf('%s: integrated spatial validation', subject_id), ...
    'Interpreter', 'none', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'k');
style_figure_for_export(fig);
exportgraphics(fig, output_file, 'Resolution', 240);
if ~show_figures
    close(fig);
else
    drawnow;
end
end

function style_figure_for_export(fig)
set(fig, 'Color', 'w');
axes_handles = findall(fig, 'Type', 'axes');
set(axes_handles, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', ...
    'GridColor', [0.72 0.72 0.72], 'MinorGridColor', [0.86 0.86 0.86]);
text_handles = findall(fig, 'Type', 'text');
set(text_handles, 'Color', 'k');
legend_handles = findall(fig, 'Type', 'legend');
set(legend_handles, 'Color', 'w', 'TextColor', 'k', ...
    'EdgeColor', [0.35 0.35 0.35]);
colorbar_handles = findall(fig, 'Type', 'colorbar');
if ~isempty(colorbar_handles)
    set(colorbar_handles, 'Color', 'k');
end
end

function plot_temporal_hrf_c3(t_epoch, t_hrf, ft_reference_hbo, ft_reference_hbr, ...
    mean_comb_hbo, mean_comb_hbr, c3_idx, long_channels, ml_src, ml_det, ...
    task_range, output_file, subject_id, show_figures)
if nargin < 14, show_figures = true; end

% Find representative C3 channel with largest positive task response in FT reference
hrf_task = t_hrf >= task_range(1) & t_hrf <= task_range(2);
c3_ft_amps = mean(ft_reference_hbo(hrf_task, c3_idx), 1, 'omitnan');
[~, best_sub] = max(c3_ft_amps);
if isempty(best_sub) || isnan(best_sub)
    best_sub = 1;
end
rep_idx = c3_idx(best_sub);
rep_ch = long_channels(rep_idx);
src_id = ml_src(rep_ch);
det_id = ml_det(rep_ch);

if show_figures
    fig_vis = 'on';
else
    fig_vis = 'off';
end

fig = figure('Visible', fig_vis, 'Color', 'w', 'Position', [60 60 1450 620]);
layout = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

colors = { ...
    [0.72 0.72 0.72], ... % M0 (grey)
    [0.25 0.75 0.90], ... % M1 (cyan)
    [0.10 0.70 0.40], ... % M2 (green)
    [0.10 0.40 0.85], ... % M3 (blue)
    [0.88 0.15 0.15], ... % M4 (red)
    [0.92 0.50 0.10], ... % M5 (orange)
    [0.75 0.15 0.75]  ... % M6 (magenta)
};
line_styles = {'-', '-', '-', '-', '-', '--', '--'};
line_widths = [1.5, 1.8, 1.8, 1.8, 2.2, 1.6, 1.6];
model_labels = { ...
    'COMB raw (M0)', ...
    'COMB M1 (SC only)', ...
    'COMB M2 (+ Resp)', ...
    'COMB M3 (+ HR)', ...
    'COMB M4 (+ SpO2)', ...
    'COMB M5 (+ Phasic EDA)', ...
    'COMB M6 (+ Tonic EDA)' ...
};

% Panel 1: HbO
ax1 = nexttile;
hold(ax1, 'on');
plot(ax1, t_hrf, ft_reference_hbo(:, rep_idx), 'k-', 'LineWidth', 2.5, 'DisplayName', 'FT reference (Homer SC-PCA)');
for m = 1:numel(mean_comb_hbo)
    plot(ax1, t_epoch, mean_comb_hbo{m}(:, rep_idx), line_styles{m}, ...
        'Color', colors{m}, 'LineWidth', line_widths(m), 'DisplayName', model_labels{m});
end
xline(ax1, 0, 'k:', 'Task onset', 'HandleVisibility', 'off');
xline(ax1, 21, 'k:', 'Task offset', 'HandleVisibility', 'off');
yline(ax1, 0, 'Color', [0.35 0.35 0.35], 'LineStyle', '--', 'HandleVisibility', 'off');
xlim(ax1, [-5, 40]); grid(ax1, 'on');
xlabel(ax1, 'Time from onset (s)'); ylabel(ax1, '\Delta[O_2Hb] (\muM)');
title(ax1, sprintf('\\Delta[O_2Hb] Hemodynamic Recovery: Ch %d (S%d-D%d, C3 ROI)', rep_ch, src_id, det_id), ...
    'FontSize', 12, 'FontWeight', 'bold');
legend(ax1, 'Location', 'best', 'NumColumns', 2);

% Panel 2: HHb
ax2 = nexttile;
hold(ax2, 'on');
plot(ax2, t_hrf, ft_reference_hbr(:, rep_idx), 'k-', 'LineWidth', 2.5, 'DisplayName', 'FT reference (Homer SC-PCA)');
for m = 1:numel(mean_comb_hbr)
    plot(ax2, t_epoch, mean_comb_hbr{m}(:, rep_idx), line_styles{m}, ...
        'Color', colors{m}, 'LineWidth', line_widths(m), 'DisplayName', model_labels{m});
end
xline(ax2, 0, 'k:', 'Task onset', 'HandleVisibility', 'off');
xline(ax2, 21, 'k:', 'Task offset', 'HandleVisibility', 'off');
yline(ax2, 0, 'Color', [0.35 0.35 0.35], 'LineStyle', '--', 'HandleVisibility', 'off');
xlim(ax2, [-5, 40]); grid(ax2, 'on');
xlabel(ax2, 'Time from onset (s)'); ylabel(ax2, '\Delta[HHb] (\muM)');
title(ax2, sprintf('\\Delta[HHb] Hemodynamic Recovery: Ch %d (S%d-D%d, C3 ROI)', rep_ch, src_id, det_id), ...
    'FontSize', 12, 'FontWeight', 'bold');
legend(ax2, 'Location', 'best', 'NumColumns', 2);

title(layout, sprintf('%s: Temporal Hemodynamic Recovery in Motor Cortex (COMB vs FT Reference)', subject_id), ...
    'Interpreter', 'none', 'FontSize', 15, 'FontWeight', 'bold', 'Color', 'k');

style_figure_for_export(fig);
exportgraphics(fig, output_file, 'Resolution', 240);
if ~show_figures
    close(fig);
else
    drawnow;
end
end
