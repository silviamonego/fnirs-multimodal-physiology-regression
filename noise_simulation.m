%% NOISE_SIMULATION
% 1/f spectral noise analysis and Monte Carlo spurious correlation simulation.
%
% Objectives:
%   1. Estimate Power Spectral Density (PSD) of fNIRS short channels and systemic physiology.
%   2. Verify the 1/f^beta power law decay (beta = 1.7) in low-frequency noise.
%   3. Demonstrate via Monte Carlo simulation (N = 1000) that block-by-block linear detrending
%      is necessary to prevent spurious correlations between independent time series.

clearvars -except subj_dir; close all; clc;

% Simulation parameters
noise_exponent = 1.7; % Exponent beta for 1/f^beta noise
num_iterations = 1000;
task_duration = 21;    % seconds

%% 1. Load fNIRS and physiological time series
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

assert(exist(file_snirf, 'file') == 2, 'Satori SNIRF not found: %s', file_snirf);
assert(exist(file_snirf_raw, 'file') == 2, 'Raw SNIRF not found: %s', file_snirf_raw);
assert(exist(file_csv, 'file') == 2, 'Physiology CSV not found: %s', file_csv);

time_vector = h5read(file_snirf, '/nirs/data1/time');
num_time_points = length(time_vector);
fs_fnirs = 1 / mean(diff(time_vector));

% Triggers
stim1 = h5read(file_snirf, '/nirs/stim1/data');
stim2 = h5read(file_snirf, '/nirs/stim2/data');
stim3 = h5read(file_snirf, '/nirs/stim3/data');
onsets_stim1 = stim1(1, :)'; % FT
onsets_stim2 = stim2(1, :)'; % BH
onsets_stim3 = stim3(1, :)'; % COMB

% Load HbO data
data_time_series = h5read(file_snirf, '/nirs/data1/dataTimeSeries')'; 
info_snirf = h5info(file_snirf, '/nirs/data1');
num_elements = length(info_snirf.Groups);
data_labels = cell(num_elements, 1);
for i = 1:num_elements
    path_ml = info_snirf.Groups(i).Name; 
    raw_label = h5read(file_snirf, [path_ml, '/dataTypeLabel']);
    data_labels{i} = strtrim(char(raw_label));
end
idx_hbo = find(strcmp(data_labels, 'HbO')); 
num_channels = length(idx_hbo);

y_hbo = zeros(num_time_points, num_channels);
for ch = 1:num_channels
    y_hbo(:, ch) = data_time_series(:, idx_hbo(ch));
end

% Identify Short Channels (< 15 mm)
src_pos = h5read(file_snirf_raw, '/nirs/probe/sourcePos3D')';
det_pos = h5read(file_snirf_raw, '/nirs/probe/detectorPos3D')';
sd_dist = zeros(num_channels, 1);
for ch = 1:num_channels
    path_ch = sprintf('/nirs/data1/measurementList%d', ch);
    s_idx = double(h5read(file_snirf, [path_ch, '/sourceIndex']));
    d_idx = double(h5read(file_snirf, [path_ch, '/detectorIndex']));
    sd_dist(ch) = norm(src_pos(s_idx, :) - det_pos(d_idx, :));
end
short_channels = find(sd_dist < 15)';

% Extract clean SC PCA PC1
sc_data_hbo = y_hbo(:, short_channels);
R_sc_hbo = corr(zscore(sc_data_hbo), 'Rows', 'complete');
mean_corr_hbo = zeros(length(short_channels), 1);
for ch = 1:length(short_channels)
    other_idx = setdiff(1:length(short_channels), ch);
    mean_corr_hbo(ch) = mean(R_sc_hbo(ch, other_idx), 'omitnan');
end
sc_data_clean_hbo = y_hbo(:, short_channels(mean_corr_hbo >= 0.1));
[~, score_sc_hbo] = pca(zscore(sc_data_clean_hbo));
sc_pca1_hbo = zscore(score_sc_hbo(:, 1));

% Load and align ECG HR
physio_table = readtable(file_csv); 
hr_ecg_aligned = interp1(physio_table.Time_s, physio_table.HR_ECG, time_vector, 'linear', 'extrap');

%% 2. Define experimental masks
all_onsets = [onsets_stim1; onsets_stim2; onsets_stim3];
first_onset = min(all_onsets);
idx_baseline = time_vector >= (first_onset - 30) & time_vector < first_onset;

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

idx_single_task = time_vector >= onsets_stim1(1) & time_vector <= (onsets_stim1(1) + task_duration);

%% 3. Power Spectral Density (PSD) analysis
[psd_hr, f_hr]   = periodogram(zscore(hr_ecg_aligned), hamming(num_time_points), [], fs_fnirs);
[psd_hbo, f_hbo] = periodogram(sc_pca1_hbo, hamming(num_time_points), [], fs_fnirs);

% Generate 1/f^beta colored noise
num_unique_pts = ceil((num_time_points + 1) / 2);
f_bins = (1:num_unique_pts)';
filter_amp = 1 ./ (f_bins .^ (noise_exponent / 2));

w_synth = randn(num_time_points, 1);
W_synth = fft(w_synth);
W_shaped_synth = W_synth;
W_shaped_synth(1:num_unique_pts) = W_synth(1:num_unique_pts) .* filter_amp;
if mod(num_time_points, 2) == 0
    W_shaped_synth(num_unique_pts+1:end) = conj(W_shaped_synth(num_unique_pts-1:-1:2));
else
    W_shaped_synth(num_unique_pts+1:end) = conj(W_shaped_synth(num_unique_pts:-1:2));
end
W_shaped_synth(1) = 0;
synthetic_noise = zscore(real(ifft(W_shaped_synth)));
[psd_pink, f_pink] = periodogram(synthetic_noise, hamming(num_time_points), [], fs_fnirs);

% Plot PSD
fig_psd = figure('Name', 'PSD Analysis', 'Color', 'w', 'Position', [100 100 1000 450]);

subplot(1, 2, 1);
loglog(f_pink, psd_pink, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5); hold on;
loglog(f_hr, psd_hr, 'Color', [0.85 0.25 0.25], 'LineWidth', 1.8);
grid on; xlim([0.01 fs_fnirs/2]);
xlabel('Frequency (Hz)'); ylabel('Power Density (V^2/Hz)');
title(sprintf('HR (ECG) vs. 1/f^{%.1f} Noise', noise_exponent), 'FontWeight', 'bold');
legend({'1/f Synthetic Noise', 'HR (ECG)'}, 'Location', 'southwest');

subplot(1, 2, 2);
loglog(f_pink, psd_pink, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5); hold on;
loglog(f_hbo, psd_hbo, 'Color', [0.2 0.45 0.75], 'LineWidth', 1.8);
grid on; xlim([0.01 fs_fnirs/2]);
xlabel('Frequency (Hz)'); ylabel('Power Density (V^2/Hz)');
title(sprintf('SC PCA HbO vs. 1/f^{%.1f} Noise', noise_exponent), 'FontWeight', 'bold');
legend({'1/f Synthetic Noise', 'SC PCA HbO'}, 'Location', 'southwest');

% Detect cardiac peak in optical signal (0.8 - 1.8 Hz)
c_idx = find(f_hbo >= 0.8 & f_hbo <= 1.8);
if ~isempty(c_idx)
    [~, max_c] = max(psd_hbo(c_idx));
    xline(f_hbo(c_idx(max_c)), 'r--', sprintf('Cardiac Peak (%.2f Hz)', f_hbo(c_idx(max_c))), 'LineWidth', 1.2);
end

sgtitle(sprintf('1/f^{%.1f} Power Law Spectrum - %s', noise_exponent, subject_name), 'FontWeight', 'bold');
fig_psd_file = fullfile(subj_dir, [subject_name, '_PSD_Analysis.png']);
try exportgraphics(fig_psd, fig_psd_file, 'Resolution', 300); catch; saveas(fig_psd, fig_psd_file); end

%% 4. Monte Carlo simulation: Spurious correlations
corr_dist_raw = zeros(num_iterations, 4);
corr_dist_det = zeros(num_iterations, 4);
corr_single_raw = zeros(num_iterations, 1);
corr_single_det = zeros(num_iterations, 1);

fprintf('Running Monte Carlo simulation (N = %d iterations)...\n', num_iterations);

for iter = 1:num_iterations
    % Two independent 1/f noise signals
    w_sim = randn(num_time_points, 2);
    W_sim = fft(w_sim);
    W_shaped_sim = W_sim;
    W_shaped_sim(1:num_unique_pts, :) = W_sim(1:num_unique_pts, :) .* filter_amp;
    if mod(num_time_points, 2) == 0
        W_shaped_sim(num_unique_pts+1:end, :) = conj(W_shaped_sim(num_unique_pts-1:-1:2, :));
    else
        W_shaped_sim(num_unique_pts+1:end, :) = conj(W_shaped_sim(num_unique_pts:-1:2, :));
    end
    W_shaped_sim(1, :) = 0;
    p_sim = real(ifft(W_shaped_sim));
    p1 = zscore(p_sim(:, 1));
    p2 = zscore(p_sim(:, 2));
    
    % Raw correlations
    corr_dist_raw(iter, 1) = abs(corr(p1(idx_baseline), p2(idx_baseline), 'Rows', 'complete'));
    corr_dist_raw(iter, 2) = abs(corr(p1(idx_ft), p2(idx_ft), 'Rows', 'complete'));
    corr_dist_raw(iter, 3) = abs(corr(p1(idx_bh), p2(idx_bh), 'Rows', 'complete'));
    corr_dist_raw(iter, 4) = abs(corr(p1(idx_comb), p2(idx_comb), 'Rows', 'complete'));
    corr_single_raw(iter)  = abs(corr(p1(idx_single_task), p2(idx_single_task), 'Rows', 'complete'));
    
    % Block-by-block detrended correlations
    corr_dist_det(iter, 1) = abs(corr(detrend(p1(idx_baseline)), detrend(p2(idx_baseline)), 'Rows', 'complete'));
    corr_single_det(iter)  = abs(corr(detrend(p1(idx_single_task)), detrend(p2(idx_single_task)), 'Rows', 'complete'));
    
    % FT blocks detrended individually
    p1_ft_det = []; p2_ft_det = [];
    for i = 1:length(onsets_stim1)
        idx = time_vector >= onsets_stim1(i) & time_vector <= (onsets_stim1(i) + task_duration);
        p1_ft_det = [p1_ft_det; detrend(p1(idx))];
        p2_ft_det = [p2_ft_det; detrend(p2(idx))];
    end
    corr_dist_det(iter, 2) = abs(corr(p1_ft_det, p2_ft_det, 'Rows', 'complete'));
    
    % BH blocks detrended individually
    p1_bh_det = []; p2_bh_det = [];
    for i = 1:length(onsets_stim2)
        idx = time_vector >= onsets_stim2(i) & time_vector <= (onsets_stim2(i) + task_duration);
        p1_bh_det = [p1_bh_det; detrend(p1(idx))];
        p2_bh_det = [p2_bh_det; detrend(p2(idx))];
    end
    corr_dist_det(iter, 3) = abs(corr(p1_bh_det, p2_bh_det, 'Rows', 'complete'));
    
    % COMB blocks detrended individually
    p1_comb_det = []; p2_comb_det = [];
    for i = 1:length(onsets_stim3)
        idx = time_vector >= onsets_stim3(i) & time_vector <= (onsets_stim3(i) + task_duration);
        p1_comb_det = [p1_comb_det; detrend(p1(idx))];
        p2_comb_det = [p2_comb_det; detrend(p2(idx))];
    end
    corr_dist_det(iter, 4) = abs(corr(p1_comb_det, p2_comb_det, 'Rows', 'complete'));
end

mean_raw = mean(corr_dist_raw, 1);
mean_det = mean(corr_dist_det, 1);

fprintf('\nMonte Carlo Results (Mean |R| across %d iterations):\n', num_iterations);
fprintf('   Baseline (30s):    Raw = %.3f | Detrended = %.3f\n', mean_raw(1), mean_det(1));
fprintf('   FT (210s concat):  Raw = %.3f | Detrended = %.3f\n', mean_raw(2), mean_det(2));
fprintf('   BH (105s concat):  Raw = %.3f | Detrended = %.3f\n', mean_raw(3), mean_det(3));
fprintf('   COMB (105s concat):Raw = %.3f | Detrended = %.3f\n', mean_raw(4), mean_det(4));
fprintf('   Single block (21s):Raw = %.3f | Detrended = %.3f\n', mean(corr_single_raw), mean(corr_single_det));

%% 5. Plot Detrending vs. Raw Noise Comparison
fig_cmp = figure('Name', 'Detrending vs Raw Noise Comparison', 'Color', 'w', 'Position', [150 150 1000 450]);

subplot(1, 2, 1);
histogram(corr_single_raw, 'Normalization', 'pdf', 'FaceColor', [0.85 0.25 0.25], 'EdgeColor', 'k', 'FaceAlpha', 0.3); hold on;
histogram(corr_dist_raw(:, 2), 'Normalization', 'pdf', 'FaceColor', [0.2 0.45 0.75], 'EdgeColor', 'k', 'FaceAlpha', 0.3);
histogram(corr_dist_det(:, 2), 'Normalization', 'pdf', 'FaceColor', [0.47 0.67 0.19], 'EdgeColor', 'k', 'FaceAlpha', 0.6);
xline(mean(corr_single_raw), 'Color', [0.85 0.25 0.25], 'LineStyle', '--', 'LineWidth', 1.8);
xline(mean_raw(2), 'Color', [0.2 0.45 0.75], 'LineStyle', '--', 'LineWidth', 1.8);
xline(mean_det(2), 'Color', [0.47 0.67 0.19], 'LineStyle', '--', 'LineWidth', 2.0);
xlim([0 1]); xlabel('|R|'); ylabel('Probability Density');
title('Fingertapping Task Blocks', 'FontWeight', 'bold');
legend({'Single Block Raw (21s)', 'Concatenated Raw (210s)', 'Concatenated Detrended (210s)'}, 'Location', 'northeast');
grid on;

subplot(1, 2, 2);
histogram(corr_dist_raw(:, 1), 'Normalization', 'pdf', 'FaceColor', [0.2 0.45 0.75], 'EdgeColor', 'k', 'FaceAlpha', 0.4); hold on;
histogram(corr_dist_det(:, 1), 'Normalization', 'pdf', 'FaceColor', [0.47 0.67 0.19], 'EdgeColor', 'k', 'FaceAlpha', 0.6);
xline(mean_raw(1), 'Color', [0.2 0.45 0.75], 'LineStyle', '--', 'LineWidth', 1.8);
xline(mean_det(1), 'Color', [0.47 0.67 0.19], 'LineStyle', '--', 'LineWidth', 2.0);
xlim([0 1]); xlabel('|R|'); ylabel('Probability Density');
title('Baseline Block (30s)', 'FontWeight', 'bold');
legend({'Baseline Raw', 'Baseline Detrended'}, 'Location', 'northeast');
grid on;

sgtitle(sprintf('Detrending Effect on Spurious 1/f^{%.1f} Noise Correlation - %s', noise_exponent, subject_name), 'FontWeight', 'bold');
fig_cmp_file = fullfile(subj_dir, [subject_name, '_Noise_Simulation_Comparison.png']);
try exportgraphics(fig_cmp, fig_cmp_file, 'Resolution', 300); catch; saveas(fig_cmp, fig_cmp_file); end

fprintf('Noise simulation complete. Outputs saved to %s.\n', subj_dir);
