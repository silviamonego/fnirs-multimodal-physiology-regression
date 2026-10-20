%% PHYSIOLOGICAL PREPROCESSING AND FEATURE EXTRACTION
% Preprocesses multimodal autonomic signals (NIRx WINGS2) and synchronizes with fNIRS (.snirf).
%
% Operations performed:
%   1. Respiration: Savitzky-Golay filtering and apnea compliance validation for BH/COMB.
%   2. EDA/GSR: Downsampling to 4 Hz, z-score normalization, and cvxEDA tonic/phasic decomposition.
%   3. PPG: Bandpass filtering (0.5-5 Hz), systolic peak detection, and continuous HR interpolation.
%   4. ECG: Bandpass filtering (0.5-40 Hz), QRS R-peak detection, and gold-standard HR derivation.
%   5. SpO2: Outlier rejection and median filtering.
%   6. Export: Saves <subject>_ProcessedPhysio.csv, <subject>_ValidTrials.mat, and <subject>_PhysioRegressors.sdm.

clearvars -except subj_dir; clc; close all;

% Add toolboxes if present
script_dir = fileparts(mfilename('fullpath'));
external_dirs = { ...
    'cvxEDA', '../cvxEDA', '../../cvxEDA', ...
    'Homer3_analysis', '../Homer3_analysis', '../../Homer3_analysis', ...
    'homer2', '../homer2', '../../homer2', ...
    fullfile(script_dir, '..', '..', 'Homer3_analysis'), ...
    fullfile(script_dir, '..', '..', 'homer2') ...
};
for d = 1:numel(external_dirs)
    if exist(external_dirs{d}, 'dir'), addpath(genpath(external_dirs{d})); end
end

% Prioritize Signal Processing Toolbox findpeaks
spt_path = fullfile(matlabroot, 'toolbox', 'signal', 'signal');
if exist(spt_path, 'dir'), addpath(spt_path, '-begin'); end

% Channel mapping for physiological signals in .wings file
CH_RESP = 9;   % Thoracic respiration
CH_GSR  = 10;  % Galvanic Skin Response
CH_ECG  = 11;  % ECG Lead II
CH_PPG  = 12;  % Earlobe PPG
CH_SPO2 = 13;  % SpO2
CH_HR   = 14;  % Hardware HR

experimental_task_duration = 21; % seconds
fs_target = 4;                   % Target sampling rate for EDA (Hz)

%% 1. Select subject folder and load raw signals
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
file_wings = fullfile(subj_dir, [subject_name, '.wings']);
file_snirf = fullfile(subj_dir, [subject_name, '.snirf']);

assert(exist(file_wings, 'file') == 2, 'Wings file not found: %s', file_wings);
assert(exist(file_snirf, 'file') == 2, 'SNIRF file not found: %s', file_snirf);

fprintf('Processing subject: %s\n', subject_name);

raw_wings = importdata(file_wings);
data_matrix = raw_wings.data;

real_time   = data_matrix(:, 1);  
respiration = data_matrix(:, CH_RESP);  
gsr         = data_matrix(:, CH_GSR); 
ecg         = data_matrix(:, CH_ECG); 
ppg_raw     = data_matrix(:, CH_PPG); 
spo2        = data_matrix(:, CH_SPO2); 
heart_rate  = data_matrix(:, CH_HR);

fs_original = 1 / mean(diff(real_time));
fprintf('   -> Original physiological sampling rate: %.2f Hz\n', fs_original);

%% 2. Extract triggers from .snirf
info_snirf = h5info(file_snirf, '/nirs');
group_list = {info_snirf.Groups.Name};
stim_indices = find(contains(group_list, '/nirs/stim'));

trigger_struct = struct('name', {}, 'onsets', {}, 'durations', {});
valid_count = 0;

for c = 1:length(stim_indices)
    path_base = sprintf('/nirs/stim%d', c);
    raw_name = h5read(file_snirf, [path_base, '/name']);
    if iscell(raw_name), cond_name = strtrim(raw_name{1});
    else, cond_name = strtrim(char(raw_name')); end
    
    if ismember(cond_name, {'1', '2', '3'})
        valid_count = valid_count + 1;
        stim_data = h5read(file_snirf, [path_base, '/data']);
        if size(stim_data, 2) == 3, onsets = stim_data(:, 1);
        else, onsets = stim_data(1, :)'; end
        
        switch cond_name
            case '1', mapped_name = 'FT';
            case '2', mapped_name = 'BH';
            case '3', mapped_name = 'COMB';
        end
        trigger_struct(valid_count).name = mapped_name;
        trigger_struct(valid_count).onsets = onsets;
        trigger_struct(valid_count).durations = experimental_task_duration * ones(length(onsets), 1);
    end
end

% Set timeline bounds
all_onsets_exp = vertcat(trigger_struct.onsets);
all_dur_exp    = vertcat(trigger_struct.durations);
exp_start_time = max(0, min(all_onsets_exp) - 60);
exp_end_time   = min(real_time(end), max(all_onsets_exp + all_dur_exp) + 60);

%% 3. Respiration preprocessing and apnea trial validation
frame_length = round(1.5 * fs_original);
if mod(frame_length, 2) == 0, frame_length = frame_length + 1; end
respiration_smoothed = sgolayfilt(respiration, 3, frame_length);

% Apnea validation threshold (detect non-compliant breathing during BH/COMB)
window_sec = [-2, 2];
threshold_multiplier = 1.8;
normal_breath_amp = prctile(respiration_smoothed, 95);

valid_onsets_struct = struct('BH', [], 'COMB', []);
tInc_physio = ones(length(real_time), 1);

validation_tasks = {'BH', 'COMB'};
for task_idx = 1:2
    t_name = validation_tasks{task_idx};
    v_idx = find(strcmp({trigger_struct.name}, t_name));
    if isempty(v_idx), continue; end
    
    onsets_task = trigger_struct(v_idx).onsets;
    n_trials = length(onsets_task);
    is_valid = true(n_trials, 1);
    
    for i = 1:n_trials
        t_start = onsets_task(i);
        idx_win = real_time >= (t_start + window_sec(1)) & real_time <= (t_start + window_sec(2));
        if max(respiration_smoothed(idx_win)) > (normal_breath_amp * threshold_multiplier)
            is_valid(i) = false;
            idx_trial = real_time >= t_start & real_time <= (t_start + experimental_task_duration);
            tInc_physio(idx_trial) = 0;
        end
    end
    valid_onsets_struct.(t_name) = onsets_task(is_valid);
    fprintf('   -> %s compliance: %d/%d valid trials retained.\n', t_name, sum(is_valid), n_trials);
end

valid_onsets_BH = valid_onsets_struct.BH;
valid_onsets_COMB = valid_onsets_struct.COMB;
v_ft = find(strcmp({trigger_struct.name}, 'FT'));
if ~isempty(v_ft), valid_onsets_FT = trigger_struct(v_ft).onsets; else, valid_onsets_FT = []; end

valid_trials_file = fullfile(subj_dir, [subject_name, '_ValidTrials.mat']);
save(valid_trials_file, 'valid_onsets_FT', 'valid_onsets_BH', 'valid_onsets_COMB', 'tInc_physio');

%% 4. EDA/GSR preprocessing and cvxEDA decomposition
[gsr_downsampled, time_downsampled] = resample(gsr, real_time, fs_target);
gsr_normalized = (gsr_downsampled - mean(gsr_downsampled)) / std(gsr_downsampled);

delta = 1 / fs_target;
tau0 = 2.0; tau1 = 0.7; delta_knot = 10; alpha = 0.0008; gamma = 0.01;

fprintf('   -> Running cvxEDA decomposition at %d Hz...\n', fs_target);
[gsr_phasic, ~, gsr_tonic] = cvxEDA(gsr_normalized, delta, tau0, tau1, delta_knot, alpha, gamma);

%% 5. PPG preprocessing and continuous HR derivation
[b_ppg, a_ppg] = butter(3, [0.5 5] / (fs_original / 2), 'bandpass');
ppg_filtered = filtfilt(b_ppg, a_ppg, ppg_raw);

win_samples = round(15 * fs_original);
local_std = movstd(ppg_filtered, win_samples);
local_max_amp = movmax(ppg_filtered, win_samples);
local_thresh = 0.5 * local_std;

[all_pks, all_locs, ~, all_prom] = findpeaks(ppg_filtered, ...
    'MinPeakDistance', 0.45 * fs_original, 'MinPeakHeight', 0);
valid_idx = (all_prom >= local_thresh(all_locs)) & (all_pks >= 0.25 * local_max_amp(all_locs));
locs = all_locs(valid_idx);

beat_times = real_time(locs);
instantaneous_hr = 60 ./ diff(beat_times);
hr_times = beat_times(2:end);
valid_mask = (instantaneous_hr >= 40) & (instantaneous_hr <= 140);
hr_clean = medfilt1(instantaneous_hr(valid_mask), 5, 'truncate');
continuous_hr = interp1(hr_times(valid_mask), hr_clean, real_time, 'pchip', 'extrap');

%% 6. ECG preprocessing and R-peak detection
[b_ecg, a_ecg] = butter(3, [0.5 40] / (fs_original / 2), 'bandpass');
ecg_filtered = filtfilt(b_ecg, a_ecg, ecg);

min_dist_ecg = 0.35 * fs_original;
h_thresh_ecg = max(0.3, 0.25 * max(ecg_filtered));
[~, locs_ecg] = findpeaks(ecg_filtered, ...
    'MinPeakDistance', min_dist_ecg, ...
    'MinPeakHeight', h_thresh_ecg, ...
    'MinPeakProminence', h_thresh_ecg);

beat_times_ecg = real_time(locs_ecg);
instantaneous_hr_ecg = 60 ./ diff(beat_times_ecg);
hr_times_ecg = beat_times_ecg(2:end);
valid_mask_ecg = (instantaneous_hr_ecg >= 40) & (instantaneous_hr_ecg <= 140);
hr_clean_ecg = medfilt1(instantaneous_hr_ecg(valid_mask_ecg), 5, 'truncate');
continuous_hr_ecg = interp1(hr_times_ecg(valid_mask_ecg), hr_clean_ecg, real_time, 'pchip', 'extrap');

%% 7. SpO2 outlier removal and smoothing
spo2_clean = spo2;
spo2_clean(spo2_clean > 100) = 100;
spo2_clean(spo2_clean < 70) = NaN;
spo2_clean = fillmissing(spo2_clean, 'nearest');
spo2_filtered = medfilt1(spo2_clean, round(5 * fs_original), 'truncate');

%% 8. Export synchronized regressors (CSV and SDM)
gsr_tonic_interp = interp1(time_downsampled, gsr_tonic, real_time, 'pchip', 'extrap');
gsr_phasic_interp = interp1(time_downsampled, gsr_phasic, real_time, 'pchip', 'extrap');

physio_table = table(real_time(:), ...
                     respiration_smoothed(:), ...
                     gsr_tonic_interp(:), ...
                     gsr_phasic_interp(:), ...
                     continuous_hr(:), ...
                     continuous_hr_ecg(:), ...
                     spo2_filtered(:), ...
                     'VariableNames', {'Time_s', 'Resp_Clean', 'EDA_Tonic', 'EDA_Phasic', 'HR_PPG', 'HR_ECG', 'SpO2_Clean'});

output_csv = fullfile(subj_dir, [subject_name, '_ProcessedPhysio.csv']);
writetable(physio_table, output_csv);
fprintf('   -> Saved processed physiology: %s\n', output_csv);

% Export Satori design matrix (.sdm)
sdm_file = fullfile(subj_dir, [subject_name, '_PhysioRegressors.sdm']);
pred_names = physio_table.Properties.VariableNames(2:end);
n_preds = length(pred_names);
n_rows = height(physio_table);
sdm_mat = zscore(table2array(physio_table(:, 2:end)));

fid = fopen(sdm_file, 'w');
fprintf(fid, 'FileVersion:        1\nNrOfPredictors:     %d\nNrOfDataPoints:     %d\nIncludesConstant:   0\nFirstConfoundPredictor: 1\n\n', n_preds, n_rows);
for i = 1:n_preds, fprintf(fid, '150 150 150\n'); end
fprintf(fid, '\n');
for i = 1:n_preds, fprintf(fid, '"%s" ', pred_names{i}); end
fprintf(fid, '\n');
for r = 1:n_rows
    fprintf(fid, '%f ', sdm_mat(r, :));
    fprintf(fid, '\n');
end
fclose(fid);
fprintf('   -> Saved Satori SDM matrix: %s\n', sdm_file);

%% 9. Generate diagnostic figures
fig_overview = figure('Name', 'Physiological Preprocessing Overview', 'Color', 'w', 'Position', [100 100 1100 700]);

subplot(4, 1, 1);
plot(real_time, respiration_smoothed, 'Color', [0 0.45 0.74], 'LineWidth', 1.2);
title('Respiration (Savitzky-Golay Filtered)'); ylabel('A.U.'); grid on; xlim([exp_start_time exp_end_time]);

subplot(4, 1, 2);
plot(time_downsampled, gsr_tonic, 'Color', [0.85 0.33 0.1], 'LineWidth', 1.2); hold on;
plot(time_downsampled, gsr_phasic, 'Color', [0.47 0.67 0.19], 'LineWidth', 1.2);
title('EDA Decomposition (cvxEDA)'); ylabel('Amplitude'); legend('Tonic (SCL)', 'Phasic (SCR)'); grid on; xlim([exp_start_time exp_end_time]);

subplot(4, 1, 3);
plot(real_time, continuous_hr_ecg, 'Color', [0.49 0.18 0.56], 'LineWidth', 1.2); hold on;
plot(real_time, continuous_hr, 'Color', [0.47 0.67 0.19], 'LineWidth', 1.0);
title('Heart Rate (ECG Gold Standard vs PPG Derived)'); ylabel('BPM'); legend('HR ECG', 'HR PPG'); grid on; xlim([exp_start_time exp_end_time]);

subplot(4, 1, 4);
plot(real_time, spo2_filtered, 'Color', [0.3 0.75 0.93], 'LineWidth', 1.2);
title('Oxygen Saturation (SpO2 Median Filtered)'); ylabel('%'); xlabel('Time (s)'); grid on; xlim([exp_start_time exp_end_time]);

fig_path = fullfile(subj_dir, [subject_name, '_Physio_Overview.png']);
try
    exportgraphics(fig_overview, fig_path, 'Resolution', 300);
catch
    saveas(fig_overview, fig_path);
end

fprintf('Preprocessing complete for subject %s.\n', subject_name);
