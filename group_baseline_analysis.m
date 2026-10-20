%% GROUP_BASELINE_ANALYSIS
% Group-level trial survival and baseline physiological response analysis.
% Computes:
%   1. Trial retention and respiratory pre-screening compliance across tasks (FT, BH, COMB).
%   2. Group physiological statistics (HR, SpO2, EDA) for validated blocks.

clearvars; clc; close all;

% Set dataset directory
script_dir = fileparts(mfilename('fullpath'));
candidates = { ...
    fullfile(pwd, 'DATASET'), ...
    fullfile(script_dir, '..', '..', 'DATASET'), ...
    fullfile(script_dir, '..', 'DATASET'), ...
    fullfile(pwd, '..', 'DATASET'), ...
    fullfile(pwd, '..', '..', 'DATASET') ...
};
dataset_dir = '';
for i = 1:numel(candidates)
    if exist(candidates{i}, 'dir') && ~isempty(dir(fullfile(candidates{i}, '20*')))
        dataset_dir = candidates{i};
        break;
    end
end
if isempty(dataset_dir)
    dataset_dir = uigetdir(pwd, 'Select DATASET directory');
    if isequal(dataset_dir, 0)
        disp('Operation canceled by user.');
        return;
    end
end

output_dir = fullfile(dataset_dir, 'group_analysis_results');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% Discover valid subjects
sub_dirs = dir(dataset_dir);
sub_dirs = sub_dirs([sub_dirs.isdir]);
sub_dirs = sub_dirs(~ismember({sub_dirs.name}, {'.', '..', 'group_analysis_results', 'group_outputs'}));
sub_dirs = sub_dirs(~contains({sub_dirs.name}, 'Emma', 'IgnoreCase', true));

n_sub = numel(sub_dirs);
subject_ids = strings(n_sub, 1);
for s = 1:n_sub
    subject_ids(s) = string(sub_dirs(s).name);
end
subject_ids = sort(subject_ids);

fprintf('=======================================================\n');
fprintf('Group Baseline Analysis: N = %d subjects\n', n_sub);
fprintf('=======================================================\n\n');

%% 1. Trial Survival and Compliance Summary
trial_table = table('Size', [n_sub, 6], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'Subject', 'FT_Valid', 'BH_Valid', 'COMB_Valid', 'Total_Valid', 'Percent_Retained'});

for s = 1:n_sub
    val_file = fullfile(dataset_dir, subject_ids(s), [char(subject_ids(s)) '_ValidTrials.mat']);
    if exist(val_file, 'file')
        V = load(val_file);
        n_ft = numel(V.valid_onsets_FT);
        n_bh = numel(V.valid_onsets_BH);
        n_comb = numel(V.valid_onsets_COMB);
        tot = n_ft + n_bh + n_comb;
        pct = (tot / 20) * 100; % 20 total planned blocks (10 FT, 5 BH, 5 COMB)
        trial_table(s, :) = {subject_ids(s), n_ft, n_bh, n_comb, tot, pct};
    end
end

disp('--- TRIAL SURVIVAL TABLE ---');
disp(trial_table);
fprintf('FT Retained:   %d/%d (%.1f%%)\n', sum(trial_table.FT_Valid), n_sub * 10, sum(trial_table.FT_Valid) / (n_sub * 10) * 100);
fprintf('BH Retained:   %d/%d (%.1f%%)\n', sum(trial_table.BH_Valid), n_sub * 5, sum(trial_table.BH_Valid) / (n_sub * 5) * 100);
fprintf('COMB Retained: %d/%d (%.1f%%)\n', sum(trial_table.COMB_Valid), n_sub * 5, sum(trial_table.COMB_Valid) / (n_sub * 5) * 100);
fprintf('Total Blocks:  %d/%d (%.1f%%)\n\n', sum(trial_table.Total_Valid), n_sub * 20, sum(trial_table.Total_Valid) / (n_sub * 20) * 100);

writetable(trial_table, fullfile(output_dir, 'Table0_Group_Trial_Retention.csv'));

%% 2. Optional: External Figure Generation
py_script = fullfile(fileparts(mfilename('fullpath')), 'generate_group_baseline_figures.py');
if exist(py_script, 'file')
    fprintf('Executing figure generation: %s\n', py_script);
    system(sprintf('python3 "%s"', py_script));
end
