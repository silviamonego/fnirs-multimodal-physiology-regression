function group_results = group_analysis(input_root, output_dir)
% GROUP_ANALYSIS
% Group-level statistical analysis and figure generation across subjects.
%
% Outcomes evaluated:
%   1. BH_CV_nRMSE: Held-out breath-holding prediction error (nRMSE)
%   2. C3 HRF RMSE: Local temporal error in motor cortex C3 channel (uM)
%   3. Spatial Map Correlation: Topographical Pearson correlation with FT reference
%   4. Model Selection: 1-SE parsimony rule frequency
%
% Usage:
%   group_analysis();
%   group_analysis(input_root);
%   group_analysis(input_root, output_dir);

default_dataset_dir = fullfile(pwd, 'DATASET');
if ~exist(default_dataset_dir, 'dir')
    default_dataset_dir = pwd;
end

if nargin < 1 || isempty(input_root)
    input_root = uigetdir(default_dataset_dir, 'Select directory containing subject CSV summaries');
    if isequal(input_root, 0)
        disp('Operation cancelled by user.');
        group_results = struct();
        return;
    end
end
if nargin < 2 || isempty(output_dir)
    output_dir = fullfile(input_root, 'group_analysis_results');
end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

%% 1. Load and aggregate subject summary tables
files = dir(fullfile(input_root, '**', '*_IntegratedSpatial_ModelSummary.csv'));
assert(~isempty(files), 'No *_IntegratedSpatial_ModelSummary.csv files found.');

n_subjects = numel(files);
subject_ids = strings(n_subjects, 1);
tables = cell(n_subjects, 1);

for s = 1:n_subjects
    subject_ids(s) = erase(string(files(s).name), '_IntegratedSpatial_ModelSummary.csv');
    tables{s} = readtable(fullfile(files(s).folder, files(s).name), 'TextType', 'string');
    tables{s}.Subject = repmat(subject_ids(s), height(tables{s}), 1);
end

long_table = vertcat(tables{:});
model_ids = 0:6;
model_names = ["M0 (Raw)", "M1 (SC)", "M2 (+Resp)", "M3 (+ECG)", ...
               "M4 (+SpO2)", "M5 (+EDA ph)", "M6 (+EDA ton)"];

fprintf('\n=======================================================\n');
fprintf('Group-Level Analysis: %d subjects, %d models (M0-M6)\n', n_subjects, numel(model_ids));
fprintf('=======================================================\n');

%% 2. Extract outcome matrices (Subjects x Models)
bh_nrmse   = extract_matrix(long_table, subject_ids, model_ids, 'BH_CV_nRMSE_Combined');
c3_rmse    = extract_matrix(long_table, subject_ids, model_ids, 'COMB_to_HomerFT_HRF_RMSE_C3_HbO_uM');
spatial_r  = extract_matrix(long_table, subject_ids, model_ids, 'SpatialMap_R_HbO');

% Extract subject-specific 1-SE selected model ID
selected_1se_id = nan(n_subjects, 1);
for s = 1:n_subjects
    row = long_table.Subject == subject_ids(s) & long_table.SelectedByOneSE;
    selected_1se_id(s) = long_table.ModelID(row);
end

%% 3. Model Descriptive Performance Table (Median [IQR])
table_perf = table('Size', [numel(model_ids), 8], ...
    'VariableTypes', {'double','string','double','double','double','double','double','double'}, ...
    'VariableNames', {'ModelID','Model','BH_nRMSE_Median','BH_nRMSE_IQR', ...
                      'C3_RMSE_Median_uM','C3_RMSE_IQR_uM','SpatialR_Median','SpatialR_IQR'});

for m = 1:numel(model_ids)
    table_perf.ModelID(m)           = model_ids(m);
    table_perf.Model(m)             = model_names(m);
    table_perf.BH_nRMSE_Median(m)   = median(bh_nrmse(:, m));
    table_perf.BH_nRMSE_IQR(m)      = iqr(bh_nrmse(:, m));
    table_perf.C3_RMSE_Median_uM(m) = median(c3_rmse(:, m));
    table_perf.C3_RMSE_IQR_uM(m)    = iqr(c3_rmse(:, m));
    table_perf.SpatialR_Median(m)   = median(spatial_r(:, m));
    table_perf.SpatialR_IQR(m)      = iqr(spatial_r(:, m));
end

%% 4. Hypothesis Testing (Wilcoxon Signed-Rank Test with Holm Correction)
adaptive_c3_rmse = nan(n_subjects, 1);
adaptive_spatial = nan(n_subjects, 1);
for s = 1:n_subjects
    col = find(model_ids == selected_1se_id(s));
    adaptive_c3_rmse(s) = c3_rmse(s, col);
    adaptive_spatial(s) = spatial_r(s, col);
end

comparisons = {
    'BH nRMSE: M0 (Raw) vs M1 (Short Channels)',     bh_nrmse(:, 1),   bh_nrmse(:, 2);
    'BH nRMSE: M1 (SC) vs M2 (+ Respiration)',       bh_nrmse(:, 2),   bh_nrmse(:, 3);
    'BH nRMSE: M1 (SC) vs M4 (Cardiorespiratory)',   bh_nrmse(:, 2),   bh_nrmse(:, 5);
    'C3 RMSE: M1 (SC) vs M4 (Cardiorespiratory)',    c3_rmse(:, 2),    c3_rmse(:, 5);
    'Spatial R: M1 (SC) vs M4 (Cardiorespiratory)',  spatial_r(:, 2),  spatial_r(:, 5);
    'C3 RMSE: M1 (SC) vs Adaptive (1-SE)',           c3_rmse(:, 2),    adaptive_c3_rmse;
    'Spatial R: M1 (SC) vs Adaptive (1-SE)',         spatial_r(:, 2),  adaptive_spatial
};

n_comp = size(comparisons, 1);
table_tests = table('Size', [n_comp, 6], ...
    'VariableTypes', {'string','double','double','double','double','double'}, ...
    'VariableNames', {'Comparison','Reference_Median','Test_Median','Median_Difference','Wilcoxon_P_Value','Holm_P_Value'});

for c = 1:n_comp
    ref_vals  = comparisons{c, 2};
    test_vals = comparisons{c, 3};
    diff_vals = test_vals - ref_vals;
    
    p = signrank(ref_vals, test_vals);
    
    table_tests.Comparison(c)        = string(comparisons{c, 1});
    table_tests.Reference_Median(c)  = median(ref_vals);
    table_tests.Test_Median(c)       = median(test_vals);
    table_tests.Median_Difference(c) = median(diff_vals);
    table_tests.Wilcoxon_P_Value(c)  = p;
end

% Holm-Bonferroni correction
[sorted_p, sort_idx] = sort(table_tests.Wilcoxon_P_Value);
holm_p = zeros(n_comp, 1);
for i = 1:n_comp
    holm_p(i) = sorted_p(i) * (n_comp - i + 1);
end
holm_p = min(cummax(holm_p), 1);
table_tests.Holm_P_Value(sort_idx) = holm_p;

%% 5. Model Selection Frequencies (1-SE Rule)
selection_counts = histcounts(selected_1se_id, -0.5:1:6.5)';
table_selection = table(model_ids', model_names', selection_counts, ...
    100 * selection_counts / n_subjects, ...
    'VariableNames', {'ModelID','Model','Count','Percentage'});

%% 6. Save Tables
writetable(table_perf, fullfile(output_dir, 'Table1_Group_Model_Performance.csv'));
writetable(table_tests, fullfile(output_dir, 'Table2_Group_Hypothesis_Tests.csv'));
writetable(table_selection, fullfile(output_dir, 'Table3_Model_Selection_Counts.csv'));

%% 7. Generate Publication Figure
fig = figure('Color', 'w', 'Position', [100 100 1100 800]);

% A) BH CV nRMSE
subplot(2, 2, 1); hold on;
plot(model_ids, bh_nrmse', '-o', 'Color', [0.75 0.75 0.75], 'MarkerSize', 4);
plot(model_ids, median(bh_nrmse, 1), '-sk', 'LineWidth', 2.5, 'MarkerFaceColor', 'k');
yline(1.0, 'r--', 'Zero prediction', 'LineWidth', 1.2, 'FontSize', 10);
title('A) Breath-Holding Prediction Error (nRMSE)', 'FontWeight', 'bold', 'FontSize', 11);
xlabel('Model', 'FontSize', 10); ylabel('nRMSE (lower is better)', 'FontSize', 10);
xticks(model_ids); xticklabels({'M0','M1','M2','M3','M4','M5','M6'}); grid on;
ylim([0.55 1.65]);

% B) C3 HRF RMSE
subplot(2, 2, 2); hold on;
plot(model_ids, c3_rmse', '-o', 'Color', [0.75 0.75 0.75], 'MarkerSize', 4);
plot(model_ids, median(c3_rmse, 1), '-sk', 'LineWidth', 2.5, 'MarkerFaceColor', 'k');
title('B) Motor Cortex C3 HRF Error (\muM)', 'FontWeight', 'bold', 'FontSize', 11);
xlabel('Model', 'FontSize', 10); ylabel('HRF RMSE [\muM]', 'FontSize', 10);
xticks(model_ids); xticklabels({'M0','M1','M2','M3','M4','M5','M6'}); grid on;

% C) Spatial Map Correlation
subplot(2, 2, 3); hold on;
plot(model_ids, spatial_r', '-o', 'Color', [0.75 0.75 0.75], 'MarkerSize', 4);
plot(model_ids, median(spatial_r, 1), '-sk', 'LineWidth', 2.5, 'MarkerFaceColor', 'k');
yline(0, 'k:');
title('C) Spatial Map Fidelity (Pearson r)', 'FontWeight', 'bold', 'FontSize', 11);
xlabel('Model', 'FontSize', 10); ylabel('Pearson r', 'FontSize', 10);
xticks(model_ids); xticklabels({'M0','M1','M2','M3','M4','M5','M6'}); grid on;

% D) 1-SE Model Selection Counts
subplot(2, 2, 4);
bar(model_ids, selection_counts, 'FaceColor', [0.2 0.4 0.6]);
title('D) 1-SE Selected Models', 'FontWeight', 'bold', 'FontSize', 11);
xlabel('Model', 'FontSize', 10); ylabel('Subject count', 'FontSize', 10);
xticks(model_ids); xticklabels({'M0','M1','M2','M3','M4','M5','M6'}); grid on;
ylim([0 max(selection_counts) + 0.5]);

% Save figure
fig_path = fullfile(output_dir, 'Figure_Group_Analysis.png');
try
    exportgraphics(fig, fig_path, 'Resolution', 300);
catch
    saveas(fig, fig_path);
end

fprintf('\nAnalysis complete. Outputs saved to: %s\n', output_dir);
group_results = struct('Performance', table_perf, 'Tests', table_tests, 'Selection', table_selection);
end

function M = extract_matrix(long_table, subjects, models, var_name)
M = nan(numel(subjects), numel(models));
for s = 1:numel(subjects)
    for m = 1:numel(models)
        row = long_table.Subject == subjects(s) & long_table.ModelID == models(m);
        M(s, m) = double(long_table.(var_name)(row));
    end
end
end
