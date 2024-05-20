%% “Real Exchange Rates and Sectoral Productivity in the Eurozone”
% by Berka, Devereux and Engel (2018), American Economic Review
% Master program for model simulations and model regressions
% It runs 3 versions of the model (different price stickiness), and saves
% results. Then, it produces medians, and confidence intervals for the
% model-based regressions in the article. 
% See 1readme.pdf in this folder for details.

clear; clc;
kkk_in = 999; % price stickiness parameter (same in both sectors)
BDE2017_simul
d1_table = array2table(d1);  % Convert the data to a table
writetable(d1_table, 'BDE_program_output.xlsx', 'Sheet', 'flexible', 'Range', 'A1');
writecell(d1, 'BDE_program_output.xlsx', 'Sheet', 'flexible', 'Range', 'A1');
%xlswrite('BDE_program_output.xls', d1, 'flexible', 'A1');

clear
kkk_in = 0.1; % price stickiness parameter (same in both sectors)
BDE2017_simul
d1_table = array2table(d1);  % Convert the data to a table
writetable(d1_table, 'BDE_program_output.xlsx', 'Sheet', 'sticky_1', 'Range', 'A1');
writecell(d1, 'BDE_program_output.xlsx', 'Sheet', 'sticky_1', 'Range', 'A1');
%xlswrite('BDE_program_output.xls', d1, 'sticky_1', 'A1');

clear
kkk_in = 0.2; % price stickiness parameter (same in both sectors)
BDE2017_simul
d1_table = array2table(d1);  % Convert the data to a table
writetable(d1_table, 'BDE_program_output.xlsx', 'Sheet', 'sticky_2', 'Range', 'A1');
writecell(d1, 'BDE_program_output.xlsx', 'Sheet', 'sticky_2', 'Range', 'A1');
%xlswrite('BDE_program_output.xls', d1, 'sticky_2', 'A1');

%% adjust signs to match variable definion in article tables, calculate medians and CIs
clear

% index of slope parameter (to convert to signs that match tables in the article)
a=[-1 -1 1 1 1 1 1 1 1 1 1 1 1 -1 -1 -1 -1 -1 -1 1 -1 -1 1 -1 -1 -1 -1 -1 -1];

d_flexible = readmatrix('BDE_program_output.xlsx','Sheet','flexible', 'Range', 'A2:AC502');
d_sticky1 = readmatrix('BDE_program_output.xlsx', 'Sheet', 'sticky_1', 'Range', 'A2:AC502');
d_sticky2 = readmatrix('BDE_program_output.xlsx', 'Sheet','sticky_2', 'Range', 'A2:AC502');

d_flexible = a.*d_flexible;
d_sticky1 = a.*d_sticky1;
d_sticky2 = a.*d_sticky2;

% Writing modified data back to the Excel file if they are cell arrays
writematrix(d_flexible, 'BDE_program_output.xlsx', 'Sheet', 'flexible', 'Range', 'A2');
writematrix(d_sticky1, 'BDE_program_output.xlsx', 'Sheet', 'sticky_1', 'Range', 'A2');
writematrix(d_sticky2, 'BDE_program_output.xlsx', 'Sheet', 'sticky_2', 'Range', 'A2');

% medians
coef_flexible = median(d_flexible);
coef_sticky1 = median(d_sticky1);
coef_sticky2 = median(d_sticky2);

% precentiles
p10_flexible = prctile(d_flexible,10,1);
p90_flexible = prctile(d_flexible,90,1);
p10_sticky1 = prctile(d_sticky1,10,1);
p90_sticky1 = prctile(d_sticky1,90,1);
p10_sticky2 = prctile(d_sticky2,10,1);
p90_sticky2 = prctile(d_sticky2,90,1);

% Writing the medians and percentiles to the Excel file using writematrix for numeric data
writematrix([coef_flexible; p10_flexible; p90_flexible], 'BDE_program_output.xlsx', 'Sheet', 'flexible', 'Range', 'A503');
writematrix([coef_sticky1; p10_sticky1; p90_sticky1], 'BDE_program_output.xlsx', 'Sheet', 'sticky_1', 'Range', 'A503');
writematrix([coef_sticky2; p10_sticky2; p90_sticky2], 'BDE_program_output.xlsx', 'Sheet', 'sticky_2', 'Range', 'A503');



