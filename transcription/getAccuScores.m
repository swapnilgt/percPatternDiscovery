function [Corr Accu H D S I N] = getAccuScores(str)
% Extracts performance measures from HTK output 
try
    A = textscan(str, '%s', 'delimiter', sprintf('\n'));    % Split by lines
    A = A{1};
    vals = textscan(A{end-1}, '%s', 'delimiter', sprintf(' ')); % Get the result line
    vals = vals{1};
    % Corr
    st = find(vals{2} == '=') + 1;
    en = find(vals{2} == ',') - 1;
    Corr = str2double(vals{2}(st:en));
    % Accu 
    st = find(vals{3} == '=') + 1;
    Accu = str2double(vals{3}(st:end));
    % H
    st = find(vals{4} == '=') + 1;
    en = find(vals{4} == ',') - 1;
    H = str2double(vals{4}(st:en));
    % D
    st = find(vals{5} == '=') + 1;
    en = find(vals{5} == ',') - 1;
    D = str2double(vals{5}(st:en));
    % S
    st = find(vals{6} == '=') + 1;
    en = find(vals{6} == ',') - 1;
    S = str2double(vals{6}(st:en));
    % I
    st = find(vals{7} == '=') + 1;
    en = find(vals{7} == ',') - 1;
    I = str2double(vals{7}(st:en));
    % N
    st = find(vals{8} == '=') + 1;
    en = find(vals{8} == ']') - 1;
    N = str2double(vals{8}(st:en));
catch 
    disp('Could not get accuracy!');
    Corr = -1;
    Accu = -1; 
    H = -1;
    D = -1; 
    S = -1;
    I = -1;
    N = -1;
end
