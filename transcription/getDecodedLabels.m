function [strtTime endTime label logLik] = getDecodedLabels(recfile)
fp = fopen(recfile,'rt');
A = textscan(fp,'%s','delimiter','\n');
fclose(fp);
A = A{1};
A(1:2) = [];
A(end) = [];
for k = 1:length(A)
    vals = textscan(A{k}, '%f %f %s %f\n');
    strtTime(k) = vals{1}/1e7;
    endTime(k) = vals{2}/1e7; 
    label(k) = vals{3}; 
    logLik(k) = vals{4};
    clear vals
end