function ll = getLogLikelihood(str)
% Reads loglikelihood from HTK output
A = strread(str, '%s', 'delimiter', sprintf('\n'));
ll = sscanf(A{end-1},'%s%s%s%s%s%s%s%s%s%f');
ll = ll(end);
