% Runs a single HTK command string
% Done only to avoid code repetition
% To be called only from the parent m-file
% Parent: exptWrapper.m
function sts = ExecOneCommand(logFile,ipstr,magicstr)
fprintf(logFile,['\n>>', ipstr, '\n', magicstr '\n']);
[sts res] = system(ipstr);
fprintf(logFile, res);
fprintf(logFile,'%% %s\n', magicstr);
if sts ~= 0
    error('Error: Cannot Continue; Please see the log file for more details');
end
