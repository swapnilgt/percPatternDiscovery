% This is a script file to run the percussion pattern transcription
% experiments. The experiments use the HMM toolkit (HTK), which can be
% obtained from http://htk.eng.cam.ac.uk/ The script assumes a working copy
% of HTK is installed. 
% 
% The experiments do a leave one out cross validation using HMM models obtained 
% from isolated HMM initialization for each syllable, using time aligned
% transcriptions. 
% 
% The experiments are optimized to run percussion transcription on tabla
% solo recordings on the Arvind Mulgaonkar Tabla Solo (MTS) dataset 
% More details on the dataset can be obtained from: 
% http://compmusic.upf.edu/tabla-solo-dataset
% 
% These experiments were described in the paper: 
% S. Gupta, A. Srinivasamurthy, M. Kumar, H. Murthy, X. Serra, Discovery of 
% Syllabic Percussion Patterns in Tabla Solo Recordings, Proceedings of the 
% 16th International Society for Music Information Retrieval Conference 
% (ISMIR 2015) (pp. 385-391), 2015, Malaga, Spain.
% See http://compmusic.upf.edu/ismir-2015-tabla
% 
% The experiments assume that the data has be obtained separately and the 
% following data pre-processing steps for the data are already done. <root>
% refers to the percPatternDiscovery folder
% 1. Copy the contents of the setup folder containing some setup related 
%    information in the dataset into <root>/transcriptions/setup/
% 2. Extract the features from the audio files using HCopy. e.g. you can
%    use HCopy_0_D_A.config file to generate MFCC_0_D_A features. 
% 
% The data is now prepared. We then initialize the files needed by HTK
% in the <root>/transcriptions/setup/ folder. The following files are
% already provided with the dataset, but if not, generate them. 
% 3. Generate a list of files or edit the paths to the feature files 
%    listFeatFiles_mfcc_0_d_a.txt accordingly. Use FULL paths. 
% 4. Create the dictionary files (dictionary.txt, dictionaryExt.txt) and
%    the HMM list file (hmmListExt). Make sure dictionary is alphabetical. 
% 5. Create a grammar file called grammar.txt
% 6. Create a prototype HMM definition file and call it 'proto_<featureType>'
% 7. Create the HCompV config file
% 8. If a HMM topology update is needed, then store the mixing commands in 
%    mixmcds.hhed
% 
% All set now!
% 
% This script does the following steps: 
% 1. Generates a dictionary and grammar network, along with a bigram language
%    model
% 2. Does a isolated start of each syllable HMM using HInit and the time 
%    aligned transcriptions
% 3. Using the initialized HMMs, does an embedded reestimation, along with 
%    updates to HMM shapes using HHed if needed
% 5. Testing using HVite and HResults (Bothy training and test data 
%    performance is measured)
% 6. Prepare a log file summarising actions and results
% 
% The script does a leave one out cross validation (CV), with many (user set)
% experiments per validation fold. Stores all the results. 
clear
clc
close all;
%% Add paths if needed
NumTrainIter = 10;  % Training iterations 
NumMixIt = 2;       % Mix iterations using HHed
NumExptPerFold = 1; % Experiments per CV fold
%% Initialize the filenames
basepath = './setup/'; % Empty string if this m-file is the current working directory
featExt = 'mfcc_0_d_a';   % feature
exptPath = ['../results' filesep featExt filesep]; % All results stored here
dataPath = '../data/'; % The data is assumed to be stored here
labTimedPath = [dataPath 'lblTimedLang' filesep]; % Label files in data
labPath = [dataPath 'lblLang' filesep]; 
if ~isdir(exptPath)
    mkdir(exptPath);        % Create the result folder
else
    rmdir(exptPath,'s');    % Remove all old files and start afresh
    mkdir(exptPath);
end
% Syllable list
hmmlist = [basepath, 'hmmListExt'];
fp = fopen(hmmlist,'rt');
syl = textscan(fp,'%s\n');
syl = syl{1};   
fclose(fp);
% HMM prototype
proto = [basepath, 'proto_' featExt];
% Flat start parameters
compvconfig = [basepath, 'HCompV.config'];
% Mixing parameters
mixcmds = [basepath, 'mixcmds.hhed'];
% Syllable alphabet
alphabetFile = [basepath, 'alphabet.txt'];
alphabetExtFile = [basepath, 'alphabetExt.txt'];
% Dictionaries
dictFile = [basepath, 'dictionary.txt'];
dictExtFile = [basepath, 'dictionaryExt.txt'];
dictOutFile = [exptPath, 'dictionary.dic'];
% Log files
logDumpFile = [exptPath, 'simulation.log'];
% Feature files
listFeatFiles = [basepath 'listFeatFiles_' featExt '.txt'];
copyfile(listFeatFiles,exptPath);   % Keep a copy of which files were used
% Grammar
gramFile = [exptPath, 'grammar.net'];
% Language model
biGramFile = [exptPath 'biGramModel.lm'];
% Temporary HMM files storage
tempHMMdir = [exptPath, 'allHMMsTemp/'];
if ~isdir(tempHMMdir)
    mkdir(tempHMMdir) 
end
% HMMs
initHMMfull = [exptPath, 'initAllHMM.hmm'];
opHMMfull = [exptPath, 'trainedHMM.hmm'];
mlFile = [labPath 'master.lab'];
% Housekeeping
tempTrainFile = [basepath 'tempTrainList.txt'];
tempScratchFile = [basepath 'tempScratch.txt'];
magicstr = '@@@@@';
msg = 'Start';      % Some message to be written to log
% Read the filelist of features
ffid = fopen(listFeatFiles,'rt');
temp = textscan(ffid,'%s\n');
featFiles = temp{1};
clear temp;
fclose(ffid);
% End of definitions
%% Begin Experiments
% Open log file
logFile = fopen(logDumpFile,'a+t');
if logFile == 0
    error('Cannot open log file');
end
fprintf(logFile,'# Starting experiment...\n');
%% Create dictionary and grammar
fprintf(logFile,'# Creating dictionary...\n');
ipstr = ['HDMan ', dictOutFile, ' ' dictExtFile];
ExecOneCommand(logFile,ipstr,magicstr);
fprintf(logFile,'# Generating bigram Language model\n');
ipstr = strcat({'HLStats -o -b '}, biGramFile, {' '}, alphabetFile, {' '}, mlFile);
ExecOneCommand(logFile,ipstr{1},magicstr);
fprintf(logFile,'# Creating Word network with Language model...\n');
ipstr = strcat({'HBuild -n '}, biGramFile, {' '}, alphabetExtFile, {' '}, gramFile);
ExecOneCommand(logFile,ipstr{1},magicstr);
ipstr =[];
%% Main loop
fprintf(logFile,'# Main loop starts here...\n');
tic
for k = 1:length(featFiles)     % Loop for Leave-One-Out CV
    trainFiles = featFiles;
    trainFiles(k) = [];
    testFile = featFiles{k};
    fTemp = fopen(tempTrainFile,'wt');
    for t = 1:length(trainFiles)
        fprintf(fTemp,'%s\n',trainFiles{t});
    end
    fclose(fTemp);
    ttmp = strfind(testFile,filesep);
    ttmp = ttmp(end);
    tfname = testFile(ttmp+1:end-4);    
    logLikFile = [];
    fprintf(logFile,'#### Test File: %s...\n',testFile);
    fprintf('Processing %s, %d/%d\n',testFile,k,length(featFiles));
    perf(k).name = tfname;
    for n = 1:NumExptPerFold    % Number of expts per fold
        fprintf(logFile,'# Expt. No. %d/%d...\n',n,NumExptPerFold);
        fprintf('# Expt. No. %d/%d...\n',n,NumExptPerFold);
        fprintf(logFile,'# Isolated Unit Training start...\n');
        fprintf('# Isolated Unit Training start...\n');
        for s = 1:length(syl)
            % Hinit the syllable
            ipstr = ['HInit -M ', tempHMMdir, ' -o ', syl{s}, ' -l ', syl{s},...
                ' -L ', labTimedPath, ' -S ', tempTrainFile, ' ', proto];
            try
                ExecOneCommand(logFile,ipstr,magicstr);
            catch
                % If it cannot run, just do a flat HCompV start 
                ipstr = ['HCompV -m -o ', syl{s}, ' -S ', tempTrainFile, ' ', proto];
                ExecOneCommand(logFile,ipstr,magicstr);
                movefile(syl{s}, [tempHMMdir syl{s}]);
                fprintf(logFile,'Could not find enough examples for %s, using flat start instead\n', syl{s});
                fprintf('Could not find enough examples for %s, using flat start instead\n', syl{s});
            end
            % Split the HMM
            tempFile = fopen(tempScratchFile,'wt');
            fprintf(tempFile, '%s\n', syl{s});
            fclose(tempFile);
            ipstr = ['HHEd -d ', tempHMMdir, ' -M ', tempHMMdir, ...
                ' ', mixcmds, ' ', tempScratchFile];
            ExecOneCommand(logFile,ipstr,magicstr);
            delete(tempScratchFile);
            % Now train with HRest
            ipstr = ['HRest -T 1 -M ', tempHMMdir, ' -l ', syl{s}, ...
                ' -L ', labTimedPath, ' -S ', tempTrainFile, ' ', tempHMMdir, syl{s}];
            try
                ExecOneCommand(logFile,ipstr,magicstr);
            catch
                fprintf(logFile,'Could not find enough examples for %s, using flat start instead\n', syl{s});
                fprintf('Could not find enough examples for %s, using flat start instead\n', syl{s});
            end
            fprintf('Training syllable %d/%d...\n',s,length(syl));
        end
        % Use HHEd to put all HMMs together
        tempFile = fopen(tempScratchFile,'wt'); % Dummy script
        fclose(tempFile);
        ipstr = ['HHEd -d ', tempHMMdir, ' -M ', exptPath, ...
            ' -w ', initHMMfull, ' ', tempScratchFile, ' ', hmmlist];
        ExecOneCommand(logFile,ipstr,magicstr);
        % Cleanup
        delete(tempScratchFile);
        rmdir(tempHMMdir, 's');
        mkdir(tempHMMdir);
        copyfile(initHMMfull, opHMMfull);   % backup of init HMM
        fprintf(logFile,'# Embedded reestimation training...\n');
        fprintf('Embedded reestimation training...\n');
        % Initialize Embedded Training
        logLik = zeros(NumMixIt,NumTrainIter);
        for iter1 = 1:NumMixIt
            fprintf(logFile,'# HERest Iterations...\n');
            for iter2 = 1:NumTrainIter
                ipstr = ['HERest -S ', tempTrainFile, ' -L ', labPath,...
                    ' -H ', opHMMfull, ' -T 3 ', hmmlist];
                [sts res] = system(ipstr);
                logLik(iter1, iter2) = getLogLikelihood(res);
                fprintf(logFile,'MixIt: %d, HERestIt: %d, Likelihood: %f\n',...
                    iter1-1, iter2, logLik(iter1, iter2));
                fprintf('MixIt: %d, HERestIt: %d, Likelihood: %f\n',...
                    iter1-1, iter2, logLik(iter1, iter2));
            end
            fprintf(logFile,'# HHEd mix up...\n');
            ipstr = ['HHEd -T 2 -H ', opHMMfull, ' ', mixcmds, ' ', hmmlist];
            if iter1 ~= NumMixIt
                ExecOneCommand(logFile,ipstr,magicstr);
            end
        end
        fprintf(logFile,'# Training done...\n');
        logLikFile = [logLikFile; logLik];
        % Training done... Now for testing
        % Save the trained HMM
        if ~isdir([exptPath 'models/' tfname])
            mkdir([exptPath 'models/' tfname]);
        end
        if ~isdir([exptPath 'transcriptions/' tfname])
            mkdir([exptPath 'transcriptions/' tfname]);
        end
        copyfile(opHMMfull, [exptPath 'models/' tfname filesep tfname ...
            '_fold-' num2str(n) '.hmm']);
        % Get results
        recfile = [exptPath 'transcriptions/' tfname filesep tfname ...
            '_expt-' num2str(n) '.lab'];
        ipstr = ['HVite -D -H ', opHMMfull, ' -C ', compvconfig, ...
            ' -w ', gramFile, ' -i ', recfile, ' ', dictOutFile, ' ', hmmlist, ' ', testFile];
        ExecOneCommand(logFile,ipstr,magicstr);
        ipstr = ['HResults -T 3 -t -L ', labPath, ' ', dictOutFile, ' ', recfile];
        fprintf(logFile,['\n>>', ipstr, '\n', magicstr '\n']);
        [sts res] = system(ipstr);
        % Get the values into structures apart from storing the files
        % Accuracy values
        [perf(k).Corr(n) perf(k).Accu(n) perf(k).H(n) perf(k).D(n)...
            perf(k).S(n) perf(k).I(n) perf(k).N(n)] = getAccuScores(res);
        % Labels
        dec(k).name = tfname;
        [dec(k).expt(n).strtTime dec(k).expt(n).endTime ...
           dec(k).expt(n).syl dec(k).expt(n).logLik] = getDecodedLabels(recfile);
        % Store in a csv file too, to be read by SV or the likes
        opFile = [exptPath 'transcriptions/' tfname filesep tfname ...
            '_expt-' num2str(n) '.csv'];
        fpo = fopen(opFile,'wt');
        for s = 1:length(dec(k).expt(n).syl)
            fprintf(fpo, '%10.6f\t%s\t%10.6f\n',dec(k).expt(n).strtTime(s), ...
               dec(k).expt(n).syl{s}, dec(k).expt(n).logLik(s));
        end
        fclose(fpo);
        for s = 1:length(dec(k).expt(n).syl)
            dec(k).expt(n).sylID(s) = find(ismember(syl, dec(k).expt(n).syl{s}));
        end
        % Sanity check
        if (perf(k).Corr(n) < 0 | perf(k).Accu(n) < 0 |...
                perf(k).H(n) < 0 | perf(k).D(n) < 0 | perf(k).S(n) < 0 |...
                perf(k).I(n) < 0 | perf(k).N(n) < 0)
            disp('Negative accuracy!!!');
        end
        % 
        res(res == '%') = '@';
        fprintf(logFile, res);
        resFile = fopen([recfile(1:end-4) '.result'],'wt');
        fprintf(resFile, res);
        fclose(resFile);
        fprintf(logFile,'%s\n', magicstr);
        clear res
        
        % Compute performance on training data
        fprintf('Computing Training accuracy...\n');
        if ~isdir([exptPath 'transcriptionsTrain/fold' num2str(k)])
            mkdir([exptPath 'transcriptionsTrain/fold' num2str(k)]);
        end
        for tt = 1:length(trainFiles)
            ttmp = strfind(trainFiles{tt},filesep);
            ttmp = ttmp(end);
            trainName = trainFiles{tt}(ttmp+1:end-4);    
            trainperf(k).exp(tt).name = trainName;
            recfile = [exptPath 'transcriptionsTrain/fold' num2str(k) filesep trainName ...
                '_expt-' num2str(n) '.lab'];
            ipstr = ['HVite -D -H ', opHMMfull, ' -C ', compvconfig, ...
                ' -w ', gramFile, ' -i ', recfile, ' ', dictOutFile, ' ', hmmlist, ' ', trainFiles{tt}];
            ExecOneCommand(logFile,ipstr,magicstr);
            ipstr = ['HResults -T 3 -t -L ', labPath, ' ', dictOutFile, ' ', recfile];
            fprintf(logFile,['\n>>', ipstr, '\n', magicstr '\n']);
            [sts res] = system(ipstr);
            % Get the values into structures apart from storing the files
            % Accuracy values
            [trainperf(k).exp(tt).Corr(n) trainperf(k).exp(tt).Accu(n) trainperf(k).exp(tt).H(n) trainperf(k).exp(tt).D(n)...
                trainperf(k).exp(tt).S(n) trainperf(k).exp(tt).I(n) trainperf(k).exp(tt).N(n)] = getAccuScores(res);
            % Labels
            traindec(k).exp(tt).name = trainName;
            [traindec(k).exp(tt).expt(n).strtTime traindec(k).exp(tt).expt(n).endTime ...
               traindec(k).exp(tt).expt(n).syl traindec(k).exp(tt).expt(n).logLik] = getDecodedLabels(recfile);
            opFile = [exptPath 'transcriptionsTrain/fold' num2str(k) filesep trainName ...
                     '_expt-' num2str(n) '.csv'];
            fpo = fopen(opFile,'wt');
            for s = 1:length(dec(k).expt(n).syl)
                fprintf(fpo, '%10.6f\t%s\t%10.6f\n',dec(k).expt(n).strtTime(s), ...
                dec(k).expt(n).syl{s}, dec(k).expt(n).logLik(s));
            end
            fclose(fpo);
            for s = 1:length(traindec(k).exp(tt).expt(n).syl)
                traindec(k).exp(tt).expt(n).sylID(s) = find(ismember(syl, traindec(k).exp(tt).expt(n).syl{s}));
            end
            res(res == '%') = '@';
            resFile = fopen([recfile(1:end-4) '.result'],'wt');
            fprintf(resFile, res);
            fclose(resFile);
            clear res
        end
        fprintf(logFile,'==============Finished Processing========\n');
        delete(opHMMfull);
    end
    perf(k)
    if ~isdir([exptPath 'log/'])
        mkdir([exptPath 'log/']);
    end
    dlmwrite([exptPath 'log/' tfname '.trainlog'],logLikFile);
    delete(tempTrainFile);
    fprintf('Processed fold %d/%d. Average time per fold %f seconds...\n',k,length(featFiles), toc/k);
end
save([exptPath 'results.mat'], 'perf', 'trainperf', 'NumTrainIter', 'NumMixIt', 'NumExptPerFold', 'traindec', 'dec')
fclose(logFile);
rmdir(tempHMMdir, 's');
for k = 1:length(trainperf)
    mnoda_corr(k) = mean([trainperf(k).exp.Corr]);
    mnoda_accu(k) = mean([trainperf(k).exp.Accu]);
end
allResults = [mean(mnoda_corr) mean(mnoda_accu) mean([perf.Corr]) mean([perf.Accu])];
dlmwrite([exptPath 'resultsSummary.txt'],allResults,'precision','%.2f');