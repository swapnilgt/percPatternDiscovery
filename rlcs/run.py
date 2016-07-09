import os
import sys
from src import impl as rlcs
import utils as ut
import analysis as anls
import matplotlib.pyplot as plt
import logging
import pickle as pkl
import time

config = ut.loadConfig('config')

sylbSimFolder=config['sylbSimFolder']
transFolder=config['transFolder']
lblDir=config['lblDir']
onsDir=config['onsDir']
resultDir=config['resultDir']
sylbListFile=config['sylbListFile']

print sylbListFile

queryList = [['DHE','RE','DHE','RE','KI','TA','TA','KI','NA','TA','TA','KI','TA','TA','KI','NA'],['TA','TA','KI','TA','TA','KI','TA','TA','KI','TA','TA','KI','TA','TA','KI','TA'], ['TA','KI','TA','TA','KI','TA','TA','KI'], ['TA','TA','KI','TA','TA','KI'], ['TA', 'TA','KI', 'TA'],['KI', 'TA', 'TA', 'KI'], ['TA','TA','KI','NA'], ['DHA','GE','TA','TA']]
queryLenCheck = [4,6,8,16]

for query in queryList:
    if len(query) not in queryLenCheck:
        print 'The query is not of correct length!!'
        sys.exit()

# Checking if we want to run for baseline or not.
baseline = False
if "baseline" in sys.argv:
    baseline = True



#similarityListFile = os.path.join(sylbSimFolder,'simMatList.txt')
#similarityList = [line.strip().split('.')[0] for line in open(similarityListFile)]
# For correctness check
#similarityList = ['TablaDB_3_kmeans_mahal'] # looks promising ....
#similarityList = ['KLMonteCarlo-5','KLMonteCarlo-6', 'KLMonteCarlo-7'] # looks promising ....
#similarityList = ['KLGaussApprox-3', 'KLGaussApprox-4', 'KLGaussApprox-5', 'KLGaussApprox-6', 'KLGaussApprox-7'] # looks promising ....
similarityList = ['binaryDistance'] # results without similarity
ignrSimList = ['TablaDB_10_GMM_euclidean']

simObject = None

masterData = ut.getAllSylbData(tPath = transFolder, lblDir = lblDir, onsDir = onsDir)

#simDict = ut.getSimilarityDict('/home/swapnil/SMC/MasterThesis/sylbSimilarity/TablaDBstrokes','/home/swapnil/SMC/MasterThesis/sylbSimilarity/results_mat/TablaDB_6_kmeans_euclidean.mat')

def getAccuracies(payload, tres = 70.0, fp = None):

    #fo.write('run.getAccuracies::')

    totalRelvRetrieved = 0
    totalRetrieved = 0
    totalRelevant = 0
    totalRelvRetrievedInGt = 0
    ptrInTransLen = [] # List for the length of all the patterns that are the candidate patterns 

    for inst in payload:
        retrieved = inst[0]
        relevant = inst[1]
        overlaps = inst[2]
        retrievedInGT = inst[3] # We have keep this because there are multiple patterns at the same location of the GT (for recall)

        if len(overlaps) != retrieved:
            print 'The length of the two are not equal..'
            print 'retrieved:' + str(retrieved)
            print 'len(overlaps):' + str(len(overlaps))
            sys.exit()
        
        # Getting the ones that are relevant based on the threshold
        relvInTrans, ptrLenInTransFoComp = anls.getRelevantTrans(overlaps, tres = tres)

        totalRetrieved += retrieved
        totalRelevant += relevant
        totalRelvRetrieved += relvInTrans
        totalRelvRetrievedInGt += retrievedInGT
        ptrInTransLen.extend(ptrLenInTransFoComp)

    fp.write('Total patterns retrieved in transcription:' + str(totalRetrieved) + '\n')
    fp.write('Total patterns retrieved in ground truth:' + str(totalRelevant) + '\n')
    fp.write('Total correct patterns retrieved :' + str(totalRelvRetrieved)+ '\n') 
    fp.write('Posiions of the GT at which the patterns were recovered :' + str(totalRelvRetrievedInGt)+ '\n') 


    # Return precision, recall
    if totalRetrieved is not 0:
        precision = (totalRelvRetrieved * 1.0/totalRetrieved) * 100
    else:
        precision = None
    if totalRelevant is not 0:
        recall = (totalRelvRetrievedInGt * 1.0/totalRelevant) * 100
    else:
        recall = None
    return (precision, recall, ptrInTransLen)

def getUniqMatchesForAllComp(query, betaDir, beta, p, formula, baseline = False):
    ''' This method takes in the masterData, query, compare, beta and p and returns the uniqMatchesList for the entire masterData'''
    if not os.path.exists(betaDir):
        print 'The base director for logging does not exist.. Exiting' + str(betaDir)
        sys.exit()

    dataPerComp = []
    for compData in masterData:
        compName = compData[2]
        fComp = open(os.path.join(betaDir,compName + '.log'), 'a')
        if baseline is False:
            uniqMatchesTrans = rlcs.getUniqMatches(compData[0][0], query, simObject, beta, p, fComp, formula)
        else:
            uniqMatchesTrans = rlcs.getBaselinePatterns(compData[0][0], query, fComp)
        uniqMatchesGT = rlcs.getGTPatterns(compData[1][0], query, fComp)

        dataPerComp.append((uniqMatchesTrans, uniqMatchesGT))
        fComp.close()
    return dataPerComp

def runOneIter(query, compMatchList, scoreTres, tresDir, overlapTres):

    if not os.path.exists(tresDir):
        print 'The base directory for logging does not exist.. Exiting' + str(tresDir)
        sys.exit()
    
    result = []

    # List for storing the length of the patterns in the patterns searched (including false positives) in the transcribed score
    ptrLenInTrans = []

    for compData, compMatches in zip(masterData, compMatchList):
        
        compName = compData[2]

        fComp = open(os.path.join(tresDir,compName + '.log'), 'a')

        transMatches = ut.getMatchesForTres(compMatches[0], scoreTres = scoreTres)

        # getting the pattern length list and adding it to ptrLenInTrans
        ptrLenInTrans.extend(ut.getPtrLenList([tup[1] for tup in transMatches]))

        GTMatches = compMatches[1]
        fComp.write('Running for the query:' + str(query) + '\n')
        fComp.write('The matches in transcription that have score above the threshold are:' + str(transMatches) + '\n')
        fComp.write('The matches in ground truth that have score above the threshold are:' + str(GTMatches) + '\n')

        # Getting the onsets ..
        origOnsets = compData[1][1]
        transOnsets = compData[0][1]
        #transOnsets = compData[1][1]

        # Find the closest match of the transcribed pattern based on the time overlap ...
        # TODO: Remove thses print statements later
        #print 'transMatches=' + str(transMatches)
        #print 'GTMatches=' + str(GTMatches)
        overlaps, notFound = anls.getClosestPairs(transMatches, GTMatches, origOnsets, transOnsets, overlapTres)
        
        # Get the length of the patterns in the overlaps

        fComp.write('The overlaps are:' + str(overlaps) + '\n')
        fComp.write('The patterns in ground truth that do not have any match:' + str(notFound) + '\n')
        
        # Getting the transcription for the positions where there is no hit by the RLCS algorithm..
        falseNegs = anls.getFalseNegativesInTranscription(compData[0][0], notFound, origOnsets, transOnsets)
        fComp.write('The patterns in transcription which are not discovered by RLCS:' + str(falseNegs) + '\n')
        
        fComp.write('The number of matches in transcription that have score above the threshold are:' + str(len(transMatches)) + '\n')
        fComp.write('The number matches in ground truth that have score above the threshold are:' + str(len(GTMatches)) + '\n')
        fComp.write('The number overlaps are:' + str(len(overlaps)) + '\n')
        fComp.write('The patterns in ground truth that do not have any match:' + str(len(notFound)) + '\n')
        fComp.write('The patterns in transcription which are not discovered by RLCS:' + str(len(falseNegs)) + '\n\n')
        if len(transMatches) != len(overlaps):
            print 'The size of the overlaps and the matches in the transcription are different.'
            sys.exit()
        result.append((len(transMatches), len(GTMatches), overlaps, len(GTMatches) - len(notFound)))
        fComp.close()
 
    fAll = open(os.path.join(tresDir,'master.log'), 'a')
    fAll.write('Running for the query:' + str(query) + '\n')
    precision, recall, ptrLenInTransOverlaps = getAccuracies(result, tres = overlapTres, fp = fAll)
    fAll.write('The precision is:' + str(precision) + '\n')
    fAll.write('The recall is:' + str(recall) + '\n\n')
    fAll.close()
    
    # Writing the result per query into the pkl file .. 
    ptrObj = (precision, recall, ptrLenInTransOverlaps, ptrLenInTrans) # added ptrLenInTrans to have array for the length of the patterns that have been retrieved
    fPtrName = os.path.join(tresDir,'!'.join(query)+ '.pkl')
    resFile = open(fPtrName, 'wb')
    pkl.dump(ptrObj, resFile)
    resFile.close()
    return result


def runWithScoreTres(uniqMatchListPerQuery, queryList, betaDir, overlapTres):
    scoreMinTres = 0.0
    scoreMaxTres = 1.0
    scoreStep = 0.05

    if not os.path.exists(betaDir):
        print 'The base directory for logging does not exist.. Exiting' + str(betaDir)
        sys.exit()
    
    result = []


    scoreTres = scoreMinTres

    while scoreTres <= scoreMaxTres:
        resPerQuery = []

        tresDir = os.path.join(betaDir,'scoreTres_' + str(scoreTres))
        os.makedirs(tresDir)

        for query, compMatchList in zip(queryList, uniqMatchListPerQuery):
            resPerComp = runOneIter(query, compMatchList, scoreTres, tresDir, overlapTres)
            resPerQuery.extend(resPerComp)
        
        fAll = open(os.path.join(tresDir,'master.log'), 'a')
        precision, recall, ptrLenInTrans = getAccuracies(resPerQuery, tres = overlapTres, fp = fAll)
        fAll.write('The precision is:' + str(precision) + '\n')
        fAll.write('The recall is:' + str(recall) + '\n')
        fAll.close()

        result.append((scoreTres,(precision, recall)))
        scoreTres += scoreStep

    # Dumping output to the pickle file
    resFileName = os.path.join(betaDir, 'result.pkl')
    resFile = open(resFileName, 'wb')
    pkl.dump(result, resFile)
    resFile.close()
    
    # Plotting a graph and saving it
    imgFile = os.path.join(betaDir, 'figure.png')
    #plotRes(resFileName, 'tres', imgFile)
    
    return result


def runWithBeta(queryList, pDir, betaMin, betaMax, betaStep, p, overlapTres, formula, baseline = False):
    if not os.path.exists(pDir):
        print 'The base directory for logging does not exist.. Exiting' + str(pDir)
        sys.exit()

    result = []
    
    beta = betaMin
    while(beta <= betaMax):
        print '###### Running iteration for a new beta = '+ str(beta)+' #######'
        uniqMatchListPerQuery = []
        
        # Appending the feature type to the subDir name.
        betaDir = 'beta_' + str(beta)
        betaDir = os.path.join(pDir, betaDir)
        os.makedirs(betaDir)

        for query in queryList:
            # Should get data from both the transcription and the ground truth...
            uniqMatchDataPerQuery = getUniqMatchesForAllComp(query, betaDir, beta, p, formula, baseline = baseline)
            uniqMatchListPerQuery.append(uniqMatchDataPerQuery)

        resScore = runWithScoreTres(uniqMatchListPerQuery, queryList, betaDir, overlapTres)
        result.append((beta, resScore))
        beta += betaStep

    # Dumping output to the pickle file
    resFileName = os.path.join(pDir, 'result.pkl')
    resFile = open(resFileName, 'wb')
    pkl.dump(result, resFile)
    resFile.close()

    return result


def runWithAllParams(queryList, logDir, betaMin = 0.01, betaMax = 0.99, betaStep = 0.05, pMax = 1.0, pStep = 0.05, overlapTres = 70.0, formula = None, baseline = False):

    result = []

    #if not os.path.exists(logDir):
        #print 'The directory for outputting results does not exist!! Exiting..'
        #sys.exit()

    timeStamp = time.strftime("%Y-%m-%d_%H-%M-%S",time.gmtime(time.time())) # get the current timestamp !
    baseDir = os.path.join(logDir, timeStamp)
    os.makedirs(baseDir) # Creating the directory with timestamp
    
    # Getting the minimum p based on the 
    minLenQ = 100
    for query in queryList:
        if len(query) < minLenQ:
            minLenQ = len(query)
    p = 1.1 / minLenQ
    
    if baseline is True:
        p = 0.0
    
    while p <= pMax:
        print '$$$$$ Running iteration for a new p = '+ str(p)+' $$$$$'
        pDir = os.path.join(baseDir, 'p_' + str(p))
        os.makedirs(pDir) # Creating the directory for the p value
        pResult = runWithBeta(queryList, pDir, betaMin = betaMin, betaMax = betaMax,\
                betaStep = betaStep, p = p, overlapTres = overlapTres, formula = formula, baseline = baseline)
        result.append((p, pResult))
        p += pStep


    # Dumping output to the pickle file
    resFileName = os.path.join(baseDir, 'result.pkl')
    resFile = open(resFileName, 'wb')
    pkl.dump(result, resFile)
    resFile.close()
    
    return result    


if __name__ == '__main__':
    
    for similarity in similarityList:
        if similarity in ignrSimList:
            print 'Ignoring similarity..Going to next similarity measure!!'
            continue
        simDict = ut.getSimilarityDict(sylbListFile, os.path.join(sylbSimFolder, similarity + '.mat'))
        #simDict = ut.readPickle(os.path.join('/home/swapnil/SMC/MasterThesis/gitPercPatterns/code/sylbSimilarity/sim', similarity + '.pkl'))
        simTresMax = 0.9
        simTresStep = 0.3
        simTres = 0.3
        
        # In case the baseline is true.
        if baseline is True:
            simTres = 0.9

        print 'Running for similarity:' + similarity 
        formula = 2
        while simTres <= simTresMax:
            resultSimDir = os.path.join(resultDir, 'formula' + str(formula), similarity, similarity + '_' + str(simTres))
            simObject = ut.Similarity(simDict, simTres) # second arguments is the threshold for the distance between the two sylbls.
            if baseline is False:
                runWithAllParams(queryList, resultSimDir, formula = formula, baseline = baseline)
            else:
                runWithAllParams(queryList, resultSimDir, formula = formula, betaMin = 0.0, betaMax = 0.0, pMax = 0.0, baseline = baseline)
            print resultSimDir
            print simObject.tres
            simTres += simTresStep
