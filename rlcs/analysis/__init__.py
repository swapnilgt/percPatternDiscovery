import sys
import os
sys.path.append('../utils')
sys.path.append('../src')
import utils as ut
from src import impl as rlcs
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
eps = np.finfo(np.float).eps 
'''
This module has the code to analyze the data obtaied fro the LCS
'''

# TODO: Remove this
def groupPatterns(patternDic):
    '''
    This function takes in a dictionary with the format:
    {<fileName1>:[<pattern1>, <pattern2>, ...], <fileName2>:[<pattern3>, <pattern4>,...], ...}
    and returns the patterns by grouping them in the form:
    {<pattern1>:<count1>, <pattern2>:<count2>, ...}
    '''
    patternCount = {}

    for key, patternList in patternDic.iteritems():
        for pattern in patternList:
            if pattern in patternCount.keys():
                patternCount[pattern] += 1
            else:
                patternCount[pattern] = 1
    return patternCount

#TODO: Remove this
def getAccuracy(patternInGT, matches):
    '''
    Takes in the the rlcs output on the grond truth <patternInGT> and in the test phrase <matches>.
    Return the precision, recall and f-measure  the each values
    '''
    matchSet = set()
    gtSet = set()
    # Get all the positions in matches and the ground truth in as set
    # We might have multiple subsequences starting at the same location
    # We take the one which has largest length
    # If the length of two patterns starting at the same place is same, we take the one which
    # has the maximum number of matched 
    for tup in matches:
        matchSet.add(tup[0])
    for tup in patternInGT:
        gtSet.add(tup[0])

    numOrig = len(gtSet)
    numTrans = len(matchPosSet)
    diff = numOrig - numTrans

    print 'The poistions in the transcribed are:' + str(numTrans)
    print 'The poistions in the grount truth are:' + str(numOrig)
    print 'The difference is:' + str(diff)
    return (diff, numOrig, numTrans)

def getClosestPairs(transMatches, scoreMatches, origOnsets, transOnsets, overlapTres):
    '''
    This method takes the transMatches and scoreMatches which are the list of the patterns found in
    the transciption and the ground truth score in the format:
    [(<position1>,<pattern1>), (<position2>, <pattern2>), ...]
    It also inputs the onset time for both ground truth (origOnsets) and the transcription (transOnsets)
    It returns a list of the tuples in which first element is a pattern from the transcription and
    the second element is the pattern from the ground truth score. Both the tuples are of the form:
    (<position3>, <pattern3>)

    Also, returns the transcriptions in the ground truth which had no matches associated to them in format
    same as the above.

    :Note:  
    
    1) Here we assume that the position mentioned in the tuples of transMatcehs and scoreMatches start from 1
    and the onsets position in the onset arrays start from 0 (handle this)

    '''
    # The return object ...
    closestPairs = []

    # The array that stores that closest match positions ...
    bestOverlapIndices = set()

    # The array that stores the matches in the ground score that do not have any match
    noMatchScore = []

    # Array to store the lengths of the retrieved patterns which conform to the scoreTres

    indexSet = set()

    # Iterating ...
    for trans in transMatches:
        bestOverlap = 0.0

        # the start position of the transcription
        sTrans = trans[0] - 1
        
        bestIndex = None # In case there is no match.. we check later if the bestIndex is None

        # Getting the number of syllables in transcription
        ptrTrans = trans[1].strip() # Have spaces at the end
        ptrTransLen = len(ptrTrans.split(' '))
        for index, orig in enumerate(scoreMatches):
            # the start position of the ground truth score
            #print 'The index is:' + str(index)
            #print 'The orig is:' + str(orig)
            sOrig = orig[0] - 1
            
            ptrOrigLen = len(orig[1])


            overlapTime = ut.getOverlap((origOnsets[sOrig], origOnsets[sOrig + ptrOrigLen - 1]), (transOnsets[sTrans], transOnsets[sTrans + ptrTransLen - 1]))
            percOverlap = overlapTime * 100 / (origOnsets[sOrig + ptrOrigLen - 1] -  origOnsets[sOrig])
            
            # checking if the new overlap percentage is greater than the older one
            if percOverlap > bestOverlap:
                bestOverlap = percOverlap
                bestIndex = index
        
        closestPairs.append((trans, bestOverlap))
        if bestOverlap >= overlapTres: # Appending only if the overlap is more than the threshold
            bestOverlapIndices.add(bestIndex)
    
     
    for index, val in enumerate(scoreMatches):
        if index not in bestOverlapIndices:
            noMatchScore.append(val)


    return (closestPairs, noMatchScore)

def getRelevantTrans(overlaps, tres = 0.0):
    '''
    Takes in the overlaps for each of the transcripted substring found and returns the number which have overlap above the tres
    Also, returns a list which shows the length of the retrieved patterns in the transcribed data
    '''
    total = 0
    ptrLen = []
    for tup in overlaps:
        if tup[1] != None and tup[1] >= tres:
            total += 1
            ptrLen.append(len(tup[0][1].strip().split(' ')))
    return total, ptrLen

def getFalseNegativesInTranscription(trans, notFound, origOnsets, transOnsets):
    '''
    This function takes in the transcribed syllables, the one not found, the onset positions and returns
    the list of the tuples where each tuple is of the format (<position>, <pattern>)
    '''

    falseNegs = []

    for tup in notFound:
        # The start position of the ground truth pattern..
        sOrig = tup[0] - 1
            
        # Getting the number of syllables in the original score pattern 
        #ptrOrig = tup[1].strip() # Have spaces at the end
        #ptrOrigLen = len(ptrOrig.split(' '))
        ptrOrigLen = len(tup[1])

        # Getting the start and end time of the transcription
        sTime = origOnsets[sOrig] 
        eTime = origOnsets[sOrig + ptrOrigLen - 1]

        if sTime >= eTime:
            print 'analysis.getFalseNegativesInTranscription:: sTime greater or equal than the eTime:' + str(sTime) + ' and ' + str(eTime)
            sys.exit()

        # Getting the closest indices for the start and end time in the 
        sIndex = ut.findClosestIndex(sTime, transOnsets, 0, len(transOnsets) - 1)
        eIndex = ut.findClosestIndex(eTime, transOnsets, 0, len(transOnsets) - 1)

        falseNegs.append((sIndex + 1, trans[sIndex : eIndex + 1]))

    return falseNegs

def getAnalysisBeta(resDir, p = 0.875, scoreTres = 0.75, betaMin = 0.01, betaMax = 0.99, betaStep = 0.05):
    """
    Gives the analysis for for a range of beta with other parameters fixed
    """
    result = []
    if not os.path.isdir(resDir):
        print 'The directory mentioned does not exist:' + resDir
        sys.exit()
    pDir = os.path.join(resDir, 'p_'+ str(p))
    if not os.path.isdir(pDir):
        print 'The directory mentioned does not exist:' + pDir
        sys.exit()

    beta = betaMin
    while beta <= betaMax:
        bDir = os.path.join(pDir, 'beta_' + str(beta))
        if not os.path.isdir(bDir):
            print 'The directory mentioned does not exist:' + bDir
            sys.exit()
        data = ut.readPickle(os.path.join(bDir, 'result.pkl'))
        found = False
        for tup in data:
            if abs(tup[0] - scoreTres) < eps:
                found = True
                print 'The value of scoreTres:' + str(tup[0])
                result.append((beta,tup[1]))
                break
        if found is not True:
            print 'Could not find the value for iteration with beta=' + str(beta)
            sys.exit()
        beta += betaStep

    return result


########## New set of functions for analysis of the errors in the transcription ##########
def getIndicesInTrans(origOnset, transOnset, dictPtrIndices):
    """
    Inputs the onset locations of ground truth (origOnset), transcribed score (transOnset) and the start and end locations
    of the pattern in the ground truth and returns the list of indices for the closest onset positions in the transcription
    """
    result = {}
    retRes = []

    print 'The number of patterns in the GT are:' + str(len(dictPtrIndices))

    for i, ons in enumerate(transOnset):
        closestInOrig = ut.findClosestIndex(ons, origOnset, 0, len(origOnset)-1)
        for key, val in dictPtrIndices.iteritems():
            if closestInOrig in val:
                if key in result:
                    result[key].append(i)
                else:
                    result[key] = [i]
    for key, val in result.iteritems():
        if max(val) - min(val) + 1 != len(val):
            print 'Something wrong with the positions found'
            print 'key:' + str(key)
            print 'val:' + str(val)
        retRes.append(val)
    return retRes


def getPatternsInTrans(score, indicesList):
    """
    This function takes in the score which is a list of syllables and the indices which is a list of lists of
    the indices in these score. For, each list in the indices, it returns the sequence of syllables found with 
    the ones that correspond to exact locatinos in the ground truth marked with a '!'
    """
    result = []
    for indices in indicesList:
        #print indices
        seq = score[indices[0]:indices[len(indices)-1]+1]
        result.append(seq)

    return result

def populateDictPtrIndices(ptrStartIndices, l):
    """
    Takes in the starting points of the patterns and populate a dictionary of sets with all the indices of the pattern
    """
    res = {}

    for i in ptrStartIndices:
        s = set()
        for j in range(i, i+l):
            s.add(j)
        res[i] = s
    return res


def getTranscribedPatterns(ptr = ['TA', 'TA', 'KI', 'TA']):
    """
    Gets the transcribed sequences for the pattern (ptr) provided bsed on the onset information in the transcribed data.
    """
    result = []
    l = len(ptr)
    # Getting the masterdata
    config = ut.loadConfig('/home/swapnil/SMC/MasterThesis/gitPercPatterns/code/rlcs/config')
    transFolder = config['transFolder']
    #transFolder = '/home/swapnil/Desktop/temp'
    lblDir = config['lblDir']
    onsDir = config['onsDir']
    masterData = ut.getAllSylbData(tPath = transFolder, lblDir = lblDir, onsDir = onsDir)
    
    # Find the start and the end point of patters in
    for comp in masterData:
        compName = comp[2]
        print 'Working for composition:' + compName
        transScore = comp[0][0]
        transOnset = comp[0][1]
        origScore = comp[1][0]
        origOnset = comp[1][1]

        # Get the starting indices for the pattern in the composition comp
        ptrStartIndices = ut.getSubstringPos(origScore, ptr)

        # Get the dictionaries of set for the indices of patterns in the ground truth
        dictPtrIndices = populateDictPtrIndices(ptrStartIndices, l)

        # Get the closest set onsets for the pattern in the transcription
        ptrIndicesInTrans = getIndicesInTrans(origOnset, transOnset, dictPtrIndices)
        ptrnsInTrans = getPatternsInTrans(transScore, ptrIndicesInTrans)
        result.append((compName, ptrnsInTrans))

    return result

def getTranscribedPatternsStats(ptrErrors):
    """
    Returns the statistics of the pattern errors. Return in the form of number of patterns with different lengths
    """

    stats = {}
    statsArray = []

    total = 0

    for tup in ptrErrors:
        for ptr in tup[1]:
            statsArray.append(len(ptr))
            total += 1
            if len(ptr) in stats:
                stats[len(ptr)] += 1
            else:
                stats[len(ptr)] = 1

    print 'The total number of patterns are:' + str(total)
    return stats, statsArray


def getPtrStr(ptr):
    """
    Inputs a pattern in the form of list of syllables and returns the '!' separated string for it
    """

    res = ''
    for syl in ptr:
        res += (syl + '!')
    return res.strip('!')


def getPatternGroupsFromTups(transPtrs):
    """
    Groups the patterns in the transcription and returns count for each class along with the total number of patterns
    """
    res = {}
    total = 0

    for tup in transPtrs:
        for ptr in tup[1]:
            total += 1
            strPtr = getPtrStr(ptr)
            if strPtr in res:
                res[strPtr] += 1
            else:
                res[strPtr] = 1
    return res, total

def getPatternsInTransInGTPos(masterData, queryList):
    '''
    This function takes in the masterData and queryList and returns pattern wise list of patterns in transcribed data for 
    each composition for the positions of the where there is a pattern in the GT. This is to analyze the errors in the transcription
    '''
    res = []
    for query in queryList:
        qLen = len(query)
        qRes = []
        transQ = []
        for compData in masterData:
            uniqMatchesGT = rlcs.getGTPatterns(compData[1][0],query)
            #print 'In GT for composition ' + str(compData[2] + ':' + str(uniqMatchesGT))

            # getting the onset times of the pattern boundaries
            GTOnsets = compData[1][1] # array of onsets in the ground truth of the composition
            transOnsets = compData[0][1] # array of the onsets in the transcribed data of the composition
            for match in uniqMatchesGT:

                #print 'Working for:' + str(match)

                GTStartIndex = match[0] - 1 # start index of the pattern in the GT
                GTEndIndex = GTStartIndex + qLen - 1 # end index of the pattern in the GT

                #print 'Starting index in GT:' + str(GTStartIndex)
                #print 'Ending index in GT:' + str(GTEndIndex)

                transStartIndex = ut.findClosestIndex(GTOnsets[GTStartIndex], transOnsets, 0, len(transOnsets)-1)
                transEndIndex = ut.findClosestIndex(GTOnsets[GTEndIndex], transOnsets, 0, len(transOnsets)-1)

                #print 'Starting index in Trans:' + str(transStartIndex)
                #print 'Ending index in iTrans:' + str(transEndIndex)
                if compData[0][0][transStartIndex] == 'NA' and  compData[0][0][transStartIndex+1] == 'KI' and compData[0][0][transStartIndex+2] == 'TA' and compData[0][0][transStartIndex+3] == 'TA' and transEndIndex-transStartIndex+1 == 4:
                    print compData[2]
                qRes.append(transEndIndex - transStartIndex + 1)
                transQ.append(compData[0][0][transStartIndex:transEndIndex + 1])
        res.append((query, qRes, transQ))
    return res

def getPatternGroups(ptrs):
    '''
    Takes in a group of list of patterns and returns them after groupong them with count of each
    '''
    res = {}

    for ptr in ptrs:
        strPtr = getPtrStr(ptr)
        if strPtr in res:
            res[strPtr] += 1
        else:
            res[strPtr] = 1

    # adding the length of the pattern to the payload
    for key, val in res.iteritems():
        res[key] = (val, len(key.split('!')))

    return res

def plotFramesPerSyl(bagOfFeatsFile='/home/swapnil/SMC/MasterThesis/gitPercPatterns/code/sylbSimilarity/bagOfFeats.pkl'):
    '''
    Takes in the file path for the bag of feats file and plots a bar graph for distribution of the frames for all the syllables
    '''

    fo = open(bagOfFeatsFile, 'rb')
    feats = pkl.load(fo)
    n = len(feats)
    x = np.arange(n)
    syls = []
    frms = []
    for key, val in feats.iteritems():
        syls.append(key)
        frms.append(len(feats[key]))

    fig, ax = plt.subplots()
    rects = ax.bar(x, frms, 0.35, color='b')
    ax.set_xticks(x + 0.35/2)
    ax.set_xticklabels(syls, fontsize='large')

    ax.set_title('Frames for each bol', fontsize='large')
    ax.set_ylabel('Number of Frames', fontsize='large')

    # adding the count on top of the bar
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, ' %d'%int(height), ha='center', va='bottom', fontsize='large')
    plt.show()
