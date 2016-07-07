import os
import sys
import pickle as pkl
import logging
import shutil
import scipy.io
import math
import matplotlib.pyplot as plt

'''
This module defines some of the util files that helps the RLCS to run based on our case
'''
def compareBols(a, b):
    isEqual = False
    dist = 1
    if a == b:
        isEqual = True
        dist = 0

    return (isEqual, dist)

def compareBolsNew(a, b):
    isEqual = False
    dist = 1
    if a == b or (a == 'TA' and b == 'KDA') or (a == 'KDA' and b == 'TA'):
        isEqual = True
        dist = 0

    return (isEqual, dist)


def getTranscriptionData(fPath):
    '''
    This function takes in a filepath and returns a tuple containing
    ([list of transcripted syllabels], [list of onsets for each transcribed])
    The input file is supposed to be of format:
        <startTime> <tab> <transcriptedBol> <tab> <endTime>
    :input fPath: The absolute file path for the file to be read.
    '''
    result = None
    f = open(fPath, 'r')
    syls = []
    ons = []
    for line in f:
        lineSplit = line.split('\t')
        if lineSplit[1] == '!ENTER' or lineSplit[1] == '!EXIT': #Check for entry and exit labels.. skip them
            continue
        syls.append(lineSplit[1])
        ons.append(float(lineSplit[0].strip()))
    f.close()
    return (syls, ons)

def readCSVLines(fPath):
    '''
    This function takes in a filepath and returns list of annotated bols from the .lab file
    format:
    <bol>
    <bol>
    .
    .
    :input fPath: Tthe absolute path of the file to be read
    '''
    result = None
    f = open(fPath, 'r')
    content = [line.rstrip('\n') for line in f]
    return content

def createString(*args):
    '''
    This function takes multiple arguments and concatenates all in one single string
    '''
    ret = ''
    for s in args:
        ret += str(s)
        ret += ' '
    return ret.strip()

def readPickle(filePath):
    '''
    Inputs a pickle file path and returns the data
    :input filePath: Absolute path of the pickle path
    '''
    if not os.path.isfile(filePath):
        print 'The file does not exist:' + filePath
        sys.exit()

    data = pkl.load(open(filePath, 'rb'))
    return data

#TODO: Remove this
def getSubstringPos(target, query):
    result = []
    lTar = len(target)
    lQue = len(query)
    print 'The target is:' + str(target)
    print 'The query is:' + str(query)

    # The length of the target is smaller than the length of the query
    if lTar < lQue:
        return result
    # If they are equal
    elif target == query:
        return [0]

    for i in range(lTar - lQue + 1):
        subStr = target[i:i+lQue]
        if subStr == query:
            result.append(i)
        #print 'The substring is:' + str(subStr)
        '''
        match = True
        print 'Running for i=' + str(i) + '*******'
        for j in range(lQue):
            print 'Matching ' + query[j] + ' and ' + target[i + j]
            if query[j] != target[i+j]:
                match = False
                break
        if match == True:
            result.append(i)
        '''
    #print 'The query is:' + str(query)
    return result

def uniqMatchPos(matches):
    '''
    The aim of the function is to remove the redundant starting positions for the starting location of the matched 
    subsequences. If we have multiple subsequences starting at the same position, we take the one with the highest score.
    This function takes in the list of the format:
    [((<start_pos1>,<score>), <subsquence1>), ((<start_pos2>,<score>), <subsquence2>), ...]
    :input matches: The redundant list of matches as mentioned above.
    :output uniqMatches: Non-redundant list of the matches with the above mentioned criteria.

    #Assumption: Right now, we have assumed that the for multiple substrings starting at the same position, we will have different 
    scores for all
    '''

    # Getting the list of unique starting positions
    posDic = {}

    #positions = [] # Stores the positions for the maximum scored matches

    for tup in matches:
        # Check if the position of the start of subsequence already exists in the dictionary
        if tup[0][0] in posDic.keys():
            
            # Getting the position
            oldVal = posDic[tup[0][0]]

            # Checking if the score is same; throw exception (ideally should not)
            if oldVal[0][1] == tup[0][1]:
                #If the substrings at same starting locations have same score.. take the one which has more length
                oldSeq = oldVal[1].strip().split(' ')
                newSeq = tup[1].strip().split(' ')
                if len(oldSeq) < len(newSeq):
                    posDic[tup[0][0]] = tup
                elif len(oldSeq) == len(newSeq):
                    print 'Old:' + str(oldSeq) + ' ; oldVal:' + oldVal[1]
                    print 'New:' + str(newSeq) + ' ; tup:' + tup[1]
                    print 'Same length for the sequences beginning at the same location and having same score.. Continuing.. check logs!!'
                    continue
            
            # Checking if the new score is larger than the old one 
            elif oldVal[0][1] < tup[0][1]:
                posDic[tup[0][0]] = tup
        # If not already existing in the dictionary.. add to the dictionary
        else:
            posDic[tup[0][0]] = tup


    # Return the list of the substrings with unique starting positions
    return [((key, val[0][1]), val[1]) for key, val in posDic.iteritems()]

# TODO: Remove this - Not in use anymore
def getTopNMatches(matches, topN):
    '''
    This function takes a list of matches which start at unique positions and return a list of
    matches which have the topN scores.
    '''
    if topN == None:
        print 'utils.getTopNMatches::topN value is not provided..'
        sys.exit()

    scoreSet = set()
    topNMatches = []
    
    # Creating a set of scores present...
    for tup in matches:
        if tup[0][1] > 0.0: # Add only if it is greater than 0
            scoreSet.add(tup[0][1])

    scoreList = list(scoreSet) # getting a list from the set
    scoreList.sort(reverse = True)

    # Checking if top scores are more than what we are searching for
    if len(scoreList) > topN:
        scoreList = scoreList[:topN]

    for tup in matches:
        if tup[0][1] in scoreList:
            topNMatches.append((tup[0][0],tup[1]))

    return topNMatches

def getMatchesForTres(matches, scoreTres):
    '''
    Returns the list of matches which have score greater than the scoreTres
    '''
    result = []
    for tup in matches:
        if tup[0][1] >= scoreTres:
            result.append((tup[0][0], tup[1]))

    return result

def getOverlap(first, second):
    '''
    This function takes in two tuples of the format: (<start>,<end>) and determine the overlap between them.
    Assumption is that the <start> is smaller than the <end>
    '''

    if (first[0] >= first[1]):
        print 'The start time is greater than the end time for first.. ' + str(first)
        sys.exit()
    
    if (second[0] >= second[1]):
        print 'The start time is greater than the end time for second.. ' + str(second)
        sys.exit()

    if first[1] < second[0] or first[0] > second[1]:
        return 0

    # If we reach here, there is an overlap...

    timeSeq = []
    timeSeq.append(first[0])
    timeSeq.append(first[1])
    timeSeq.append(second[0])
    timeSeq.append(second[1])

    #Sort the time sequence
    timeSeq.sort()

    return timeSeq[2] - timeSeq[1]


def getAllSylbData(tPath, lblDir, onsDir):
    '''
    This function takes in the directory path for the transcription data, the label fields driectory (which stores the ground truth)
    and the onsDir which stores the onsets for the ground truth transcription.
    The function returns the list of tuples, where each tuple has data for a particular file in the format
    ((<list_of_syllables_trans>, <list_of_onsets_trans>),(<list_of_syllables_gt>, <list_of_onsets_gt>, <composition_name>))
    '''
    print 'Running utils.getAllSylbData ....'
    masterData = []
    for d in os.listdir(tPath):
        subDir = os.path.join(tPath, d)
        if os.path.isdir(subDir):
            print 'Working the directory:' + subDir
            for f in os.listdir(subDir):
                if f.endswith('.csv'):
                    #Check for the label file
                    lblFile = os.path.join(lblDir, d + '.lab')
                    if not os.path.isfile(lblFile):
                        print 'The label file does not exist..' + lblFile
                        sys.exit()

                    # Check if the onset file exists
                    onsFile = os.path.join(onsDir, d + '.csv')
                    if not os.path.isfile(onsFile):
                        print 'The onset file does not exist..' + onsFile
                        sys.exit()

                    # Get the sequence from the label file and the respective onset information(the ground truth)
                    origScore = readCSVLines(lblFile)
                    origOnsets = map(float, readCSVLines(onsFile))

                    if len(origScore) != len(origOnsets):
                        print 'The length of the labels and the onsets are different.. Exiting'
                        sys.exit()

                    # Get the sequence from the transcribed file (the transcripted version)
                    transFile = os.path.join(subDir, f)

                    # Get the transcription and the onset information ...
                    ref = getTranscriptionData(transFile)

                    masterData.append((ref,(origScore, origOnsets), d))

    return masterData


# TODO: Remove this - probably move to a personalized library (Make in dropbox)
def cleanDir(dPath):
    '''
    This function takes in a director path and then deletes it and then create it again.. thus clean it!
    '''
    shutil.rmtree(dPath)
    os.makedirs(dPath)

# TODO: This might be used later on ... but as of now not used
def findClosestIndex(x, arr, low, high):
    '''
    This function takes in a 'x' and finds the index which has the closest value to 'x' in the searchList.
    Inputs the low and high which represent the indices within which we want to search for the closest index.
    searchList is assumed to be sorted!
    '''
    if arr[high] <= x: # x is greater than all
        return high
    if arr[low] >= x: # x is smaller than all
        return low

    mid = (low + high) / 2 # Getting the mid value

    if arr[mid] <= x and arr[mid+1] > x:
        lDist = abs(x - arr[mid]) # Calculating the distance from the element at mid
        rDist = abs(x - arr[mid+1]) # Calculating the distance from the element at the right of the mid
        if lDist <= rDist:
            return mid
        else:
            return mid + 1

    if arr[mid] < x: # Else if the mid element is smaller than the x
        return findClosestIndex(x, arr, mid, high)

    return findClosestIndex(x, arr, low, mid)

def getStrOfLen(obj, l):
    '''
    This function takes in an object and makes it of length l by appending spaces to it
    '''
    objStr = str(obj)
    if len(objStr) > l:
        print 'utils.getStrOfLen::The length of the object already greater than l.. exiting!' + objStr
        sys.exit()

    diffLen = l - len(objStr)
    while diffLen > 0:
        objStr += ' '
        diffLen -= 1

    return objStr


def comparePatterns(ptr1, ptr2):
    '''
    This function is used to get the non-overlapping patterns in the transcription.
    Returns the sorted list of patterns such that the sorting preference is given to the length and then
    to the starting point.
    '''
    #Put a check
    if ptr1[0][1] < ptr1[0][0] or ptr2[0][1] < ptr2[0][0]:
        print 'utils.comparePatterns::The end time is greater than the start time.'
    if ptr1[0][1] - ptr1[0][0] > ptr2[0][1] - ptr2[0][0]:
        return 1
    elif ptr1[0][1] - ptr1[0][0] < ptr2[0][1] - ptr2[0][0]:
        return -1
    else: # They are of equal length ...
        return ptr1[0][0] - ptr2[0][0]

def getPtrFileName(ptrNum):
    if ptrNum == 1:
        return 'DHE!RE!DHE!RE!KI!TA!TA!KI!NA!TA!TA!KI!TA!TA!KI!NA.pkl'
    elif ptrNum == 2:
        return 'TA!TA!KI!TA!TA!KI!TA!TA!KI!TA!TA!KI!TA!TA!KI!TA.pkl'
    elif ptrNum == 3:
        return 'TA!KI!TA!TA!KI!TA!TA!KI.pkl'
    elif ptrNum == 4:
        return 'TA!TA!KI!TA!TA!KI.pkl'
    elif ptrNum == 5:
        return 'TA!TA!KI!TA.pkl'
    elif ptrNum == 6:
        return 'KI!TA!TA!KI.pkl'
    elif ptrNum == 7:
        return 'TA!TA!KI!NA.pkl'
    elif ptrNum == 8:
        return 'DHA!GE!TA!TA.pkl'

def getBestMeasuresForPattern(baseDir,ptrNum):
    """
    This function returns the best results for a particular pattern.
    0: DHE RE DHE RE - 16
    1: second 16 length
    2: 
    """
    fileName = getPtrFileName(ptrNum)
    maxP = None
    maxR = None
    maxFm = None
    pDirs = os.listdir(baseDir)
    for pDir in pDirs:
        pDirPath = os.path.join(baseDir, pDir)
        if os.path.isdir(pDirPath):
            p = float(pDir.split('_')[1])
            #print '$$$$$$ Running for p=' + str(p)
            betaDirs = os.listdir(pDirPath)
            for betaDir in betaDirs:
                betaDirPath = os.path.join(pDirPath, betaDir)
                if os.path.isdir(betaDirPath):
                    beta = float(betaDir.split('_')[1])
                    #print '###### Running for beta=' + str(beta)
                    scoreDirs = os.listdir(betaDirPath)
                    for scoreDir in scoreDirs:
                        scoreDirPath = os.path.join(betaDirPath, scoreDir)
                        if os.path.isdir(scoreDirPath):
                            tres = float(scoreDir.split('_')[1])
                            #print 'Running for tres=' + str(tres)
                            pklFilePath = os.path.join(scoreDirPath, fileName)
                            fo = open(pklFilePath, 'rb')
                            tup = pkl.load(fo)
                            fo.close()
                            
                            if tup[0] is not None:
                                if maxP and tup[0] > maxP[1]:
                                    maxP = ((p, beta, tres), tup[0])
                                elif maxP is None:
                                    maxP = ((p, beta, tres), tup[0])
                                
                            if tup[1] is not None:
                                if maxR and tup[1] > maxR[1]:
                                    maxR = ((p, beta, tres), tup[1])
                                elif maxR is None:
                                    maxR = ((p, beta, tres), tup[1])
                            
                            #print 'tup[0]=' + str(tup[0])
                            #print 'tup[1]=' + str(tup[1])
                            if tup[0] is not None and tup[1] is not None and (tup[0] > 0.0 or tup[1] > 0.0):
                                fm = 2 * tup[0] * tup[1]/(tup[0] + tup[1])
                                if maxFm and fm > maxFm[1]:
                                    maxFm = ((p, beta, tres), fm, tup[0], tup[1])
                                elif maxFm is None:
                                    maxFm = ((p, beta, tres), fm, tup[0], tup[1])
    return (maxP, maxR, maxFm)





def getHighestMeasures(pklFile):
    '''Takes in the pklFile and returns the params for highest precision, recal and f-measures'''
    fo = open(pklFile, 'rb')
    data = pkl.load(fo)
    fo.close()
    maxP = None
    maxR = None
    maxFm = None
    for p, betaList in data:
        for beta, tresList in betaList:
            for tres, tup in tresList:
                if tres == 0.0:
                    continue # Not considering the zero one ..
                if tup[0] is not None:
                    if maxP and tup[0] > maxP[1]:
                        maxP = ((p, beta, tres), tup[0])
                    elif maxP is None:
                        maxP = ((p, beta, tres), tup[0])
                
                if tup[1] is not None:
                    if maxR and tup[1] > maxR[1]:
                        maxR = ((p, beta, tres), tup[1])
                    elif maxR is None:
                        maxR = ((p, beta, tres), tup[1])
                
                if tup[0] is not None and tup[1] is not None:
                    fm = 2 * tup[0] * tup[1]/(tup[0] + tup[1])
                    if maxFm and fm > maxFm[1]:
                        maxFm = ((p, beta, tres), fm, tup[0], tup[1])
                    elif maxFm is None:
                        maxFm = ((p, beta, tres), fm, tup[0], tup[1])

    return (maxP, maxR, maxFm)

# TODO: Remove once the full data is storing functionality is confirmed
def getOverAllObj(baseDir):
    '''
    Was not returning the result of the beta
    '''
    result = []
    for d in os.listdir(baseDir):
        if os.path.isdir(os.path.join(baseDir, d)):
            pVal = float(d.split('_')[1])
            print 'Working for the directory:' + d
            pklFileName = os.path.join(baseDir, d, 'result.pkl')
            fo = open(pklFileName, 'rb')
            data = pkl.load(fo)
            result.append((pVal,data))

    # Dumping output to the pickle file
    resFileName = os.path.join(baseDir, 'result.pkl')
    resFile = open(resFileName, 'wb')
    pkl.dump(result, resFile)
    resFile.close()

def loadMatData(filePath):
    """This file returns the data from a .mat file"""
    return scipy.io.loadmat(filePath)

def getStrokeList(strokeFile):
    """Gets the list of strokes from the file"""
    return [line.split(' ')[0] for line in open(strokeFile, 'r')]

def getSimilarityDict(strokeFile, simMatrixMatFile):
    """This funcion takes in the similarity marix in .mat format the strokeFile and return dictionary of dictioaries"""
    # Get the list of strokes from the strokeFile
    strkList = getStrokeList(strokeFile)
    n = len(strkList)

    # Get the similarity matrix
    data = loadMatData(simMatrixMatFile)['myMeasure']
    res = {}
    for i in range(n):
        strkDict = {}
        for j in range(n):
            strkDict[strkList[j]] = data[i][j]
        res[strkList[i]] = strkDict

    return res

class Similarity:
    """Class for storing the similarity criterion"""
    def __init__(self, simDict, tres):
        #print 'Initializing the similarity object!!'
        self.simDict = simDict
        self.tres = tres

    def getDist(self, s1, s2):
        if self.simDict[s1][s2] <= self.tres:
            return (True, self.simDict[s1][s2])
        else:
            return (False, self.simDict[s1][s2])

def loadConfig(filePath):
    config = {}
    for line in open(filePath, 'r'):
        if line.strip()[0] != '#':
            config[line.strip().split('=')[0]] = line.strip().split('=')[1]

    return config

def plotRes(res, feat = None, imgFile = None):
    """
    Takes in the res which can contain ghte pkl file or the data in the file itself and the feature that is 
    varied and plots the figure for that.
    """
    if imgFile is None:
        imgFile = 'figure.png'

    bottom = 30.0
    left = 30.0

    # Reading the result from the pickle file
    if isinstance(res, str):
        resFile = open(res, 'rb')
        res = pkl.load(resFile)
        resFile.close()
    
    # Creating labels
    if feat is None:
        tmplt = '{0}'
    else:
        tmplt = feat + '_{0}'
    labels = [ tmplt.format(i[0]) for i in res]

    precList = [1 - i[1][0]/100.0 for i in res]
    recList = [i[1][1]/100.0 for i in res]

    fig = plt.figure()
    fig.subplots_adjust(bottom = 0.1)
    
    ax1 = fig.add_subplot(111)
    ax1.scatter(precList, recList, marker = 'o')
    #ax1.set_ylim(top=80.0)
    #ax1.set_xlim(right=80.0)

    #Plotting the x = y line
    #ax1.plot([left, 75], [left, 75], 'k-')

    # Adding labels to the axes
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive rate')

    # Plotting labels
    
    for label, tup in zip(labels, res):
        ax1.annotate(label, xy=(1 - tup[1][0]/100.0, tup[1][1]/100.0), textcoords = 'offset points',\
                ha = 'right', va = 'bottom',\
                #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),\
                #arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0.2')\
                )

    fig.savefig(imgFile)


def expWarp(x, k = 5):
    return (math.exp(k*x) - 1)/(math.exp(k) - 1)

def getPtrLenList(ptrs):
    """
    Takes in a list of patterns if the format 'syl1 syl2 syl3' (string elements) and returns a list of count of elements in each of them.
    """
    res = []
    for ptr in ptrs:
        res.append(len(ptr.strip().split(' ')))

    return res

if __name__=='__main__':
    loadConfig()


#################### Functions specific for Sertan #######################
def readCSVLinesAsArrays(fPath):
    """
    Takes in a csv file and and returns a list with each element as list of comma separated str elements
    """
    if not os.path.isfile(fPath) or not fPath.endswith('.csv'):
        print  "utils::readCSVLinesAsArrays::" + fPath + ' is not a csv file'

    fo = open(fPath, 'r')
    res = [line.strip('\n').split(',') for line in fo]

    return res


def loadQueryPhrases(phraseDir='/home/swapnil/Desktop/Sertan_Data/query'):
    """
    Takes in a directory containing files having the melodic phrases
    """
    res = []
    if not os.path.isdir(phraseDir):
        print  "utils::loadQueryPhrases::" + phraseDir + ' is not a valid directory'
    for f in os.listdir(phraseDir):
        fPath = os.path.join(phraseDir, f)
        if f.endswith('.csv'):
            res.append(readCSVLinesAsArrays(fPath))

    return res


def plotHistogram(data, xlabel, ylabel):
    '''
    Plot the histogram for the data with x and y labels as mentioned
    '''
    plt.hist(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
