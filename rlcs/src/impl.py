'''
This module defines the functions to represent various blocks of the RLCS
'''
import os
import sys
import numpy as np
import math
import utils as ut
import logging
import collections

# create logger with 'rlcsRun'
logger = logging.getLogger('rlcs')
logger.setLevel(logging.INFO)

# create a file handler 
fh = logging.FileHandler('rlcs.log')
fh.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# add the handlers to logger
logger.addHandler(fh)


def getCostMatrices(reference, query, compare):  
    '''
    This function takes in the reference and the query objects and returns a tuple 
    of two cost matrices where the first(costMatrix) considers the distance between the two objects in the cost
    while the other(costMatrixLCS) does not. It also returns the trace back matrice which tells us the path back
    for the best match.
    The distance is defined based on 'compare function'
    :param reference: list of the object considered as the target list
    :param query: list of the object considered as the query list
    :param compare: a funcion which returns a truple (isEqual, distance) where
        'isEqual' says if the two are close enough
        'distance' is the distance between them
    '''
    # This is the list to be seached into; normally larger than the reference
    n = len(reference)

    #This is th list of objects to be searched for normally small compare to reference
    m = len(query)

    #Initializing the matrix with size one greater than the query and refernce sizes to fill up a dummy position
    costMatrix = np.zeros((n + 1, m + 1))
    costMatrixLCS = np.zeros((n + 1, m + 1))

    for i in range(n):
        for j in range(m):
            isEqual, dist = compare.getDist(reference[i], query[j])
            if isEqual == True:
                costMatrix[i+1][j+1] = costMatrix[i][j] + (1 - dist)
                costMatrixLCS[i+1][j+1] = costMatrixLCS[i][j] + 1
            else:
                costMatrix[i+1][j+1] = max(costMatrix[i][j+1], costMatrix[i+1][j])
                costMatrixLCS[i+1][j+1] = max(costMatrixLCS[i][j+1], costMatrixLCS[i+1][j])

    return (costMatrix, costMatrixLCS)

def getWidthMatrices(reference, query, compare, costMatrix):
    '''
    This funciton takes in the reference, query, compare function and the costMatrix and returns the
    width-across-reference and width-across-query matrices as a tuple
    :param reference: list of the object considered as the target list
    :param query: list of the object considered as the query list
    :param compare: a funcion which returns a truple (isEqual, distance) where
        'isEqual' says if the two are close enough
        'distance' is the distance between them
    :param costMatrix: the matrix computed with the help of the cost funtion
    '''
    n = len(reference)
    m = len(query)

    #Matrix storing the width acrosss reference
    war = np.zeros((n + 1, m + 1))

    #Matrix storing the width across query
    waq = np.zeros((n + 1, m + 1))

    for i in range(0, n):
        for j in range(0, m):
            isEqual, dist = compare.getDist(reference[i], query[j])
            if isEqual == True:
                war[i + 1][j + 1] = war[i][j] + 1 # Just taking the distance for now.
                waq[i + 1][j + 1] = waq[i][j] + 1 # Just taking the distance for now.

            elif costMatrix[i][j + 1] >= costMatrix[i + 1][j]:
                # for WAR
                if war[i][j + 1] == 0:
                    war[i + 1][j + 1] = 0 #Although initialized as zero; doing this for readibility
                else:
                    war[i + 1][j + 1] = war[i][j + 1] + 1
                # for WAQ
                waq[i + 1][j + 1] = waq[i][j + 1]

            else:
                # for WAR
                war[i + 1][j + 1] = war[i + 1][j]
                
                # for WAQ
                if waq[i + 1][j] == 0:
                    waq[i + 1][j + 1] = 0 # for readibility
                else:
                    waq[i + 1][j + 1] = waq[i + 1][j] + 1

    return (war, waq)

def getScoreMatrices(c, ca, wr, wq, nQuery, beta, p, formula):
    '''
    This function inputs all the matrices computed in the previous steps and then returns the scoreMatrix
    and the topN scores along with the indices on which those substrings end (format - (ref_position, query_position)).
    :param c: the cost matrix
    :param ca: the cost matrix which doesn't consider similarity score between the two objects
    :param wr: Width across reference matrix
    :param wq: Width across query matrix
    :param beta: The weight given to the score of reference
    :param p: The threshold factor for the ratio between the subsequence length and the query length
    :param nQuery: The length of the query
    :param topN: number of top scores that we want to find
    '''

    if formula == None:
        print 'formula number not specified. Exiting!!'
        sys.exit()
    n, m = c.shape
    scoreMatrix = np.zeros((n, m))
    for i in range(1, n):
        for j in range(1, m):
            #Checking the denominator is not zero.....
            if wr[i][j] * wq[i][j] != 0:
                if ca[i][j] * 1.0 / nQuery >= p:
                    if 1 == formula:
                        scoreMatrix[i][j] = (beta * c[i][j] / wr[i][j] + (1 - beta) * c[i][j] / wq[i][j]) * (c[i][j] / nQuery) #- Formula 1
                    elif 2 == formula:
                        scoreMatrix[i][j] = (beta * c[i][j] / wr[i][j] + (1 - beta) * c[i][j] / wq[i][j]) * (ca[i][j] / nQuery) #- Formula 2
                    elif 3 == formula:
                        scoreMatrix[i][j] = (beta * c[i][j] / (wr[i][j] * wr[i][j]) + (1 - beta) * c[i][j] / (wq[i][j] * wq[i][j])) * (ca[i][j] / nQuery) #- Formula 3
                    elif 4 == formula:
                        scoreMatrix[i][j] = (beta * math.tan(c[i][j] * np.pi/ (wr[i][j] * 4)) + (1 - beta) * math.tan(c[i][j] * np.pi/ (wq[i][j] * 4))) * (ca[i][j] / nQuery) #- Formula 4
                    elif 5 == formula:
                        scoreMatrix[i][j] = (beta * ut.expWarp(c[i][j] / wr[i][j]) + (1 - beta) * ut.expWarp(c[i][j] / wq[i][j])) * (ca[i][j] / nQuery) #- Formula 5
                    elif 6 == formula:
                        scoreMatrix[i][j] = (beta * ut.expWarp(c[i][j] / wr[i][j], 1) + (1 - beta) * ut.expWarp(c[i][j] / wq[i][j], 1)) * (ca[i][j] / nQuery) #- Formula 6
                    elif 7 == formula:
                        scoreMatrix[i][j] = (beta * ut.expWarp(c[i][j] / wr[i][j], 2) + (1 - beta) * ut.expWarp(c[i][j] / wq[i][j], 2)) * (ca[i][j] / nQuery) #- Formula 7
                    elif 8 == formula:
                        scoreMatrix[i][j] = (beta * ut.expWarp(c[i][j] / wr[i][j], 3) + (1 - beta) * ut.expWarp(c[i][j] / wq[i][j], 3)) * (ca[i][j] / nQuery) #- Formula 8
                    elif 9 == formula:
                        scoreMatrix[i][j] = (beta * ut.expWarp(c[i][j] / wr[i][j], 4) + (1 - beta) * ut.expWarp(c[i][j] / wq[i][j], 4)) * (ca[i][j] / nQuery) #- Formula 9
                    elif 10 == formula:
                        scoreMatrix[i][j] = (beta * ut.expWarp(c[i][j] / wr[i][j], 6) + (1 - beta) * ut.expWarp(c[i][j] / wq[i][j], 6)) * (ca[i][j] / nQuery) #- Formula 10
                    elif 11 == formula:
                        scoreMatrix[i][j] = (beta * ut.expWarp(c[i][j] / wr[i][j], 7) + (1 - beta) * ut.expWarp(c[i][j] / wq[i][j], 7)) * (ca[i][j] / nQuery) #- Formula 11
                    elif 12 == formula:
                        scoreMatrix[i][j] = (beta * ut.expWarp(c[i][j] / wr[i][j], 8) + (1 - beta) * ut.expWarp(c[i][j] / wq[i][j], 8)) * (ca[i][j] / nQuery) #- Formula 9
                    elif 13 == formula:
                        scoreMatrix[i][j] = (beta * ut.expWarp(c[i][j] / wr[i][j], 9) + (1 - beta) * ut.expWarp(c[i][j] / wq[i][j], 9)) * (ca[i][j] / nQuery) #- Formula 9
                else:
                    scoreMatrix[i][j] = 0.0
    
    # Getting the score indices ...
    scoreIndices = {}
    for i in range(1, n):
        for j in range(1, m):
            score = scoreMatrix[i][j]
            if score in scoreIndices.keys():
                scoreIndices[score].append((i,j))
            else:
                scoreIndices[score] =  [(i,j)]
    return (scoreMatrix, scoreIndices)

def traceBack(ref, query, tup, c, compare):
    '''
    This function gives back the subsequence and the starting location of the subsequence in the reference
    :input ref: The target string in which we are searching
    :input query: The pattern that we are searching for
    :input tup: The tuple giving end position of the subsequence in the ref and query respectively
    :input c: The cost matrix computed in the previous steps
    :input compare: Compare function for knowing if the two elements are similar or not.
    :input score: Score associated to the substring
    '''
    result  = ''
    x = tup[0]
    y = tup[1]

    logger.debug('Running the traceBack for x=' + str(x) + ' and y=' + str(y))

    #print 'The initial value of x=' + str(x)
    #print 'The initial value of y=' + str(y)

    # Setting this mode to take into account the last jump....
    mode = None

    while x > 0 and y > 0:
        #print 'The compare tres is:' + str(compare.tres)
        if compare.getDist(ref[x-1], query[y-1])[0] == True:
            #print 'Sylbls equal with dist=' + str(compare.getDist(ref[x-1], query[y-1])[1])
            result = '!' + ref[x - 1] + ' ' + result
            x -= 1
            y -= 1
            mode = 0

        elif c[x-1][y] > c[x][y-1]:
            #print 'c[x-1][y] > c[x][y-1]'
            #print 'c[x-1][y]' + str(c[x-1][y])
            #print 'c[x][y-1]' + str(c[x][y-1])
            
            result = ref[x - 1] + ' ' + result
            x -= 1
            mode = 1

        else:
            #print 'c[x-1][y] <= c[x][y-1]'
            #print 'c[x-1][y]' + str(c[x-1][y])
            #print 'c[x][y-1]' + str(c[x][y-1])
            y -= 1
            mode = 2

    if mode == 0: # If last jump was made in both the directions.. i.e. diagonally
        y += 1
        x += 1

    elif mode == 1: # If the last jump was made along the x-axis
        x += 1

    elif mode == 2: # If the last jump was made along the y-axis
        y += 1
    
    logger.debug('Final value for x=' + str(x) + ' and y=' + str(y))

    l = len(result.strip().split(' '))

    #if l <= 1:
        #print 'The length of the pattern found is smaller than 1.. Exiting.. l:' + str(l)
        #sys.exit()

    return ((x,x+l), result.strip())

def getMatchesWithPos(ref, query, scoreIndices, c, ca,compare):
    '''
    This function takes in the reference, query, scoreMatrix
    '''
    result = {}
    logger.debug('Inside rlcs.getMatchesWithPos...')
    for key, value in scoreIndices.iteritems():
        if key == 0.0: # If the score is 0.. we don't even consider that!
            continue
        result[key] = []
        for t in value:
            logger.debug('Getting the traceBack for ' + str(t))
            #print 'The score =' + str(key)
            #print 'The ca[x][y] is ' + str(ca[t[0]][t[1]])
            trcBack = traceBack(ref, query, t, c, compare)
            logger.debug('Trace Back is:' + str(trcBack))
            result[key].append(trcBack)
    return result

def getMatchesWithNoOverlaps(ref, query, scoreIndices, c, ca, compare):
    matchesWithPos = getMatchesWithPos(ref, query, scoreIndices, c, ca, compare)
    
    # Final Result
    result = []
    
    # Create an array for blacklisting to avoid overlaps
    blackList = [0] * (len(ref) + 1) # One length more because we have stored in that way ...
    
    # Sorting the dictionary based on the key
    keySorted = collections.OrderedDict(sorted(matchesWithPos.items(), reverse = True))
    #print keySorted
    #print '\n\n'

    # arr format: [<(start, end), score, result>,<....>]
    for key, arr in keySorted.iteritems():
        sortedArr = sorted(arr, cmp=ut.comparePatterns)
        for elem in sortedArr:
            startPos = elem[0][0]
            endPos = elem[0][1]
            if 1 not in blackList[startPos:endPos]: # There is no other higher priority pattern present here
                result.append(((startPos,key), elem[1])) # Adding this to result list
                blackList[startPos:endPos] = [1] * (endPos - startPos) # blacklisting the positions
    return result

def runRLCS(ref, query, compare, beta, p, formula):
    '''
    This function runs the RLCS with ref and query as the two input query strings and returns
    a list of matches.
    :input ref: The reference string
    :input query: The query string
    :input compare: The compare function measuring the similarity between the elements
    :input beta: The weight that should be given to the score of the reference
    :input p: Threshold for the ration of the substring to the query
    :output matches: list of the matches with each match in the form of a tuple of form (<start_position>, <pattern>)
    '''
    c, ca = getCostMatrices(ref, query, compare)
    war, waq = getWidthMatrices(ref, query, compare, c)
    sm, si = getScoreMatrices(c, ca, war, waq, len(query), beta, p, formula)
    matches =  getMatchesWithNoOverlaps(ref, query, si, c, ca,compare)
    #print matches
    #sys.exit()
    logger.debug('c=' + str(c)) 
    logger.debug('ca=' + str(ca)) 
    logger.debug('waq=' + str(waq)) 
    logger.debug('war=' + str(war)) 
    logger.debug('sm=' + str(sm)) 
    logger.debug('si=' + str(si))
    logger.debug('matches=' + str(matches))
    return matches

# TODO: Not is use.. remove eventually ...
def getRLCSDataForComp(compData, fo, fSum, query, compare, beta, p, topN):
    '''Takes in the transcription data for a compostion and returns the RLCS result for both the transcription and the gt. Also, takes in the
    parameters for the RLCS; query, beta, p, and topN'''
    
    logger.debug('Running utils.getRLCSDataForComp for ' + compData[2] + ' with query ' + str(query))
    
    # Get the RLCS result for the trancription...
    logger.debug('Running RLCS for the transcription: ' + str(compData[0][0]))
    bestMatches = runRLCS(compData[0][0], query, compare = compare, beta = beta, p = p)


    logger.debug('Running utils.uniqMatchPos for transcription matches..')
    uniqBestMatches = ut.uniqMatchPos(bestMatches)
    topNBestMatches = ut.getTopNMatches(uniqBestMatches, topN)
    
    
    # Run RLCS on the ground truth....
    logger.debug('Running RLCS for the ground truth: ' +str(compData[1][0]))
    scoreMatches = runRLCS(compData[1][0], query, compare = ut.compareBols, beta = beta, p = p)

    
    logger.debug('Running utils.uniqMatchPos for ground truth matches..')
    uniqScoreMatches = ut.uniqMatchPos(scoreMatches)
    topNScoreMatches = ut.getTopNMatches(uniqScoreMatches, topN)
    
    fo.write('The uniqBestMatches:' + str(uniqBestMatches) + '\n')
    logger.debug('Result for utils.uniqMatchPos for transcription matches..' + str(uniqBestMatches))
    fo.write('The uniqScoreMatches:' + str(uniqScoreMatches) + '\n')
    fo.write('The topNBestMatches:' + str(topNBestMatches) + '\n')
    fo.write('The topNScoreMatches:' + str(topNScoreMatches) + '\n')
    logger.debug('Result for utils.uniqMatchPos for ground truth matches..' + str(uniqScoreMatches))

    # Logging the number of the each of the above ..
    fo.write('The number of unique matches found in the transcription are:' + str(len(uniqBestMatches)) + '\n')
    fo.write('The number of unique matches found in the ground truth are:' + str(len(uniqScoreMatches)) + '\n')
    fo.write('The number of topN score unique matches found in the transcription are:' + str(len(topNBestMatches)) + '\n')
    fo.write('The number of topN score unique matches found in the score are:' + str(len(topNScoreMatches)) + '\n')
    fSum.write(ut.getStrOfLen(len(topNBestMatches), 12) + '\t')
    fSum.write(ut.getStrOfLen(len(topNScoreMatches), 12) + '\t')

    return ((uniqBestMatches, topNBestMatches), (uniqScoreMatches, topNScoreMatches))


def getUniqMatches(compData, query, compare, beta, p, fComp, formula):
    '''Takes in the transcription data for a compostion and returns the RLCS result for both the transcription and the gt. Also, takes in the
    parameters for the RLCS; query, beta, p'''    
    
    # Get the RLCS result for the trancription...
    bestMatches = runRLCS(compData, query, compare = compare, beta = beta, p = p, formula = formula)
    #uniqBestMatches = ut.uniqMatchPos(bestMatches)
    
    
    # Run RLCS on the ground truth....
    #scoreMatches = runRLCS(compData[1][0], query, compare = ut.compareBols, beta = beta, p = p)
    #uniqScoreMatches = ut.uniqMatchPos(scoreMatches)
    
    #print uniqBestMatches
    #print '\n\n'
    #print uniqScoreMatches
    #sys.exit()

    fComp.write('Running for query:' + str(query) + '\n')
    #fComp.write('The uniqBestMatches:' + str(uniqBestMatches) + '\n')
    #fComp.write('The uniqScoreMatches:' + str(uniqScoreMatches) + '\n')

    # Logging the number of the each of the above ..
    fComp.write('The bestMatches:' + str(bestMatches) + '\n')
    fComp.write('The number of unique matches found in the transcription are:' + str(len(bestMatches)) + '\n')
    #fComp.write('The number of unique matches found in the ground truth are:' + str(len(uniqScoreMatches)) + '\n\n')
    #fSum.write(ut.getStrOfLen(len(topNBestMatches), 12) + '\t')
    #fSum.write(ut.getStrOfLen(len(topNScoreMatches), 12) + '\t')

    return bestMatches

def getGTPatterns(compData, query, fComp=None):
    lC = len(compData)
    lQ = len(query)

    res = []

    pG = 0 # Global pointer
    while pG <= lC - lQ:
        pL = 0 # Local pointer
        hit = False # Initially not found the pattern
        while pL < lQ and compData[pG + pL] == query[pL]:
            pL += 1
        if pL == lQ:
            res.append((pG + 1,query))
            pG += lQ # If match found.. jump over the complete pattern
        else:
            pG += 1

    if fComp is not None:
        fComp.write('The ground truth matches:' + str(res) + '\n')
        fComp.write('The number of matches found in the ground truth are:' + str(len(res)) + '\n\n')
 
    return res

def getBaselinePatterns(compData, query, fComp):
    lC = len(compData)
    lQ = len(query)

    res = []

    pG = 0 # Global pointer
    while pG <= lC - lQ:
        pL = 0 # Local pointer
        hit = False # Initially not found the pattern
        while pL < lQ and compData[pG + pL] == query[pL]:
            pL += 1
        if pL == lQ:
            res.append(((pG + 1, 1.0), ' '.join(query)))
            pG += lQ # If match found.. jump over the complete pattern
        else:
            pG += 1

    fComp.write('The ground truth matches:' + str(res) + '\n')
    fComp.write('The number of matches found in the ground truth are:' + str(len(res)) + '\n\n')
    return res
