''' 
Name: Rajib Das
Student Id: 001201973
'''

import random
import sys
import os
import getopt
import math
from array import array
from decimal import *

import networkx as nx
import numpy as np

getcontext().prec = 2   
arguments, remain = getopt.getopt(sys.argv[1:], 'n:p:')

argument_length = len(sys.argv)

file_name = sys.argv[argument_length - 1]


# Checking if user forgot to give the file extension
if(file_name[-4:] != ".gml"):
	file_name = file_name+".gml"

for opt, arg in arguments:
	if opt == '-p':
		p = int(arg)
	if opt == '-n':
		nodeNumbers = int(arg)
		path_graph = nx.path_graph(nodeNumbers)  # create a path graph with n nodes

		for v in path_graph.nodes():
    			path_graph.node[v]['weight'] = random.randrange(1,10)

		for (u,v) in path_graph.edges():
    			path_graph[u][v]['length'] = random.randrange(1,10)
	
		nx.write_gml(path_graph, file_name)


		
try:
	gml_data = nx.read_gml(file_name)



#Checking: Nodes do not have attribute named 'weight'
	nodeFlag = 0
	for v in gml_data.nodes():
		try:
			nodeWeight = gml_data.node[v]['weight']
			nodeFlag += 1

		except KeyError:
			print ("KeyError - weight missing at node:  ", (nodeFlag+1))
			sys.exit(1)
# Checking: Edges do not have attribute named 'length'
	edgeFlag = 0
	notPath = 0
	for (u,v) in gml_data.edges():
		try:
			edgeLength = gml_data[u][v]['length']
			edgeFlag += 1
			
		except KeyError:
			print ("KeyError - length missing at node:  ", (edgeFlag+1))
			sys.exit(1)
	
# Checking: Given network is path or not
	noPathFlag = 0; 
	for u in gml_data.nodes():
		v = u
		noPathFlag = 0; 
		for i in gml_data[str(v)]:
			noPathFlag += 1
			if(abs(int(u)-int(i)) != 1):
				
				print ("The number of adjacent neighbor(s) of node ", str(u), ' ', "is ", noPathFlag,', ', "That means, the given network is not a path!" )
				sys.exit(1)

# code for median
	nodeNumbers = gml_data.number_of_nodes()
	weightAttributesValue = nx.get_node_attributes(gml_data,'weight')
	G = [[[0 for z in range(2) ] for x in range(nodeNumbers)] for y in range(p)]
	F = [[[0 for z in range(2) ] for x in range(nodeNumbers)] for y in range(p)]
	sumL = [[0 for x in range(nodeNumbers)] for y in range(nodeNumbers)]
	sumR = [[0 for x in range(nodeNumbers)] for y in range(nodeNumbers)]
	pCenterLoc = [[0,nodeNumbers-1] for i in range(0,p)]
	pExactLoc = [[] for i in range(0,p)]
	flag = [0 for i in range(0, nodeNumbers)]
	coOrdinate = {}
	coOrdinate[0] = 0
	coOrdVal = 0
	FID = []
	dominatingPairs = [[0 for k in range(0,nodeNumbers)] for i in range(0,nodeNumbers)]
	n = int((nodeNumbers*nodeNumbers-nodeNumbers)/2)
	costArray = [0.0 for k in range(0, n)]
	absoluteCenter = []
	sortedCostArray = []

	optPMedianCost = 0
	approxiPmedianCost = 0 
	optPCenterCost = 0 
	approxiPCenterCost = 0 

	for (u,v) in gml_data.edges():
		coOrdVal = coOrdVal + gml_data[u][v]['length']
		coOrdinate[int(v)] = coOrdVal
	if(p >= nodeNumbers):
		print("The Actual Cost for ",p,"-median(s): ", 0)
		print("The Actual Cost for ",p,"-Center(s): ", 0)
		print("The Approximation Cost for ",p,"-median is", 0)
		print("The Approximation Cost for ",p,"-Center is", 0)
		optPMedianCost = 0
		approxiPmedianCost = 0 
		optPCenterCost = 0 
		approxiPCenterCost = 0 

		if (p > nodeNumbers):
			print("Node ID(s): Case : p = ",p,"and Nodes = ", nodeNumbers,", that is (p is greater than n). Hence,", nodeNumbers," median(s)/Center(s) will be placed from id 0 to", nodeNumbers-1,"(started from 0) and rest", p-nodeNumbers," median(s/Center(s)) will be remain unused.")
	else:
#CM : remove it please

		def funcEnumeratePairs():
			global sortedCostArray
			k = 0
			for i in range (0, nodeNumbers):
				lengthCovered = 0
				for j in range (i+1, nodeNumbers):
					lengthCovered += gml_data[str(j)][str(j-1)].get('length')
					x = (weightAttributesValue[str(j)] * lengthCovered )/(weightAttributesValue[str(i)] + weightAttributesValue[str(j)])
					dominatingPairs[j][i] = weightAttributesValue[str(i)] * x 
					if (dominatingPairs[j][i] > 0):
						costArray[k] =  float(dominatingPairs[j][i])
						k = k+1

			sortedCostArray = np.sort(costArray, kind='mergesort')
			

		def get_dominating_pair(start, end):
			xMax = 0
			for i in range (start, end+1):
				lengthCovered = 0
				for j in range (i+1, end + 1):
					lengthCovered += gml_data[str(j-1)][str(j)].get('length')
					x = (weightAttributesValue[str(j)] * lengthCovered )/(weightAttributesValue[str(i)] + weightAttributesValue[str(j)])
					if (x > xMax):
						xMax = x
						pair1, pair2 = [i+1, j]
	
			return pair1, pair2

		def find_distance(a, b):
			return abs(coOrdinate[a] - coOrdinate[b])
	
		def weighted_distance(a, b):
			return weightAttributesValue[str(a)]*find_distance(a, b)

		def all_weighted_sum(start, end, point):
			sum = 0
			for i in range (start, end+1):
				sum = sum + weighted_distance(i, point)
			return sum

		def find_ids(fid, q, j, G, F):
			if(q < 0):
				return fid
			else:
				facility_location = F[q][j][1]
				#print ("FLOCATION Q J ", facility_location, q, j)
				fid.append(facility_location)
				temp_location = G[q][facility_location][1]
				#print ("TEMP LOCATION ", temp_location)
				fid = find_ids(fid, q-1, temp_location, G, F)
				return fid
		
		
		
		def optPMedian():
			# Base case
			global FID
			for j in range(nodeNumbers-1,-1,-1):
				G[0][j][0] = all_weighted_sum(j, nodeNumbers-1, j)
	
			for j in range(0, nodeNumbers):
				for k in range(j, nodeNumbers):
					if(j==k):
						sumL[j][k] = 0
					else:
						sumL[j][k]= sumL[j][k-1] + weighted_distance(k,j)
			
			for k in range(nodeNumbers-1,-1,-1):
				for j in range(k, -1, -1):
					if(j==k):
						sumR[j][k]=0
					else:
						sumR[j][k] = sumR[j+1][k] + weighted_distance(j,k)
	
			for q in range(0, p):
				for j in range(nodeNumbers-1,-1,-1):
					if(q>0):
						G[q][j][0] = math.inf
						for k in range(j, nodeNumbers):
							sum = sumL[j][k-1] + F[q-1][k][0]
							if(G[q][j][0]>sum):
								G[q][j][0] = sum
								G[q][j][1] = k
		
					F[q][j][0] = math.inf
					for k in range(j, nodeNumbers):
						sum = sumR[j][k] + G[q][k][0]
						if(F[q][j][0] > sum):
							F[q][j][0] = sum
							F[q][j][1] = k				  

			FID = find_ids(FID, p-1, 0, G, F)
			return F[p-1][0][0]

# Actual p-Center Cost

		def funcMaxCost(start, end):
			minimaxCost = 0
			weightAttributesValue = nx.get_node_attributes(gml_data,'weight')
			if(start >= end):
				return 0
			else:
				for i in range (start, end+1):
					lengthCovered = 0
					for j in range (i+1, end+1):
						lengthCovered += gml_data[str(j-1)][str(j)].get('length')
						x = (weightAttributesValue[str(j)] * lengthCovered )/(weightAttributesValue[str(i)] + weightAttributesValue[str(j)])
						tempCost = weightAttributesValue[str(i)] * x
						if(tempCost > minimaxCost):
							minimaxCost = tempCost
			
				return minimaxCost

		def funcBinarySearch(alist):
			first = 0
			last = len(alist)-1
			found = False
			temp = 'null'
			cost = 0

			while first<=last:
				midpoint = math.ceil((first + last)/2.0)
				if(first != last):
					temp, cost = funcCoverageProblem(sortedCostArray[midpoint])
					if(temp <= p):
						last = midpoint-1
					else:
						first = midpoint + 1
				else:
					temp, cost = funcCoverageProblem(sortedCostArray[first])
					if(temp > p):
						first = first + 1
						temp, cost = funcCoverageProblem(sortedCostArray[first])
					break
			if(temp <= p):
				found = 'true'

			return found, cost

		def funcCoverageProblem(r):
			global absoluteCenter
			absoluteCenter = []

			rv = r
			i  = 0 
			while True:
				distance = find_distance(i, i+1)
				weight = weightAttributesValue[str(i)]
				i_to_i1_cost = weight * distance
				if(i_to_i1_cost > r):
					locationX = rv/weight
					absoluteCenter.append([i, float(locationX)])
					tempDistance = distance - locationX
					flag = 'false'
			
					while True:
						i = i + 1
						if (i == nodeNumbers - 1):
							break
						if(tempDistance*weightAttributesValue[str(i)] > r):
							rv = r
							flag = 'true'
						else:
							if (i < (nodeNumbers -1)):
								rv = 0
								tempDistance += find_distance(i, i+1)
		    				
						if(flag == 'true'):
							break
				
		    				
				else:
					rv = rv - i_to_i1_cost
					i = i + 1
				if(i == nodeNumbers -1):
					if(rv < r and rv != 0):
						absoluteCenter.append([i, 0])
					break
			
			tFeasible = len(absoluteCenter)	
			return len(absoluteCenter), r

		def approxiPMedian(optCost):
			p_median_cost = 0
			global absoluteCenter, weightAttributesValue
			j = 0
			for i in range(nodeNumbers):
				if(i <=absoluteCenter[j][0]):
					cost = weightAttributesValue[str(i)] * (find_distance(i, absoluteCenter[j][0]) + absoluteCenter[j][1])
				else:
					cost = weightAttributesValue[str(i)] * (find_distance(i, absoluteCenter[j][0]) - absoluteCenter[j][1])
				if(cost > optCost and j < p-1):
					j = j+1
					cost = weightAttributesValue[str(i)] * (find_distance(i, absoluteCenter[j][0]) + absoluteCenter[j][1])
				p_median_cost += cost
			return p_median_cost

		def funcOPTPCenter():
			global approxiPmedianCost,  optPCenterCost
			funcEnumeratePairs()
			found, optPCenterCost = funcBinarySearch(sortedCostArray)
			if(found == 'true'):
				print("The Absolute Optimal Cost of ", p,"-Center(s): ", optPCenterCost)
				print("The Absolutue Center Location of ", p,"-Center(s): ", absoluteCenter)
				approxiPmedianCost = approxiPMedian(optPCenterCost)
				print("The Approximation Cost of ", p,"-median(s): ", approxiPmedianCost)
			else:
				print("P-center is not feasible!")
			

#Approxmiation Cost Calculation
		def find_weighted_midpoint_distance(start, end):
			endpoint_distance = 0
			weighted_midpoint = 0
			if(start == end):
				return endpoint_distance, weighted_midpoint
			else:
				endpoint_distance = find_distance(start, end)							
				weighted_midpoint = (weightAttributesValue[str(end)] * endpoint_distance )/(weightAttributesValue[str(start)] + weightAttributesValue[str(end)])
				return endpoint_distance, weighted_midpoint


		def calculate_1_median(start, end):
			one_median_cost = 0
			distance, weighted_midpoint = find_weighted_midpoint_distance(start, end)
	
			for i in range(start, end+1):
				if(find_distance(i,start) < weighted_midpoint):
					distance_val = weighted_midpoint - find_distance(i,start)
				else:
					distance_val = find_distance(i,start) - weighted_midpoint
				one_median_cost += distance_val * weightAttributesValue[str(i)]

			return one_median_cost
	
	
		

		def fixedCenterCost(start, center):

			weightedDistance = 0
			sumOfWeight = 0
			distanceCovered = 0

			if (start < center):
				for i in range(start, center):
					distanceCovered +=  gml_data[str(i)][str(i+1)].get('length')	
			else:
				for i in range(start, center, -1):
					distanceCovered +=  gml_data[str(i-1)][str(i)].get('length')
	
			weightedDistance =  weightAttributesValue[str(start)] * distanceCovered
		
			return weightedDistance	 



		

		def approxiCenter():
			aprroxiCost = 0
			aprroxiMaxCost = 0
			aprroxiRightCost = 1000000
			aprroxiLeftCost = 10000
			start = 0
			end = 0
			nodeIds = FID
			for i in range (0, p):
				flag[nodeIds[i]] = 1
			for i in range(0, nodeNumbers):
				if(flag[i] == 0):
					for j in range(i, nodeNumbers):
						if (flag[j] == 1):
							aprroxiRightCost = fixedCenterCost( i, j)
							#print("####For i,j = ", i,j, "aprroxiRightCost = ", aprroxiRightCost)
							break

					for j in range(i,-1,-1):
						if (flag[j] == 1):
							aprroxiLeftCost = fixedCenterCost( i, j)
							#print("For i,j = ", i,j, "aprroxiLeftCost = ", aprroxiLeftCost)
							break

					aprroxiCost = min(aprroxiRightCost, aprroxiLeftCost)
					if (aprroxiCost > aprroxiMaxCost):
						aprroxiMaxCost = aprroxiCost
			return aprroxiMaxCost


		def findExactCenterLoc():
			if(p==1):
				pCenterLoc[p-1][0], pCenterLoc[p-1][1] = get_dominating_pair(0, nodeNumbers-1)
			
			for i in range(len(pCenterLoc)):
				distance, weighted_distance = find_weighted_midpoint_distance(pCenterLoc[i][0], pCenterLoc[i][1])
				print ("distance ", distance, " weighted_midpoint ", weighted_distance)
				pExactLoc[i] = [str(weighted_distance) + " units right of Node ID: " + str(pCenterLoc[i][0])]


		optPMedianCost = optPMedian()
		print("The Optimal Cost of ", p,"-median(s): ", optPMedianCost)
		
		print("Optimal Node ID(s) for ", p, "-median(s): ", FID)

		funcOPTPCenter()
		#print("The Optimal Cost of ", p,"-Center(s): ", optPCenterCost)
		
		#findExactCenterLoc()
		#print("The Exact locations of ", p,"-Center(s) are as follows:")	
		#print (pExactLoc)

		

		approxiPCenterCost = approxiCenter()
		print("The Approximation Cost of ", p,"-center(s): ", approxiPCenterCost)

	csvFile = open('approxiRatio.csv','a')
	csvFile.write(("   %s,   %d,   %d,   %0.2f,   %0.2f,   %0.2f,  %0.2f\n" % (file_name, nodeNumbers, p, optPMedianCost, approxiPmedianCost, optPCenterCost, approxiPCenterCost )))

	csvFile.close()
	


except IOError:
	print("IOError: There is no file named: ", file_name)
	sys.exit(1)






















