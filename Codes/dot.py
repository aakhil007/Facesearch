import numpy as np
import csv
import os
ap=np.genfromtxt('meanv.csv',delimiter=',')
nclusters= ap.shape[0]
doto=np.zeros((nclusters,nclusters))
for i in range(nclusters):
	doto[i] = ap.dot(ap[i])
	for j in range(ap.shape[0]):
		if doto[i][j] > 0.60:
			doto[i][j] = int(1)
		else: 
			doto[i][j] = int(0)  	
x =  np.where(doto > 0.6) #0.6 is threshold	
writer=csv.writer(open('now2.csv','w'))
writer.writerows(doto)

print "Here are the relted clusted"
for i in range(len(x[0])):
	print "track", x[0][i] , "& track", x[1][i]
