#This code includes Spherical Clustering and Hashing
#In This code number of clusters is equal to number of tracks 
import numpy as np
import csv
import os
import cv2
#For converting into unit vectors, we defined convunit function.
def convunit(array):
	nimg= array.shape[0]
	l2norm = np.sqrt((array*array).sum(axis=1))
	array=array/l2norm.reshape(nimg,1)
	return array

array=np.genfromtxt('face.csv',delimiter=',') 

#face.csv file contains all the detected faces' data and this was gnerated by first code.
#In that face.csv, each 64*64 detected face is saved as row major matrix. Each row contains one face.
#We stored all the data in face.csv in the numpy array "array" 

array2 = np.copy(array) ##Copied our dataset, to use further
track=np.genfromtxt('track.csv',delimiter=',') #Track foldering
nimg= array.shape[0] #Number of images/face is equal to number of rows in face.csv
array= array-np.array([np.mean(array,axis=0),]*nimg) #We are making vectors as zero mean vectors
array=convunit(array) #Converted each row as an unit vector.
nclus = track.shape[0] #This should be equal to number of tracks
list1=[]
b = 0
for i in range(nclus):
	list1.append(range(b,int(track[i])))
	b = int(track[i])

#list1.append(range((nclus-1)*(nimg/nclus),nimg))
#Initilisation using folders
print array
meanv=np.zeros((nclus,array.shape[1]))
#meanv is the numpy array which contains mean vectors (as a row) of all clusters
while 1:
	prevmeanv=np.copy(meanv)

	for i in range(nclus):
		if len(list1[i])==0:
			meanv[i]=meanv[i]
		else:
			for j in range(len(list1[i])):
				meanv[i]+=array[list1[i][j]]
			meanv[i] = meanv[i]/len(list1[i])
	print meanv
	meanv=convunit(meanv)
	#print meanv 
	checkout=(meanv*prevmeanv).sum(axis=1)
	#Took the dot product present mean vectors with previous mean vectors.
	mincheckout=np.amin(checkout)	
	#mincheckout is nothing but minimum of all dot producs values
	for i in range(nclus):
		list1[i][:]=[]
	
	for i in range(nimg):
		tarray=[array[i],]*nclus
		tsum=(tarray*meanv).sum(axis=1)
		tmax=np.argmax(tsum)
		list1[tmax].append(i)

	#print list1
	#If mincheckout is greater than our threshold, we'll stop our clustering
	if mincheckout >0.99:
		break
	print mincheckout

#print list1 #List1 is clusters list :)
print 'loop completed'

writer=csv.writer(open('meanv.csv','w')) #meanv_rand.csv will be the output file which contains "meanv" numpy array
writer.writerows(meanv) 
#Spherical K-means done with "nclus" clusters and the mean vectors are stored in "meanv_rand.csv" file

#Hashing 
hashed=np.zeros((nimg,nclus)) #Created a numpy array size of nclus * nclus
#Dot product of each img vector with each mean vector. So for one image we should get nclus values
for i in range(nimg):
	hashed[i] = meanv.dot(array[i])
print "Hashing done"

writer=csv.writer(open('hashed.csv','w')) #Stored hashed vectors in 'hashed_rand.csv' file
writer.writerows(hashed)
#print hashed
#finding the mean of each track using hashed faces
#Let's call that hash_mean
hash_mean = np.zeros((nclus,nclus))
b = 0
for i in range(nclus):
	hash_mean[i] = [sum(x) for x in zip(*hashed[b:int(track[i])])]
	hash_mean[i] = hash_mean[i] / (track[i]-b) 
	b = int(track[i])
print "Hashed_mean_done"
#print hash_mean
writer=csv.writer(open('hash_mean.csv','w')) #Stored mean of each track obtained using hashed vectors of each face
writer.writerows(hash_mean)
writer=csv.writer(open('hash_cov.csv','w'))
cov_list = []
b = 0 
for i in range(nclus):
	cov_list.append(np.cov(np.transpose(hashed[b:int(track[i])])))
	writer.writerows(cov_list[i])
 	b = int(track[i])
#Covariance_done and stored in hash_cov.csv file

#print cov_list[0]
fold=0
for i in list1:
	os.makedirs('./clusters/'+'cluster'+str(fold))
	for j in i:
		cv2.imwrite('./clusters/'+'cluster'+str(fold)+'/'+str(j)+'.png',array2[j].reshape(64,64))
	fold +=1
