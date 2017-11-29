#This is code will take video file as an input and gives three CSV files mentioned below: 
#track.csv: This contains number of images in each track
#face.csv: In this we'll store each detected face as a row major matrix in each row 
#trackDetails.csv: This contains starting and ending frame numbers of each detected. track
#All the deteced faces in one track will be stored in one separate folder.
#Import the OpenCV and dlib libraries
import cv2
import dlib
import csv
import os
import threading
import time
import numpy as np

#Initialize a face cascade using the frontal face haar cascade provided with
#the OpenCV library
#Make sure that you copy this file from the opencv project to the root of this
#project folder
#faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
#writer=csv.writer(open('facevalues.csv','w'))

#The deisred output width and height
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600

def doRecognizePerson(faceNames, fid,frameCount,trackDetails):
    #time.sleep(2)
    global facedata
    
    faceNames[ fid ] = "Person " + str(fid)
    trackDetails.append([frameCount,frameCount])




def detectAndTrackMultipleFaces():
    #Open the first webcame device
    capture = cv2.VideoCapture('interview.mp4')

    #Create two opencv named windows
    cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

    #Position the windows next to eachother
    cv2.moveWindow("base-image",0,100)
    cv2.moveWindow("result-image",400,100)

    #Start the window thread for the two windows we are using
    #cv2.startWindowThread()

    #The color of the rectangle we draw around the face
    rectangleColor = (0,165,255)

    #variables holding the current frame number and the current faceid
    frameCounter = 0
    currentFaceID = 0

    #Variables holding the correlation trackers and the name per faceid
    faceTrackers = {}
    faceNames = {}
    global trackDetails
    trackDetails=[]
    
    global facedata
    facedata=[]
    
    bnv=0

    try:
        while True:
            #Retrieve the latest image from the webcam
            rc,fullSizeBaseImage = capture.read()
            
            if rc == False:
                break
            

            #Resize the image to 320x240
            #baseImage = cv2.resize( fullSizeBaseImage, (640 ,310 ))
            baseImage=fullSizeBaseImage

            #Check if a key was pressed and if it was Q, then break
            #from the infinite loop
            pressedKey = cv2.waitKey(2)
            if pressedKey & 0xFF == ord('q'):
                for fid in faceTrackers.keys():
                    trackDetails[fid][1]=frameCounter
                break



            #Result image is the image we will show the user, which is a
            #combination of the original image from the webcam and the
            #overlayed rectangle for the largest face
            resultImage = baseImage.copy()




            #STEPS:
            # * Update all trackers and remove the ones that are not 
            #   relevant anymore
            # * Every 10 frames:
            #       + Use face detection on the current frame and look
            #         for faces. 
            #       + For each found face, check if centerpoint is within
            #         existing tracked box. If so, nothing to do
            #       + If centerpoint is NOT in existing tracked box, then
            #         we add a new tracker with a new face-id


            #Increase the framecounter
            frameCounter += 1 



            #Update all the trackers and remove the ones for which the update
            #indicated the quality was not good enough
            fidsToDelete = []
            for fid in faceTrackers.keys():
                trackingQuality = faceTrackers[ fid ].update( baseImage )

                #If the tracking quality is good enough, we must delete
                #this tracker
                if trackingQuality < 7:
                    fidsToDelete.append( fid )

            for fid in fidsToDelete:
                print("Removing fid " + str(fid) + " from list of trackers")
                faceTrackers.pop( fid , None )
                trackDetails[fid][1]=frameCounter-1




            #Every 10 frames, we will have to determine which faces
            #are present in the frame
            if (frameCounter % 10) == 0:



                #For the face detection, we need to make use of a gray
                #colored image so we will convert the baseImage to a
                #gray-based image
                gray = cv2.cvtColor(baseImage, cv2.COLOR_BGR2GRAY)
                #Now use the haar cascade detector to find all faces
                #in the image
                
                #faces = faceCascade.detectMultiScale(gray, 1.3, 5)
                faces=detector(gray)



                #Loop over all faces and check if the area for this
                #face is the largest so far
                #We need to convert it to int here because of the
                #requirement of the dlib tracker. If we omit the cast to
                #int here, you will get cast errors since the detector
                #returns numpy.int32 and the tracker requires an int
                #for (_x,_y,_w,_h) in faces:
                for face in faces:
                    x = int(dlib.rectangle.left(face))
                    y = int(dlib.rectangle.top(face))
                    w = int(dlib.rectangle.right(face))-int(dlib.rectangle.left(face))
                    h = int(dlib.rectangle.bottom(face))-int(dlib.rectangle.top(face))
                    """x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)"""


                    #calculate the centerpoint
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h


                    if x <0:
                        w=x+w
                        x=0
                    if y <0:
                        h=h+y
                        y=0


                    #Variable holding information which faceid we 
                    #matched with
                    matchedFid = None

                    #Now loop over all the trackers and check if the 
                    #centerpoint of the face is within the box of a 
                    #tracker
                    for fid in faceTrackers.keys():
                        tracked_position =  faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())


                        #calculate the centerpoint
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        #check if the centerpoint of the face is within the 
                        #rectangleof a tracker region. Also, the centerpoint
                        #of the tracker region must be within the region 
                        #detected as a face. If both of these conditions hold
                        #we have a match
                        if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
                             ( t_y <= y_bar   <= (t_y + t_h)) and 
                             ( x   <= t_x_bar <= (x   + w  )) and 
                             ( y   <= t_y_bar <= (y   + h  ))):
                            matchedFid = fid


                    #If no matched fid, then we have to create a new tracker
                    if matchedFid is None:

                        print("Creating new tracker " + str(currentFaceID))

                        #Create and store the tracker 
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(baseImage,
                                            dlib.rectangle( x-(w/5),
                                                            y-(h/5),
                                                            x+w+(w/5),
                                                            y+h+(h/5)))

                        faceTrackers[ currentFaceID ] = tracker
                        
                        tempimg=cv2.resize(gray[y:y+h,x:x+w],(64,64),interpolation=cv2.INTER_LINEAR)
                        facedata.append(np.asarray(tempimg).flatten().reshape(1,4096))
                        
                        os.makedirs('./rec/'+'track'+str(currentFaceID))

                        #Start a new thread that is used to simulate 
                        #face recognition. This is not yet implemented in this
                        #version :)
                        #t = threading.Thread( target = doRecognizePerson ,
                                               #args=(faceNames, currentFaceID,frameCounter,trackDetails))
                        #t.start()
                        faceNames[ currentFaceID ] = "Person " + str(currentFaceID)
                        trackDetails.append([frameCounter,frameCounter])


                        #Increase the currentFaceID counter
                        currentFaceID += 1




            #Now loop over all the trackers we have and draw the rectangle
            #around the detected faces. If we 'know' the name for this person
            #(i.e. the recognition thread is finished), we print the name
            #of the person, otherwise the message indicating we are detecting
            #the name of the person
            for fid in faceTrackers.keys():
                tracked_position =  faceTrackers[fid].get_position()

                t_x = int(tracked_position.left())
                t_y = int(tracked_position.top())
                t_w = int(tracked_position.width())
                t_h = int(tracked_position.height())
                
                if t_x <0:
                    t_w=t_x+t_w
                    t_x=0
                if t_y <0:
                    t_h=t_h+t_y
                    t_y=0
                
                

                cv2.rectangle(resultImage, (t_x, t_y),
                                        (t_x + t_w , t_y + t_h),
                                        rectangleColor ,2)


                if fid in faceNames.keys():
                    
                    #facedata = np.vstack([facedata,[np.asarray(resize).flatten()]])
                    if (t_x-(t_w/5)>0) and (t_y-(t_h/5)>0):
                        t_x=t_x-(t_w/5)
                        t_y=t_y-(t_h/5)
                        t_w=t_w+(2/5)*t_w
                        t_h=t_h+(2/5)*t_h
                    gray1=np.copy(gray[t_y:t_y+t_h, t_x:t_x+t_w])
                    face_verification=detector(gray1)
                    if(len(face_verification)==1):
                        x1 = int(dlib.rectangle.left(face_verification[0]))
                        y1 = int(dlib.rectangle.top(face_verification[0]))
                        w1 = int(dlib.rectangle.right(face_verification[0]))-int(dlib.rectangle.left(face_verification[0]))
                        h1 = int(dlib.rectangle.bottom(face_verification[0]))-int(dlib.rectangle.top(face_verification[0]))
                        
                        if x1 <0:
                            w1=x1+w1
                            x1=0
                        if y1 <0:
                            h1=h1+y1
                            y1=0

                        resize=cv2.resize(gray1[y1:y1+h1,x1:x1+w1],(64,64),interpolation=cv2.INTER_LINEAR)
                        facedata[fid] = np.vstack([facedata[fid],[np.asarray(resize).flatten()]])

                        cv2.imwrite('./rec/'+'track'+str(fid)+'/'+str(bnv)+'.png',resize)
                        #cv2.imwrite('./copy/'+str(bnv)+'.png',resize)
                        bnv = bnv+1
                    cv2.putText(resultImage, faceNames[fid] , 
                                (int(t_x + t_w/2), int(t_y)), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)
                else:
                    cv2.putText(resultImage, "Detecting..." , 
                                (int(t_x + t_w/2), int(t_y)), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)






            #Since we want to show something larger on the screen than the
            #original 320x240, we resize the image again
            #
            #Note that it would also be possible to keep the large version
            #of the baseimage and make the result image a copy of this large
            #base image and use the scaling factor to draw the rectangle
            #at the right coordinates.
            largeResult = cv2.resize(resultImage,
                                     (OUTPUT_SIZE_WIDTH,OUTPUT_SIZE_HEIGHT))
            
            

            #Finally, we want to show the images on the screen
            cv2.imshow("base-image", baseImage)
            #cv2.imshow("result-image", largeResult)
            cv2.imshow("result-image", resultImage)
            #print len(facedata)




    #To ensure we can also deal with the user pressing Ctrl-C in the console
    #we have to check for the KeyboardInterrupt exception and break out of
    #the main loop
    except KeyboardInterrupt as e:
        pass
        
    finally:

        #Destroy any OpenCV windows and exit the application
        """if len(facedata) !=0:
            facedata=np.delete(facedata,0,0)
            np.savetxt("framerep5.csv",facedata,delimiter=",")"""
        track=[x.shape[0] for x in facedata]
        for i in xrange(len(track)):
            if i!=0:
                track[i] += track[i-1]
                
        np.savetxt("track.csv",np.array(track),delimiter=",")
        np.savetxt("face.csv",np.concatenate(tuple(facedata),axis=0),delimiter=",")
        
        np.savetxt("trackDetails.csv",trackDetails,delimiter=",")

        cv2.destroyAllWindows()
        exit(0)


if __name__ == '__main__':
    detectAndTrackMultipleFaces()
