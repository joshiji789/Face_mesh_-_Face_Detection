# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 22:49:06 2022

@author: joshi
"""

import cv2
import mediapipe as mp
import time
#%%
class FaceMeshDetector():
    def __init__(self,static_mode=False,max_faces=2,refine_ladmarks=False,mindetectioncon=0.5,mintrackingcon=0.5):
        self.static_mode=static_mode
        self.max_faces=max_faces
        self.refine_ladmarks=refine_ladmarks
        self.mindetectioncon=mindetectioncon
        self.mintrackingcon=mintrackingcon
        
        
    
        self.mpDraw=mp.solutions.drawing_utils
        self.mpFaceMesh=mp.solutions.face_mesh
        
        self.face_mesh=self.mpFaceMesh.FaceMesh(self.static_mode,self.max_faces,self.refine_ladmarks,
                                               self.mindetectioncon,self.mintrackingcon)
        
        self.drawSpec=self.mpDraw.DrawingSpec(thickness=1,circle_radius=2)

    def findfacemesh(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.face_mesh.process(imgRGB)
        
        faces=[]
        
        if self.results.multi_face_landmarks: 
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms,self.mpFaceMesh.FACEMESH_CONTOURS,
                                           self.drawSpec,self.drawSpec)
                
                face=[]
                for id,lm in enumerate(faceLms.landmark):
               
                    ih,iw,ic=img.shape
                    
                    x,y=int(lm.x*iw),int(lm.y*ih)
                    #print(id,x,y)
                    face.append([x,y])
                    
                faces.append(face)
            
        return img,faces
                

def main():
    
    cap=cv2.VideoCapture(0)
    pTime=0
    detector=FaceMeshDetector()
    while True:
        success,img=cap.read()
        img,faces=detector.findfacemesh(img)
        
        if len(faces)!=0:
            print(len(faces))
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
    
        cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,
                    3,(0,255,0),3)
    
        cv2.imshow("Image",img)
        cv2.waitKey(1)
    
    
if __name__=="__main__":
    main()