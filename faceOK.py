import cv2
import face_recognition
import numpy as np
import os


path = "/Users/antonytran/Documents/code/face_idNew/img_input"
imageIp = []
nameIp = []
myImgIputList = os.listdir(path)
print(myImgIputList)

for II in myImgIputList :
    curImg = cv2.imread(f"{path}/{II}")
    imageIp.append(curImg)
    nameIp.append(os.path.splitext(II)[0])
print(nameIp)

def encodingIp(imageIp):
    encodeIp = []
    for img in imageIp :
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeIp.append(encode)
    return encodeIp

encodeKnown = encodingIp(imageIp)
print("Encode completed")

cam = cv2.VideoCapture(0)

while True:

    raw, img = cam.read()
    imgSmall = cv2.resize(img,(0,0),fx=0.25,fy=0.25)
    imgSmall = cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

    faceLocCur = face_recognition.face_locations(imgSmall)
    encodeLocCur = face_recognition.face_encodings(imgSmall,faceLocCur)
    
    for encodeFace, faceLoc in zip(encodeLocCur,faceLocCur):
        matched = face_recognition.compare_faces(encodeKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeKnown, encodeFace)
        # print(matched)
        # name = "Unknown"
        matchedIndex = np.argmin(faceDis)
        # print(matchedIndex)
        # if matched[matchedIndex]:
        if True in matched:
            name = nameIp[matchedIndex].upper()
            # print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4-50,x2*4+20,y2*4+50,x1*4-20
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        else : 
            name = "Unknown"
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4-50,x2*4+20,y2*4+50,x1*4-20
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow("cam",img)
    if cv2.waitKey(1)& 0xFF == ord("q"):
        break 

cam.release()
cv2.destroyAllWindows()



# if __name__ == "__main__":
    # main()
