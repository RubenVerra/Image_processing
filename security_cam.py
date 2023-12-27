import cv2 as cv
import numpy as np
import time
import datetime
import emailsender
from email.message import EmailMessage
import ssl
import smtplib



backSub = cv.createBackgroundSubtractorKNN()
#backSub = cv2.createBackgroundSubtractorMOG2()

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_fullbody.xml")

##############################################email stuff
email_sender = 'ferrecreemers@gmail.com'
email_password = 'ejim sbuo jhvp wxrd'

email_receiver = 'ferrecreemers@gmail.com'
  
subject = "Intruder Alert"

body = """
interuder in home here's the video:
"""

em = EmailMessage()
em['From'] = email_sender
em['To'] = email_receiver
em['subject'] = subject

em.set_content(body)

context = ssl.create_default_context()

###############################################


detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv.VideoWriter_fourcc(*"mp4v")

while True:
    _, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    fgMask = backSub.apply(frame)
    kernel = np.ones((5,5), np.uint8)
    
    fgMask = cv.erode(fgMask, kernel, iterations=2)
    fgMask = cv.dilate(fgMask, kernel, iterations=2)
    
    
    if len(faces) + len(bodies) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv.VideoWriter(
                f"{current_time}.mp4", fourcc, 20, frame_size)
            print("Started Recording!")
           
           #######################################################################
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                smtp.login(email_sender, email_password)
                smtp.sendmail(email_sender, email_receiver, em.as_string())
           #######################################################################
    
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print('Stop Recording!')
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)

    for (x, y, width, height) in faces:
        cv.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)
    
    
    fgMask[np.abs(fgMask) < 250 ] = 0
    
    cv.imshow("FG Mask", fgMask)
    cv.imshow("Camera", frame)

    if cv.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv.destroyAllWindows()