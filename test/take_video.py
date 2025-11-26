import cv2

#Open default camera
cap = cv2.VideoCapture(0)

#Set the resolution 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20


#Video formate: use mp4v for .mp4 or XVID for .avi

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret,frame = cap.read()
    if not ret:
        print("can't receive frame (stream end?). Exiting ...")
        break

    #Write the frame to the file
    out.write(frame)

    #Show live video
    cv2.imshow('Recording', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as output.mp4")
