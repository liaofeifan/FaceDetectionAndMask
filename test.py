import cv2
import numpy as np

def change(dic):
    for i,j in dic.iteritems():
        j = "bfs"

if __name__ == "__main__":
    dic = {"f": [1, ()], "c": [3, 4]}
    count = 0
    change(dic)





    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)


    while True:
        ret, frame = cap.read()

        print frame.shape

        roi_frame = frame[100:150, 100:150]
        white_mask = np.zeros((50 + 2, 50 + 2), np.uint8)

        cv2.floodFill(roi_frame, white_mask, (0, 0), (255,255,255))

        cv2.rectangle(frame, (100, 100), (200, 200), (0,0,255))

        roi_frame = cv2.bitwise_not(roi_frame)

        frame[100:150, 100:150] = roi_frame


        cv2.imshow("test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()