import cv2
import numpy as np

def change(dic):
    for i,j in dic.iteritems():
        j = "bfs"

def is_click(frame, extreme_top):
    frame_height, frame_width = frame.shape[:2]

    click_coor = [frame_height - 50, frame_height, 0, 50]

    if (extreme_top[0] > click_coor[3] and extreme_top[0] < click_coor[2]) and (extreme_top[1] > click_coor[0] and extreme_top[1] < click_coor[1]):
        return True

    return False

if __name__ == "__main__":
    dic = {"f": [1,2], "c": [3,4]}
    d = {"a": 3, "b": 2}
    count = 0
    change(dic)
    l = []



    # print d.keys()[d.values().index(3)]
    print (4 or 2 in d.values())





    # cap = cv2.VideoCapture(0)
    # cap.set(3, 320)
    # cap.set(4, 240)
    #
    #
    # while True:
    #     ret, frame = cap.read()
    #
    #     print frame.shape
    #
    #     roi_frame = frame[100:150, 100:150]
    #     white_mask = np.zeros((50 + 2, 50 + 2), np.uint8)
    #
    #     cv2.floodFill(roi_frame, white_mask, (0, 0), (255,255,255))
    #
    #     cv2.rectangle(frame, (100, 100), (200, 200), (0,0,255))
    #
    #     roi_frame = cv2.bitwise_not(roi_frame)
    #
    #     frame[100:150, 100:150] = roi_frame
    #
    #
    #     cv2.imshow("test", frame)
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()