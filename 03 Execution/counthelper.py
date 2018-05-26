import cv2
import numpy as np
from time import sleep
    

def _count_render(obj_hist, count, frame, info, k_hist = 3):
    '''
    obj_hist -> object appeared in k-historial frame
    count -> global count of object appeared in all historial frame
    frame -> image
    info -> [{"xmin":xmin,"ymin":ymin,"xmax":xmax,"ymax":ymax,"label":label,"confidence":confidence}, ...]
    k_hist -> how long to remember objects (default 3 frames)
    '''
    h, w, d = frame.shape
    
    # render check area
    delta_h = h//10
    checkarea = ((0, 6 * delta_h), (w, 7 * delta_h))
    frame = cv2.rectangle(frame,checkarea[0], checkarea[1], (200, 0, 0), 1)
    
    # render centroid of object
    cur_objs = []
    for obj in info:
        centroid = ((obj["xmin"]+obj["xmax"])//2, (obj["ymin"]+obj["ymax"])//2)
        
        if in_checkarea(centroid, checkarea):
            cur_objs.append(centroid)  # record the object into cur_objs
            frame = cv2.circle(frame, centroid, 10, (0,0,255), -1)
            if not in_hist(centroid, obj_hist):
                count += 1
        else:
            frame = cv2.circle(frame, centroid, 4, (0,255,0), -1)
            
    # record captured object
    if len(obj_hist) >= k_hist:
        obj_hist.pop()
    obj_hist.appendleft(cur_objs)
            
    # render global count
    text = 'count: {}'.format(count)
    frame = cv2.putText(frame, text, (20, 50 ), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            
    return obj_hist, count, frame, info
    
    
def in_checkarea(centroid, checkarea):
    checkarea_lt, checkarea_rb = checkarea[0], checkarea[1]
    if checkarea_lt[0]<centroid[0] and centroid[0]<checkarea_rb[0] and checkarea_lt[1]<centroid[1] and centroid[1]<checkarea_rb[1]:
        return True
        
def in_hist(centroid, obj_hist, th_dist=(20,20)):
    in_history = False
    print(f'current obj: {centroid}')
    for h_id, objs in enumerate(obj_hist):
        print(f'{h_id+1}-history objects= {objs}')
        for o in objs:
            dist = (o[0] - centroid[0])**2 + (o[0] - centroid[0])**2
            th = (th_dist[0]**2+th_dist[1]**2) * (h_id+1)
            print(f'dist={dist} vs {th}=threshold')
            if dist < th:
                in_history = True
                print('inner threshold')
            else:
                print('uppon threshold')
    if in_history:
        print('same object')
    else:
        print('not same object')
    #input()
    return in_history
            
    