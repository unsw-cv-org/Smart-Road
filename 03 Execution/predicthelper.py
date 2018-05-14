import cv2
import numpy as np
import time

def _func_helper(func, *img):
    if func==None:
        return [{'label':'something', 
                 'confidence':0.00, 
                 'topleft':(100,100), 
                 'bottomright':(200,200)}]
    else:
        return func(img)
        
def _img_render(func, img):
    results = _func_helper(func, img)
    for result in results:
        tl, br = result['topleft'], result['bottomright']
        label = result['label']
        confidence = result['confidence']
        text = '{}: {}'.format(label, confidence )
        color = tuple(255 * np.random.rand(3))
        img = cv2.rectangle(img, tl, br, color, 5)
        img = cv2.putText(img, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    return img
            
# api                       
def predict_with_camera(func, frame_width=800, frame_height=600): 
    '''
        args: func - indicate the name of detection function.
              The function func takes an argument img and return a list detected object information, e.g.
                [{'label':'something', 
                  'confidence':0.00, 
                  'topleft':(100,100), 
                  'bottomright':(200,200)},...]
              frame_width, frame_height - indicate height and  width of output frame (can be customized if want)    
    '''     
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while True:
        stime = time.time()
        ret, frame = capture.read()
        if ret: 
            frame = _img_render(func, frame)
            delta_time = time.time() - stime
            delta_time = delta_time if delta_time!=0.0 else 0.0001 
            print('FPS {:.1f}'.format(1.0 / delta_time ))
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    capture.release()
    cv2.destroyAllWindows()
    

def predict_with_video(func, video_file):
    '''
        args: func - indicate the name of detection function.
              The function func takes an argument img and return a list detected object information, e.g.
                [{'label':'something', 
                  'confidence':0.00, 
                  'topleft':(100,100), 
                  'bottomright':(200,200)},...]
              video_file - indicate the name of video    
    '''   
    cap = cv2.VideoCapture(video_file)

    while(cap.isOpened()):
        stime = time.time()
        ret, frame = cap.read()
        if ret: 
            frame = _img_render(func, frame) 
            delta_time = time.time() - stime
            delta_time = delta_time if delta_time!=0.0 else 0.0001 
            print('FPS {:.1f}'.format(1.0 / delta_time ))
            cv2.imshow('frame', frame)
        else:
            print('no ret')
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    print('cap.release')
    cv2.destroyAllWindows()
    print('cv2.destroyAllWindows')
    
    
if __name__=='__main__':
    predict_with_camera(None)


