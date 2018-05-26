import cv2
import numpy as np
import time
import collections
import counthelper


def _func_helper(model, img):
    if model==None:
        raise RuntimeError("no model found")
    else:
        return model.predict(img)

def _img_render(model, img):
    try:
        h, w, _ = img.shape
        img = img[h//3:, 0:w//2]
        boxes = _func_helper(model, img)
    except RuntimeError:
        exit(1)
    red = (255,0,0)
    green = (0,255,0)
    yellow = (255,255,33)
    colors = [red,green,yellow]
    info = []
    for box in boxes:

        image_h, image_w, _ = img.shape
        xmin = int(box.xmin * image_w)
        ymin = int(box.ymin * image_h)
        xmax = int(box.xmax * image_w)
        ymax = int(box.ymax * image_h)
        #tl, br = result['topleft'], result['bottomright']
        label = model.labels[box.get_label()]
        confidence = box.get_score()


        text = '{}: {:.2f}'.format(label, confidence )


        info.append({"xmin":xmin,"ymin":ymin,"xmax":xmax,"ymax":ymax,"label":label,"confidence":confidence})

        text = '{}: {:.2f}'.format(label, confidence)

        if label == "car":
            color = colors[2]
        else:
            color = colors[1]

        img = cv2.rectangle(img,(xmin,ymin), (xmax,ymax), color, 1)
        img = cv2.putText(img, text, (xmin, ymin ), cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)

    return img, info
    



            
# api                       
def predict_with_camera(model, frame_width=800, frame_height=600):
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

            """
            info is a list of dict
            each dict contain information of a bounding box
            dict structure:
            {"xmin":xmin,   -> int
            "ymin":ymin,    -> int
            "xmax":xmax,    -> int
            "ymax":ymax,    -> int
            "label":label,  -> string
            "confidence":confidence -> float}
            
            you need to do extantion by this info
            
            """

            frame,info = _img_render(model, frame)
            delta_time = time.time() - stime
            delta_time = delta_time if delta_time!=0.0 else 0.0001
            fps = 1.0 / delta_time
            print('FPS {:.1f}'.format( fps ))
            cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    capture.release()
    cv2.destroyAllWindows()
    

def predict_with_video(model,video_file,saved):
    print("press q to stop and save (optional)")
    '''
        args: model - indicate the name of model, with function predict().
              video_file - indicate the name of video    
    '''
    cap = cv2.VideoCapture(video_file)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_out = video_file[:-4] + '_detected' + video_file[-4:]
    video_writer = None
    state_fps = None
    cut_count = None
    cv2.namedWindow('frame', flags=0)

    '''
    # get state_fps and cut_count by first 2 seconds
    stime = time.time()
    etime = time.time()
    while(etime-stime < 5 and cap.isOpened()):
        tf1 = time.time()
        ret, frame = cap.read()
        if ret:
            frame, _ = _img_render(model, frame)
            delta_time = time.time() - tf1
            delta_time = delta_time if delta_time != 0.0 else 0.0001
            state_fps = 1.0 / delta_time
            print(f"state_fps:{state_fps}")
            cut_count = int(FPS/state_fps)-1
            cv2.imshow('frame', frame)
            etime = time.time()
        else:
            print('no ret')
            etime = time.time()

    # creat video writer if needed
    if saved:
        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'),
                               state_fps,
                               (frame_w, frame_h))
    '''
    ret, frame = cap.read()
    count = 0
    obj_hist = collections.deque()
    while(cap.isOpened() and ret):
        stime = time.time()
        ret, frame = cap.read()
        
        '''
        for _ in range(cut_count):
            ret, frame = cap.read()
        '''    
            
        if ret:
            frame ,info = _img_render(model, frame)
            
            obj_hist, count, frame, info = counthelper._count_render(obj_hist, count, frame, info)
            if video_writer:
                video_writer.write(np.uint8(frame))

            """
            info is a list of dict
            each dict contain information of a bounding box
            dict structure:
            {"xmin":xmin,   -> int
            "ymin":ymin,    -> int
            "xmax":xmax,    -> int
            "ymax":ymax,    -> int
            "label":label,  -> string
            "confidence":confidence -> float}

            you need to do extantion by this info

            """


            delta_time = time.time() - stime
            delta_time = delta_time if delta_time!=0.0 else 0.0001 
            print('FPS {:.1f}'.format(1.0 / delta_time ))
            cv2.imshow('frame', frame)
            
        else:
            print('no ret')
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    if video_writer:
        video_writer.release()
        print(f"detected file saved on: {video_out}")
    cv2.destroyAllWindows()


    



