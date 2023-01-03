from tracking.bytetrack import BYTETracker
from tracking.utils import *
import argparse
import os
import sys
import random
from pathlib import Path
import time
import torch
import json

"""
python tracking_demo.py --source "C:/Users/jeanphilippe.cabay/Downloads/avant_peage_saint_maurice_1671609598.mp4" --view-img --agnostic-nms
"""

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def run(opt):

    # load tracker
    class BYTETrackerArgs:
        track_thresh: float = opt.track_thresh
        track_buffer: int = opt.track_buffer
        match_thresh: float = opt.match_thresh
        aspect_ratio_thresh: float = opt.aspect_ratio_thresh
        min_box_area: float = opt.min_box_area
        mot20: bool = False
    byte_tracker = BYTETracker(BYTETrackerArgs())

    # load annotators
    chars = '0123456789ABCDEF'
    COLORS = list(map(Color.from_hex_string, ['#'+''.join(random.sample(chars,6)) for i in range(85)]))
    THICKNESS = opt.line_thickness
    annotator = BaseAnnotator(COLORS, THICKNESS)

    background_color = Color.from_hex_string("#850101") #Brownish
    text_color = Color.from_hex_string("#FFFFFF") #White
    text_thickness = 1

    textannotator = TextAnnotator(background_color, text_color, text_thickness)

    # start video stream
    cap = cv2.VideoCapture(opt.source)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # load model
    if opt.device == '':
        device = 'cpu'
    else:
        device = opt.device
    model = torch.hub.load('ultralytics/yolov5', 'custom', opt.weights, device=device)

    model.conf = opt.conf_thres  # NMS confidence threshold
    model.iou = opt.iou_thres  # NMS IoU threshold
    model.agnostic = opt.agnostic_nms  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.classes = opt.classes  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    model.max_det = opt.max_det  # maximum number of detections per image
    model.amp = opt.half  # Automatic Mixed Precision (AMP) inference

    # start recording
    if opt.save:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    stride_cnt = 0
    frame_cnt = 0

    while 1:
        # Get frameset of color
        ret, frame = cap.read()
        if ret:
            stride_cnt = (stride_cnt + 1) % opt.vid_stride
            if stride_cnt == 1:
                continue
            frame_cnt += 1
            # start = time.time()
            # r_frame = resizeImage(frame, opt.imgsz)
            r_frame = frame
            rgb_frame = cv2.cvtColor(r_frame, cv2.COLOR_BGR2RGB)

            results = model(rgb_frame, size=opt.imgsz)

            detections = Detection.from_results(
                pred=results.pred[0].cpu().numpy(),
                names=model.names)

            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=r_frame.shape,
                img_size=r_frame.shape)

            detections = match_detections_with_tracks(detections=detections, tracks=tracks)

            annotated_frame = annotator.annotate(r_frame, detections)
            annotated_frame = textannotator.annotate(annotated_frame, detections)
            # fps = 1./(time.time()-start)
            # annotated_frame = draw_text(
            #     image=annotated_frame,
            #     anchor=Point(x=20, y= 20),
            #     text="{} fps".format(fps),
            #     color=text_color,
            #     thickness=text_thickness)
            if opt.save:
                out.write(annotated_frame)
            if opt.view_img:
                cv2.imshow("frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    if opt.save:
        out.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    return


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT/'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save', action='store_true', help='save video')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--half', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--track-thresh', type=float, default=0.25, help='tracking threshold')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--match-thresh', type=float, default=0.8, help='matching threshold')
    parser.add_argument('--aspect-ratio-thresh', type=float, default=3., help='aspect ratio threshold')
    parser.add_argument('--min-box-area', type=float, default=1., help='minimum box area')
    opt = parser.parse_args()
    print(vars(opt))
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
