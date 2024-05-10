import os
import cv2
import numpy as np
import onnxruntime
from pathlib import Path
from tqdm import tqdm
import urllib.request

from .detector_utils import preprocess_image


def dummy(*args, **kwargs):
    pass

def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

model_url = "https://huggingface.co/s0md3v/nudity-checker/resolve/main/detector.onnx"
classes_url = "https://huggingface.co/s0md3v/nudity-checker/resolve/main/classes"


home = Path.home()
model_folder = os.path.join(home, f".ifnude/")
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model_name = os.path.basename(model_url)
model_path = os.path.join(model_folder, model_name)
classes_path = os.path.join(model_folder, "classes")

if not os.path.exists(model_path):
    print("Downloading the detection model to", model_path)
    download(model_url, model_path)

if not os.path.exists(classes_path):
    print("Downloading the classes list to", classes_path)
    download(classes_url, classes_path)

classes = [c.strip() for c in open(classes_path).readlines() if c.strip()]

def detect(img, mode="default", min_prob=None):
    # we are loading the model on every detect() because it crashes otherwise for some reason xD
    detection_model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    if mode == "fast":
        image, scale = preprocess_image(img, min_side=480, max_side=800)
        if not min_prob:
            min_prob = 0.5
    else:
        image, scale = preprocess_image(img)
        if not min_prob:
            min_prob = 0.6

    outputs = detection_model.run(
        [s_i.name for s_i in detection_model.get_outputs()],
        {detection_model.get_inputs()[0].name: np.expand_dims(image, axis=0)},
    )

    labels = [op for op in outputs if op.dtype == "int32"][0]
    scores = [op for op in outputs if isinstance(op[0][0], np.float32)][0]
    boxes = [op for op in outputs if isinstance(op[0][0], np.ndarray)][0]

    boxes /= scale
    processed_boxes = []
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < min_prob:
            continue
        box = box.astype(int).tolist()
        label = classes[label]
        if label == "EXPOSED_BELLY":
            continue
        processed_boxes.append(
            {"box": [int(c) for c in box], "score": float(score), "label": label}
        )

    return processed_boxes

def censor(img_path, out_path=None, visualize=False, parts_to_blur=[]):
    if not out_path and not visualize:
        print(
            "No out_path passed and visualize is set to false. There is no point in running this function then."
        )
        return

    image = cv2.imread(img_path)
    boxes = detect(img_path)

    if parts_to_blur:
        boxes = [i["box"] for i in boxes if i["label"] in parts_to_blur]
    else:
        boxes = [i["box"] for i in boxes]

    for box in boxes:
        image = cv2.rectangle(
            image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), cv2.FILLED
        )

    return image


def extract_frames(inp_video_path):

    global total_frames,fps,frame_width,frame_height,count
    inp_video=inp_video = cv2.VideoCapture(inp_video_path)
    if not os.path.exists('temp'):
        try:
            os.makedirs('temp')
        except Exception as e:
            return e
        
    if inp_video is None:
        print("no input video file has been selected!!")
        return
    
    total_frames = int(inp_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = inp_video.get(cv2.CAP_PROP_FPS)
    frame_width = int(inp_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(inp_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    progress_bar=tqdm(total=total_frames,desc="Extracting frames",ncols=100)
    count=0
    if inp_video.isOpened():

        while True:
            ret, frame=inp_video.read()
            if not ret:
                break
            cv2.imwrite(os.path.join('temp', 'frame{:d}.jpg'.format(count)), frame)
            count += 1
            progress_bar.update(1)

        progress_bar.close()
        print(f"extracted {count} frames")
        inp_video.release()


def censor_video(inp_video_path, out_video_path, mode="default", min_prob=None, parts_to_blur=[]):
    
    extract_frames(inp_video_path=inp_video_path)
    p_bar=tqdm(total=total_frames,desc="detecting and censoring frames",ncols=100)
    inp_video_path=inp_video_path
    f = 0

    for i in range(total_frames):
        detection_model = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        img=os.path.join('temp',f'frame{f}.jpg')
        if mode == "fast":
            image, scale = preprocess_image(img, min_side=480, max_side=800)
            if not min_prob:
                min_prob = 0.5
        else:
            image, scale = preprocess_image(img)
            if not min_prob:
                min_prob = 0.6

        outputs = detection_model.run(
            [s_i.name for s_i in detection_model.get_outputs()],
            {detection_model.get_inputs()[0].name: np.expand_dims(image, axis=0)},
        )

        labels = [op for op in outputs if op.dtype == "int32"][0]
        scores = [op for op in outputs if isinstance(op[0][0], np.float32)][0]
        boxes = [op for op in outputs if isinstance(op[0][0], np.ndarray)][0]

        boxes /= scale
        processed_boxes = []

        frame_path=os.path.join('temp','frame{:d}.jpg'.format(i))
        
        print('frame path',frame_path)
        # Initiating Detection
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < min_prob:
                continue
            box = box.astype(int).tolist()
            label = classes[label]
            if label == "EXPOSED_BELLY":
                continue
            processed_boxes.append(
                {"box": [int(c) for c in box], "score": float(score), "label": label}
            )

        image=cv2.imread(frame_path)
        boxes=processed_boxes
        # Censoring the frame
        if parts_to_blur:
            boxes = [i["box"] for i in boxes if i["label"] in parts_to_blur]
        else:
            boxes = [i["box"] for i in boxes]

        for box in boxes:
            image = cv2.rectangle(
                image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), cv2.FILLED
            )

        cv2.imwrite(frame_path,image)
        p_bar.update(1)
        f += 1

    p_bar.close()

    

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_width,frame_height))

    for i in range(total_frames):
        frame_path=os.path.join('temp', 'frame{:d}.jpg'.format(i))
        frame=cv2.imread(frame_path)
        out.write(frame)

    out.release()

    if out_video_path is not None:
        print("file saved as {}".format(out_video_path))
    else:
        print("Error: No output video file specified")

    
    print("cleaning temporary files")
    try:
        os.removedirs('temp')
    except Exception as e:
        return e
