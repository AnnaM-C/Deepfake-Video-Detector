"Frame extraction code adapted from https://github.com/SCLBD/DeepfakeBench/blob/main/preprocessing/preprocess.py"

import argparse
import cv2
import dlib
import numpy as np
from tqdm import tqdm
from skimage import transform as trans
import os
from imutils import face_utils
from PIL import Image
import torchvision.transforms.functional as TF
import random
from torchvision import transforms
import torch
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX


def crop_frame(pil_img, left, top, right, bottom):
    width, height = pil_img.size
    # left = width * 0.15
    # top = height * 0.15
    # right = width * 0.85
    # bottom = height
    # pil_img_cropped = pil_img.crop((left, top, right, bottom))
    # return pil_img_cropped
    left = width * left
    top = height * top
    right = width - (width * right)
    bottom = height
    pil_img_cropped = pil_img.crop((left, top, right, bottom))
    return pil_img_cropped

def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts

def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
    # align and crop the face according to the given bbox and landmarks, landmark: 5 key points
        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        
        if target_size[1] == 112:
            dst[:, 0] += 8.0
            
        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img, None

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)
        if len(face_align) == 0:
            return None, None, None
        landmark = predictor(cropped_face, face_align[0])
        landmark = face_utils.shape_to_np(landmark)

        return cropped_face, landmark, mask_face
    
    else:
        return None, None, None


def preprocess(video_path, num_frames):
    # Processes a single video file by detecting and cropping the largest face in each frame and saving the results.
    # Define face detector and predictor models
    # Crop face and extracting landmarks

    face_detector = dlib.get_frontal_face_detector()
    predictor_path = './dlib_tools/shape_predictor_81_face_landmarks.dat'
    ## Check if predictor path exists
    if not os.path.exists(predictor_path):
        print(f"Predictor path does not exist: {predictor_path}")
    face_predictor = dlib.shape_predictor(predictor_path)

    cap_org = cv2.VideoCapture(str(video_path))
    if not cap_org.isOpened():
        print(f"Failed to open {video_path}")
        return
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.arange(0, frame_count_org, 1, dtype=int)
    frames=[]
    for cnt_frame in range(frame_count_org):
        ret_org, frame_org = cap_org.read()
        frame_mask = None
        height, width = frame_org.shape[:-1]
        if not ret_org:
            print(f"Failed to read frame {cnt_frame} of {video_path}")
            break
        if cnt_frame not in frame_idxs:
                continue
        cropped_face, landmarks, _ = extract_aligned_face_dlib(face_detector, face_predictor, frame_org, mask=frame_mask)
        # Check if a face was detected and cropped
        if cropped_face is None:
            print(f"No faces in frame {cnt_frame} of {video_path}")
            continue
        if landmarks is None:
            print(f"No landmarks in frame {cnt_frame} of {video_path}")
            continue

        # TODO: change this instead of saving the frames, store them in a variable to use in model prediction loop
        # save_path_ = save_path / 'frames' / org_path.stem
        # save_path_.mkdir(parents=True, exist_ok=True)
        frames.append(cropped_face)
        
    cap_org.release()
    
    rotation_angle = random.uniform(-5, 5)
    crop_percents = {
        'left': random.uniform(0, 0.20),
        'top': random.uniform(0, 0.20),
        'right': random.uniform(0, 0.20),
        'bottom': random.uniform(0, 0.20)
    }
    transform = transforms.Compose([
            transforms.Resize((128, 128))
        ])

    #TODO: make a for loop to go through frames input to the model
    frame_list=[]
    sample_list=[]
    print("Number of frames, ", len(frames))
    for i in range(0, len(frames) - 1, 32):
        chunk = frames[i:i + 32]
        frame_list.append(chunk)

    for chunk in frame_list:
        sample = []
        for raw_image in chunk:
            pil_img = Image.fromarray(raw_image)
            
            # rotate and crop the image
            pil_img_rotated = pil_img.rotate(rotation_angle)
            pil_img_cropped = crop_frame(pil_img_rotated, **crop_percents)  # Assuming crop_frame returns a PIL Image
            
            # convert Image to a tensor
            img = TF.to_tensor(pil_img_cropped)
            
            img = transform(img) 
            
            sample.append(img)
        sample = np.stack([s.numpy() for s in sample])
        sample_list.append(sample)

    #TODO: Get the preditions for frames chunks
    model = PhysNet_padding_Encoder_Decoder_MAX(frames=num_frames)
    pretrained_weights = torch.load("runs/exp/logs/PhysNet_NeuralTextures_SGD_LR=0.001_LRReducer_32_frames_new_preprocessed_ds_real_paths_more_frames_skip_2_random_crop_rotation/PreTrainedModels/PhysNet_NeuralTextures_SGD_LR=0.001_LRReducer_32_frames_new_preprocessed_ds_real_paths_more_frames_skip_2_random_crop_rotation_Epoch15.pth")
    if pretrained_weights:
        model.load_state_dict(pretrained_weights, strict=False)
        print("Pretrained Loaded")
    predictions=[]
    with torch.no_grad():
        for sample in sample_list:
            print(sample.shape)
            if sample.shape[0] != 32:
                break
            sample=torch.tensor([sample])
            sample=sample.permute(0, 2, 1, 3, 4)
            prediction=model(sample)
            proba = torch.softmax(prediction, dim=1)
            pred_labels = np.argmax(proba.cpu().detach().numpy(), axis=1)
            
            predictions.append(pred_labels)

    #TODO: average to get a final prediction
    predictions_np = [p for p in predictions]
    print(predictions)
    print(predictions_np)
    video_prediction = np.mean(predictions_np)
    print(video_prediction)
    print("Fake" if video_prediction == 1 else "Real")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference video")
    
    # example: python inference.py --video_path /vol/research/DeepFakeDet/notebooks/FaceForensics++/manipulated_sequences/NeuralTextures/c23/videos/000_003.mp4
    parser.add_argument("--video_path", type=str, help="Path to the video")
    parser.add_argument("--num_frames", type=str, help="Number of frames", default=32)

    args = parser.parse_args()
    video_path = args.video_path
    num_frames = args.num_frames

    if video_path == "":
        raise ValueError("Please set video path")
    
    # preprocess the video
    preprocess(video_path, num_frames)
    
