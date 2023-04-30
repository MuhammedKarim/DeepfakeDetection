from efficientnet_autoatt import *
from tranform_util import *
from video_reader import *
from face_extractor import *
import os
from scipy.special import expit
from torch.utils.model_zoo import load_url


# See if we are using GPU or CPU for calculations
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print(f'{device} is available\n')
print(f'Using {device} for inference..')

#Pretrained model:
net_model = 'EfficientNetAutoAttB4'

#Choosing the training dataset
train_db = 'DFDC'

face_policy = 'scale'
face_size = 224
frames_per_video = 32

weight_url = {

'EfficientNetAutoAttB4_DFDC':'https://f002.backblazeb2.com/file/icpr2020/EfficientNetAutoAttB4_DFDC_bestval-72ed969b2a395fffe11a0d5bf0a635e7260ba2588c28683630d97ff7153389fc.pth'
}

#Loading the pretrained model and dataset :
model_url = weight_url['{:s}_{:s}'.format(net_model,train_db)]

net = EfficientNetAutoAttB4().eval().to(device)

#map_location : target device, which is in our case cpu, where the model parameters will be loaded.
net.load_state_dict(load_url(model_url,map_location=device,check_hash=True))

transf = get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

# Create an instance of the BlazeFace model and move it to the specified device
facedet = BlazeFace().to(device)

# Load the pre-trained weights of the BlazeFace model
facedet.load_weights(os.path.abspath("weights/blazeface.pth"))

# Load the anchor boxes used for facial detection by BlazeFace
facedet.load_anchors(os.path.abspath("weights/anchors.npy"))

# Create an instance of the VideoReader class to read video frames
videoreader = VideoReader(verbose=False)

# Define a lambda function to read frames from a video file
video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)

# Create an instance of the FaceExtractor class
face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)

vid_real_faces = face_extractor.process_video(os.path.abspath('videos/real/lynaeydofd.mp4'))
vid_fake_faces = face_extractor.process_video(os.path.abspath('videos/deepfake/mqzvfufzoq.mp4'))

# For each frame, we consider the face with the highest confidence score found by BlazeFace (= frame['faces'][0])
faces_real_t = torch.stack(
    [transf(image=frame['faces'][0])['image'] for frame in vid_real_faces if len(frame['faces'])])
faces_fake_t = torch.stack(
    [transf(image=frame['faces'][0])['image'] for frame in vid_fake_faces if len(frame['faces'])])

with torch.no_grad():
    faces_real_pred = net(faces_real_t.to(device)).cpu().numpy().flatten()
    faces_fake_pred = net(faces_fake_t.to(device)).cpu().numpy().flatten()
"""
Print average scores.
An average score close to 0 predicts REAL. An average score close to 1 predicts FAKE.
"""
#print('Average score for REAL video: {:.4f}'.format(expit(faces_real_pred.mean())))
#print('Average score for FAKE face: {:.4f}'.format(expit(faces_fake_pred.mean())))
