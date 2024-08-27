import os
import sys
import time
import logging
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import numpy as np
from PIL import Image as PILImage
import torch
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.arguments import load_opt_command, load_default_cmd
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from openseed.BaseModel import BaseModel
from openseed import build_model
# from utils.visualizer import Visualizer

logger = logging.getLogger(__name__)


class OpenSeeDProcessor:
    def __init__(self):
        rospy.init_node('openseed_processor', anonymous=True)

        # self.opt, self.cmdline_args = load_opt_command()
        # if self.cmdline_args.user_dir:
        #     absolute_user_dir = os.path.abspath(self.cmdline_args.user_dir)
        #     self.opt['user_dir'] = absolute_user_dir

        #default
        command = 'evaluate'
        conf_files = ['/home/url/ros1_cuda_docker/openseed_img2pc_generator_ws/src/OpenSeeD_LiDAR_fusion/configs/openseed_swint_lang.yaml']
        overrides = {'WEIGHT': '/home/url/ros1_cuda_docker/openseed_img2pc_generator_ws/src/OpenSeeD_LiDAR_fusion/checkpoints/model_state_dict_swint_51.2ap.pt'}

        self.opt, self.cmdline_args = load_default_cmd(command, conf_files, overrides)

        # pretrained_pth = os.path.join(self.opt['WEIGHT'])
        pretrained_pth = '/home/url/ros1_cuda_docker/openseed_img2pc_generator_ws/src/OpenSeeD_LiDAR_fusion/checkpoints/model_state_dict_swint_51.2ap.pt'
        
        self.model = BaseModel(self.opt, build_model(self.opt)).from_pretrained(pretrained_pth).eval().cuda()

        self.transform = transforms.Compose([transforms.Resize(512, interpolation=PILImage.BICUBIC)])

        stuff_classes = self.opt['TEXT_CLASSES']['STUFF']
        stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(len(stuff_classes))]
        stuff_dataset_id_to_contiguous_id = {x: x for x in range(len(stuff_classes))}

        MetadataCatalog.get("seg").set(
            stuff_colors=stuff_colors,
            stuff_classes=stuff_classes,
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
        )
        self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes, is_eval=True)
        self.metadata = MetadataCatalog.get('seg')
        self.model.model.metadata = self.metadata
        self.model.model.sem_seg_head.num_classes = len(stuff_classes)

        self.bridge = CvBridge()

        self.pub_semseg_labeled_img = rospy.Publisher('/semseg_labeled_img', ROSImage, queue_size=10)

        self.image_sub = rospy.Subscriber('/go1_d435/color/image_raw/compressed', CompressedImage, self.callback)

        self.rate = rospy.Rate(10)  # 10Hz

        # self.image_saved = 0

    def callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        image_ori = PILImage.fromarray(cv_image)

        image = self.transform(image_ori)
        image = np.asarray(image)
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(image.copy()).permute(2, 0, 1).cuda()

        batch_inputs = [{'image': images, 'height': image_ori.shape[0], 'width': image_ori.shape[1]}]
        outputs = self.model.forward(batch_inputs, inference_task="sem_seg")
        # visual = Visualizer(image_ori, metadata=self.metadata)

        sem_seg = outputs[-1]['sem_seg'].max(0)[1]

        # demo = visual.draw_sem_seg(sem_seg.cpu(), alpha=0.5) 
        semseg_labeled_image = self.create_label_index_image(sem_seg)

        
        # if self.image_saved == 100:
            # output_path = '/home/url/ros1_cuda_docker/openseed_img2pc_generator_ws/src/lidar-camera-fusion/segmented_img.png'
            # demo.save(output_path)
            # print("example image printed-----------")

        # self.image_saved += 1

        # result_image = Image.fromarray(demo.get_image())
        semseg_labeled_msg = self.bridge.cv2_to_imgmsg(semseg_labeled_image, encoding="mono8")
        self.pub_semseg_labeled_img.publish(semseg_labeled_msg)
        
    def create_label_index_image(self, sem_seg):
        """
        Create an image where each pixel is assigned the index of its label's color.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.

        Returns:
            label_index_image (ndarray): an image where each pixel contains the integer index
                                        of the corresponding label color.
        """
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.cpu().numpy()
        
        # Initialize an image to store the label indices
        label_index_image = np.zeros(sem_seg.shape, dtype=np.uint8)
        
        # Get unique labels and their areas
        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        
        # Iterate over the labels and assign the corresponding index
        for label in filter(lambda l: l < len(self.metadata.stuff_classes), labels):
            binary_mask = (sem_seg == label).astype(np.uint8)
            label_index_image[binary_mask == 1] = label  # Assign the label index to the corresponding pixels
            # print("label: ", label) # integer printing
        return label_index_image

    

    def spin(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

if __name__ == "__main__":
    openseed_node = OpenSeeDProcessor()
    openseed_node.spin()
