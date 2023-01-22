from PIL import Image
from tqdm.auto import tqdm
import glob
import os

from face_alignment import mtcnn
from question_utils import getLogger

logger = getLogger(__name__, "INFO")

root = "./data/gallery"
img_paths = glob.glob(os.path.join(root, "*.jpg"))
logger.info(f"Processing {len(img_paths)} images at {root}")

outdir = "./data/gallery_processed"
os.makedirs(outdir, exist_ok=True)

idx = 0
mtcnn_model = mtcnn.MTCNN(device='cuda:0', crop_size=(112, 112))
for img_path in tqdm(img_paths):
    img = Image.open(img_path).convert('RGB')
    bboxes, faces = mtcnn_model.align_multi(img, limit=1)
    for face in faces:
        face.save(os.path.join(outdir, f"{idx}.png"))
        # face.save(os.path.join(outdir, f"{os.path.basename(img_path)[:-4]}.png"))
        idx += 1

logger.info(f"Done. Total cropped faces = {idx}")