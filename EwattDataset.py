# -*- coding:utf8 -*-

from mrcnn import utils
import os
import json
import numpy as np
import skimage.draw
from skimage import io
class EwattDataset(utils.Dataset):
   def load_antenna(self,dataset_dir,subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        self.add_class("antenna",1,"antenna")
        assert subset in ["train","val"]
        dataset_dir = os.path.join(dataset_dir,subset)
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
            image_path = os.path.join(dataset_dir, a['filename'])
            image = io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                "antenna",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

   def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "antenna":
            return super(self.__class__,self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"],info["width"],len(info["polygons"])],dtype=np.uint8)
        for i, p in enumerate(info['polygons']):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        return mask.astype(np.bool),np.ones([mask.shape[-1]],dtype=np.int32)

   def image_reference(self, image_id):
       info = self.image_info[image_id]
       if info["source"] != "antenna":
           return info["path"]
       else:
           super(self.__class__,self).image_reference(image_id)

