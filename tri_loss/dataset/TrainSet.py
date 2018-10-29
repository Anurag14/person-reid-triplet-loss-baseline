from .Dataset import Dataset
from ..utils.dataset_utils import parse_im_name

import os.path as osp
from PIL import Image
import numpy as np
from collections import defaultdict


class TrainSet(Dataset):
  """Training set for triplet loss.
  Args:
    ids2labels: a dict mapping ids to labels
  """

  def __init__(
      self,
      im_dir=None,
      im_names=None,
      ids2labels=None,
      ids_per_batch=None,
      ims_per_id=None,
      **kwargs):

    # The im dir of all images
    self.im_dir = im_dir
    self.im_names = im_names
    self.ids2labels = ids2labels
    self.ids_per_batch = ids_per_batch
    self.ims_per_id = ims_per_id

    im_ids = [parse_im_name(name, 'id') for name in im_names]
    self.ids_to_im_inds = defaultdict(list)
    for ind, id in enumerate(im_ids):
      self.ids_to_im_inds[id].append(ind)
    self.ids = self.ids_to_im_inds.keys()

    super(TrainSet, self).__init__(
      dataset_size=len(self.ids),
      batch_size=ids_per_batch,
      **kwargs)

  def get_sample(self, ptr):
    """Here one sample means several images (and labels etc) of one id.
    Returns:
      ims: a list of images
    """
    inds = self.ids_to_im_inds[self.ids[ptr]]
    if len(inds) < self.ims_per_id:
      inds = np.random.choice(inds, self.ims_per_id, replace=True)
    else:
      inds = np.random.choice(inds, self.ims_per_id, replace=False)
    im_names = [self.im_names[ind] for ind in inds]
    ims = [np.asarray(Image.open(osp.join(self.im_dir, name)))
           for name in im_names]
    ims, mirrored = zip(*[self.pre_process_im(im) for im in ims])
    labels = [self.ids2labels[self.ids[ptr]] for _ in range(self.ims_per_id)]
    return ims, im_names, labels, mirrored
  
  def set_feat_func(self, extract_feat_func):
    self.extract_feat_func = extract_feat_func
    
  def next_batch(self):
    """Next batch of images and labels.
    Returns:
      ims: numpy array with shape [N, H, W, C] or [N, C, H, W], N >= 1
      img_names: a numpy array of image names, len(img_names) >= 1
      labels: a numpy array of image labels, len(labels) >= 1
      mirrored: a numpy array of booleans, whether the images are mirrored
      self.epoch_done: whether the epoch is over
    """
    # Start enqueuing and other preparation at the beginning of an epoch.
    if self.epoch_done and self.shuffle:
      np.random.shuffle(self.ids)
    samples, self.epoch_done = self.prefetcher.next_batch()
    im_list, im_names, labels, mirrored = zip(*samples)
    # t = time.time()
    # Transform the list into a numpy array with shape [N, ...]
    ims = np.stack(np.concatenate(im_list))
    # print '---stacking time {:.4f}s'.format(time.time() - t)
    im_names = np.concatenate(im_names)
    labels = np.concatenate(labels)
    mirrored = np.concatenate(mirrored)
    return ims, im_names, labels, mirrored, self.epoch_done
  
  def extract_feat(self, normalize_feat, verbose=True):
    """Extract the features of the whole image set.
    Args:
      normalize_feat: True or False, whether to normalize feature to unit length
      verbose: whether to print the progress of extracting feature
    Returns:
      feat: numpy array with shape [N, C]
      ids: numpy array with shape [N]
      cams: numpy array with shape [N]
      im_names: numpy array with shape [N]
      marks: numpy array with shape [N]
    """
    feat, ids, cams, im_names, mirrored = [], [], [], [], []
    done = False
    step = 0
    printed = False
    st = time.time()
    last_time = time.time()
    while not done:
      ims_, im_names_, labels_, mirrored_, done = self.next_batch()
      feat_ = self.extract_feat_func(ims_)
      feat.append(feat_)
      im_names.append(im_names_)
      labels.append(labels_)
      #mirrored.append(mirrored_)

      if verbose:
        # Print the progress of extracting feature
        total_batches = (self.prefetcher.dataset_size
                         // self.prefetcher.batch_size + 1)
        step += 1
        if step % 20 == 0:
          if not printed:
            printed = True
          else:
            # Clean the current line
            sys.stdout.write("\033[F\033[K")
          print('{}/{} batches done, +{:.2f}s, total {:.2f}s'
                .format(step, total_batches,
                        time.time() - last_time, time.time() - st))
          last_time = time.time()

    feat = np.vstack(feat)
    labels = np.hstack(labels)
    #cams = np.hstack(cams)
    im_names = np.hstack(im_names)
    #marks = np.hstack(marks)
    if normalize_feat:
      feat = normalize(feat, axis=1)
    return feat, im_names, labels
