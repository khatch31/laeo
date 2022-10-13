import os
# from utils import pickle_object
from collections import namedtuple
# from tf_agents.trajectories import time_step as ts
# import tensorflow as tf
# import tensorflow.compat.v1 as tf1
import numpy as np
from subprocess import Popen, PIPE

SaveTraj = namedtuple("SaveTraj", "traj info image")

# class EpisodeSaver:
#     def __init__(self, savedir, save_freq=1, save_seperate_episodes=True, save_video_summary=False, image_size=128):
#         self._savedir = savedir
#         self._save_freq = save_freq
#         self._save_seperate_episodes = save_seperate_episodes
#         self._save_video_summary = save_video_summary
#         self._trajectories = []
#         self._batch_no = 1
#         self._image_size = image_size
#
#     def add(self, traj, info, image):
#         save_traj = SaveTraj(traj, info, image)
#         self._trajectories.append(save_traj)
#
#     def write(self, train_step=None):
#         if train_step is None:
#             batch_save_dir = os.path.join(self._savedir, f"batch_{self._batch_no}")
#         else:
#             batch_save_dir = os.path.join(self._savedir, f"train_step_{train_step.numpy()}")
#
#         if not os.path.isdir(batch_save_dir):
#             os.makedirs(batch_save_dir)
#
#         if self._save_seperate_episodes:
#             episodes = []
#             current_episode = []
#             for save_traj in self._trajectories:
#                 current_episode.append(save_traj)
#
#                 if save_traj.traj.step_type == ts.StepType.LAST:
#                     episodes.append([t for t in current_episode])
#                     current_episode.clear()
#
#             for ep_idx, episode in enumerate(episodes):
#                 ep_no = ep_idx + 1
#                 if ep_no % self._save_freq == 0:
#                     episode_file = os.path.join(batch_save_dir, f"ep-{ep_no}.pkl")
#                     print(f"Saving {len(episode)} transitions to \"{episode_file}\"...")
#                     pickle_object(episode_file, episode)
#
#             if self._save_video_summary:
#                 B = len(episodes)
#                 T = len(episodes[0])
#                 H, W, C = self._image_size, self._image_size, 3
#
#                 video = np.zeros((B, T, H, W, C), dtype=np.uint8)
#                 for ep_idx, episode in enumerate(episodes):
#                     for t, save_traj in enumerate(episode):
#                         image = save_traj.image
#                         assert image.shape == (1, H , W , C)
#                         video[ep_idx, t] = image
#
#                 # video_summary("eval_episodes", video, step=train_step.numpy(), fps=20)
#                 video_summary("eval_episodes", video, step=train_step.numpy(), fps=20)
#
#         else:
#             raise NotImplementedError
#         self._trajectories.clear()
#         self._batch_no += 1
#
#
# def video_summary(name, video, step=None, fps=20):
#     name = name if isinstance(name, str) else name.decode('utf-8')
#     if np.issubdtype(video.dtype, np.floating):
#         video = np.clip(255 * video, 0, 255).astype(np.uint8)
#
#     B, T, H, W, C = video.shape
#     try:
#         frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
#         summary = tf1.Summary()
#         image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
#         image.encoded_image_string = encode_gif(frames, fps)
#         summary.value.add(tag=name + '/gif', image=image)
#         tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
#     except (IOError, OSError) as e:
#         print('GIF summaries require ffmpeg in $PATH.', e)
#         frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
#         tf.summary.image(name + '/grid', frames, step)
#
#     tf.summary.flush()
#
# def encode_gif(frames, fps):
#     h, w, c = frames[0].shape
#     pxfmt = {1: 'gray', 3: 'rgb24'}[c]
#     cmd = ' '.join([
#             f'ffmpeg -y -f rawvideo -vcodec rawvideo',
#             f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
#             f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
#             f'-r {fps:.02f} -f gif -'])
#     proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
#     for image in frames:
#         proc.stdin.write(image.tostring())
#     out, err = proc.communicate()
#     if proc.returncode:
#         raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
#     del proc
#     return out
#
# def load_pickled_object(filepath, gzipped=False):
#     if gzipped:
#         with gzip.open(filepath, "rb") as f:
#             obj = pickle.load(f)
#     else:
#         with open(filepath, "rb") as f:
#             obj = pickle.load(f)
#
#     return obj
