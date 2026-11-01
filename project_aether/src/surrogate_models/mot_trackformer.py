# Copyright 2023 The Project AETHER Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformer track queries for particle / debris tracking (report Part IV).

Follows the track-query design of MOTR (F. Zeng et al., ECCV 2022) and
TrackFormer's joint detection-and-tracking formulation (T. Meinhardt et al.,
CVPR 2022):

  * Each live track is a query embedding that is carried from frame to frame
    and decoded against the current frame's detection tokens.
  * A fixed set of learned detect queries proposes newborn objects.
  * A query-interaction step (MOTR's QIM) keeps tracks whose score stays
    above `keep_threshold`, retires tracks after `max_misses` low-score
    frames, and turns confident detect queries into new tracks.
  * Training uses tracklet-aware label assignment: track queries keep the
    identity they already follow; only detect queries are matched to newborn
    objects with the Hungarian algorithm (`hungarian_match`).

Track state lives in fixed-capacity arrays so a whole frame update jits, and
"follow this parcel" queries in serving cost one incremental step.

`detect_debris` turns frames of the hypervelocity-impact shadowgraph clip in
`data/sample_clip` into detection tokens with a classical detector.

Typical usage example:

  state = mot_trackformer.init_track_state(cfg)
  for frame in frames:
    feats, pos, mask = mot_trackformer.detect_debris(frame, background, cfg)
    state, out = mot_trackformer.tracker_step(params, state, feats, pos,
                                              mask, cfg)
"""

import functools
import math
import pathlib
import re
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy import optimize

from src.surrogate_models import quantization_fp8

_dense = quantization_fp8.dense
_init_dense = quantization_fp8.init_dense


class TrackerConfig(NamedTuple):
    """Hyper-parameters of the tracker (hashable, used as a static argument).

    Attributes:
      embed_dim: Query / token width.
      num_heads: Attention heads.
      num_layers: Decoder layers.
      num_detect_queries: Learned detect (newborn) queries.
      max_tracks: Capacity of the track-query table.
      feature_dim: Width of a detection token's feature vector.
      max_detections: Detection tokens per frame (padded).
      birth_threshold: Score above which a detect query starts a track.
      keep_threshold: Score above which a track counts as observed.
      max_misses: Consecutive low-score frames before a track is retired.
    """
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    num_detect_queries: int = 32
    max_tracks: int = 64
    feature_dim: int = 6
    max_detections: int = 128
    birth_threshold: float = 0.6
    keep_threshold: float = 0.4
    max_misses: int = 3


class TrackState(NamedTuple):
    """Fixed-capacity table of track queries.

    Attributes:
      queries: Track query embeddings, shape (max_tracks, embed_dim).
      positions: Last position of each track in [0, 1]^2, (max_tracks, 2).
      ids: Track identity per slot, -1 for an empty slot, (max_tracks,).
      misses: Consecutive low-score frames per slot, (max_tracks,).
      next_id: Next identity to assign.
    """
    queries: jax.Array
    positions: jax.Array
    ids: jax.Array
    misses: jax.Array
    next_id: jax.Array


class FrameOutputs(NamedTuple):
    """Decoder outputs for the track slots followed by the detect queries."""
    hidden: jax.Array
    scores: jax.Array
    positions: jax.Array


def init_track_state(cfg: TrackerConfig) -> TrackState:
    return TrackState(
        queries=jnp.zeros((cfg.max_tracks, cfg.embed_dim)),
        positions=jnp.zeros((cfg.max_tracks, 2)),
        ids=jnp.full((cfg.max_tracks,), -1, jnp.int32),
        misses=jnp.zeros((cfg.max_tracks,), jnp.int32),
        next_id=jnp.zeros((), jnp.int32))


def _init_mha(key: jax.Array, dim: int) -> dict[str, Any]:
    kq, kk, kv, ko = jax.random.split(key, 4)
    return {'q': _init_dense(kq, dim, dim), 'k': _init_dense(kk, dim, dim),
            'v': _init_dense(kv, dim, dim), 'o': _init_dense(ko, dim, dim)}


def _init_norm(dim: int) -> dict[str, jax.Array]:
    return {'scale': jnp.ones((dim,)), 'bias': jnp.zeros((dim,))}


def init_params(key: jax.Array, cfg: TrackerConfig) -> dict[str, Any]:
    """Initializes the tracker parameters."""
    d = cfg.embed_dim
    keys = iter(jax.random.split(key, 16 + 6 * cfg.num_layers))
    layers = []
    for _ in range(cfg.num_layers):
        layers.append({
            'self': _init_mha(next(keys), d),
            'cross': _init_mha(next(keys), d),
            'norm1': _init_norm(d), 'norm2': _init_norm(d),
            'norm3': _init_norm(d),
            'ffn': {'up': _init_dense(next(keys), d, 2 * d),
                    'down': _init_dense(next(keys), 2 * d, d)},
        })
    return {
        'det_proj': _init_dense(next(keys), cfg.feature_dim, d),
        'pos_mlp': {'l1': _init_dense(next(keys), 2, d),
                    'l2': _init_dense(next(keys), d, d)},
        'detect_queries': 0.02 * jax.random.normal(
            next(keys), (cfg.num_detect_queries, d)),
        'detect_ref': jax.random.uniform(next(keys),
                                         (cfg.num_detect_queries, 2),
                                         minval=0.05, maxval=0.95),
        'layers': layers,
        'score_head': _init_dense(next(keys), d, 1),
        'delta_head': {'hidden': _init_dense(next(keys), d, d),
                       'out': _init_dense(next(keys), d, 2)},
        'qim': {'up': _init_dense(next(keys), d, 2 * d),
                'down': _init_dense(next(keys), 2 * d, d),
                'norm': _init_norm(d)},
    }


def _layer_norm(p: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + 1e-5) * p['scale'] + p['bias']


def _mha(p: dict[str, Any], x_q: jax.Array, x_kv: jax.Array,
         key_mask: jax.Array, num_heads: int) -> jax.Array:
    """Softmax multi-head attention of (Nq, D) queries over (Nk, D) tokens."""
    nq, d = x_q.shape
    nk = x_kv.shape[0]
    hd = d // num_heads
    q = _dense(p['q'], x_q).reshape(nq, num_heads, hd)
    k = _dense(p['k'], x_kv).reshape(nk, num_heads, hd)
    v = _dense(p['v'], x_kv).reshape(nk, num_heads, hd)
    scores = jnp.einsum('qhd,khd->hqk', q, k) / math.sqrt(hd)
    scores = jnp.where(key_mask[None, None, :], scores, -1e9)
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum('hqk,khd->qhd', attn, v).reshape(nq, d)
    return _dense(p['o'], out)


def _position_embedding(params: dict[str, Any], pos: jax.Array) -> jax.Array:
    h = jax.nn.relu(_dense(params['pos_mlp']['l1'], pos))
    return _dense(params['pos_mlp']['l2'], h)


def _logit(p: jax.Array) -> jax.Array:
    p = jnp.clip(p, 1e-4, 1.0 - 1e-4)
    return jnp.log(p) - jnp.log1p(-p)


def decode_frame(params: dict[str, Any], state: TrackState,
                 det_feats: jax.Array, det_pos: jax.Array,
                 det_mask: jax.Array, cfg: TrackerConfig) -> FrameOutputs:
    """Runs the decoder for all track slots and detect queries on one frame.

    Args:
      params: Tracker parameters.
      state: Current track table.
      det_feats: Detection token features, (max_detections, feature_dim).
      det_pos: Detection positions in [0, 1]^2, (max_detections, 2).
      det_mask: True for real (non-padding) detections.
      cfg: Tracker configuration.

    Returns:
      Outputs for max_tracks track slots followed by the detect queries.
    """
    memory = (_dense(params['det_proj'], det_feats) +
              _position_embedding(params, det_pos))
    refs = jnp.concatenate([state.positions, params['detect_ref']], axis=0)
    x = jnp.concatenate([state.queries, params['detect_queries']], axis=0)
    query_mask = jnp.concatenate(
        [state.ids >= 0, jnp.ones((cfg.num_detect_queries,), bool)])
    pos_embed = _position_embedding(params, refs)
    for layer in params['layers']:
        h = x + pos_embed
        x = _layer_norm(layer['norm1'],
                        x + _mha(layer['self'], h, h, query_mask,
                                 cfg.num_heads))
        x = _layer_norm(layer['norm2'],
                        x + _mha(layer['cross'], x + pos_embed, memory,
                                 det_mask, cfg.num_heads))
        ffn = _dense(layer['ffn']['down'],
                     jax.nn.gelu(_dense(layer['ffn']['up'], x)))
        x = _layer_norm(layer['norm3'], x + ffn)
    scores = jax.nn.sigmoid(_dense(params['score_head'], x)[:, 0])
    delta = _dense(params['delta_head']['out'],
                   jax.nn.relu(_dense(params['delta_head']['hidden'], x)))
    positions = jax.nn.sigmoid(_logit(refs) + delta)
    return FrameOutputs(x, scores, positions)


def _query_interaction(params: dict[str, Any], hidden: jax.Array) -> jax.Array:
    """Updates a decoded query into next frame's track query (MOTR's QIM)."""
    q = params['qim']
    h = _dense(q['down'], jax.nn.gelu(_dense(q['up'], hidden)))
    return _layer_norm(q['norm'], hidden + h)


def update_tracks(params: dict[str, Any], state: TrackState,
                  out: FrameOutputs, cfg: TrackerConfig) -> TrackState:
    """Applies the keep / retire / birth rules to the track table."""
    t = cfg.max_tracks
    alive = state.ids >= 0
    observed = alive & (out.scores[:t] >= cfg.keep_threshold)
    misses = jnp.where(observed, 0, jnp.where(alive, state.misses + 1, 0))
    survive = alive & (misses <= cfg.max_misses)

    new_queries = _query_interaction(params, out.hidden)
    queries = jnp.where(survive[:, None], new_queries[:t], 0.0)
    positions = jnp.where(survive[:, None], out.positions[:t], 0.0)
    ids = jnp.where(survive, state.ids, -1)
    misses = jnp.where(survive, misses, 0)

    # Births: confident detect queries fill free slots in index order.
    born = out.scores[t:] >= cfg.birth_threshold
    free = ~survive
    num_free = jnp.sum(free.astype(jnp.int32))
    rank = jnp.cumsum(born.astype(jnp.int32)) - 1
    free_slots = jnp.argsort(jnp.where(free, 0, 1), kind='stable')
    accepted = born & (rank < num_free)
    slot = jnp.where(accepted, free_slots[jnp.clip(rank, 0, t - 1)], t)
    queries = queries.at[slot].set(new_queries[t:], mode='drop')
    positions = positions.at[slot].set(out.positions[t:], mode='drop')
    ids = ids.at[slot].set(state.next_id + rank, mode='drop')
    misses = misses.at[slot].set(0, mode='drop')
    next_id = state.next_id + jnp.sum(accepted.astype(jnp.int32))
    return TrackState(queries, positions, ids, misses, next_id)


@functools.partial(jax.jit, static_argnames='cfg')
def tracker_step(params: dict[str, Any], state: TrackState,
                 det_feats: jax.Array, det_pos: jax.Array,
                 det_mask: jax.Array,
                 cfg: TrackerConfig) -> tuple[TrackState, FrameOutputs]:
    """Decodes one frame and updates the track table.

    Args:
      params: Tracker parameters.
      state: Track table before the frame.
      det_feats: Detection token features, (max_detections, feature_dim).
      det_pos: Detection positions in [0, 1]^2.
      det_mask: True for real detections.
      cfg: Tracker configuration.

    Returns:
      A tuple (new_state, outputs).
    """
    out = decode_frame(params, state, det_feats, det_pos, det_mask, cfg)
    return update_tracks(params, state, out, cfg), out


# ------------------------------------------------------------------------------
# Training with tracklet-aware label assignment.
# ------------------------------------------------------------------------------


def hungarian_match(pred_pos: np.ndarray, pred_scores: np.ndarray,
                    gt_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bipartite matching of detect-query predictions to newborn objects.

    Args:
      pred_pos: Predicted positions of the detect queries, (Nd, 2).
      pred_scores: Predicted scores of the detect queries, (Nd,).
      gt_pos: Positions of objects not yet followed by a track, (G, 2).

    Returns:
      A tuple (query_index, object_index) of matched pairs.
    """
    if gt_pos.shape[0] == 0:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    cost = (np.abs(pred_pos[:, None, :] - gt_pos[None, :, :]).sum(-1) -
            pred_scores[:, None])
    return optimize.linear_sum_assignment(cost)


def tracking_loss(params: dict[str, Any], state: TrackState,
                  det_feats: jax.Array, det_pos: jax.Array,
                  det_mask: jax.Array, track_targets: jax.Array,
                  track_valid: jax.Array, detect_targets: jax.Array,
                  detect_valid: jax.Array, cfg: TrackerConfig) -> jax.Array:
    """Score (binary cross-entropy) plus L1 position loss for one frame.

    Args:
      params: Tracker parameters.
      state: Track table at the start of the frame.
      det_feats: Detection token features.
      det_pos: Detection positions.
      det_mask: True for real detections.
      track_targets: Target position for each track slot, (max_tracks, 2).
      track_valid: True where the slot's object is visible in this frame.
      detect_targets: Matched newborn position per detect query, (Nd, 2).
      detect_valid: True where the detect query was matched.
      cfg: Tracker configuration.

    Returns:
      Scalar loss.
    """
    out = decode_frame(params, state, det_feats, det_pos, det_mask, cfg)
    targets = jnp.concatenate([track_targets, detect_targets], axis=0)
    valid = jnp.concatenate([track_valid, detect_valid]).astype(jnp.float32)
    eps = 1e-6
    bce = -(valid * jnp.log(out.scores + eps) +
            (1.0 - valid) * jnp.log(1.0 - out.scores + eps))
    l1 = jnp.sum(jnp.abs(out.positions - targets), axis=-1)
    return jnp.mean(bce) + 5.0 * jnp.sum(valid * l1) / jnp.maximum(
        jnp.sum(valid), 1.0)


# ------------------------------------------------------------------------------
# Detections from the hypervelocity-impact shadowgraph clip.
# ------------------------------------------------------------------------------

_FRAME_NUMBER = re.compile(r'_(\d+)\.bmp$')


def load_clip(directory: str | pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Loads a BMP frame sequence as grayscale floats in [0, 1].

    Args:
      directory: Folder of `<name>_<frame>.bmp` files.

    Returns:
      A tuple (frames, frame_numbers) with frames of shape (T, H, W).

    Raises:
      FileNotFoundError: If the folder contains no BMP frames.
    """
    paths = sorted(pathlib.Path(directory).glob('*.bmp'),
                   key=lambda p: int(_FRAME_NUMBER.search(p.name).group(1)))
    if not paths:
        raise FileNotFoundError(f'No .bmp frames in {directory}')
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert('L'), np.float32) / 255.0)
    numbers = np.array([int(_FRAME_NUMBER.search(p.name).group(1))
                        for p in paths])
    return np.stack(frames), numbers


def detect_debris(frame: np.ndarray, background: np.ndarray,
                  cfg: TrackerConfig, threshold: float = 0.12,
                  min_area: int = 4
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detects dark debris fragments against the bright shadowgraph background.

    Args:
      frame: Grayscale frame in [0, 1], shape (H, W).
      background: Per-pixel background (e.g. the temporal maximum), (H, W).
      cfg: Tracker configuration (feature_dim must be 6).
      threshold: Minimum darkening relative to the background.
      min_area: Minimum fragment area in pixels.

    Returns:
      A tuple (features, positions, mask) padded to cfg.max_detections:
      features are [x, y, log(area), mean darkening, width, height] with
      coordinates normalized to [0, 1].
    """
    height, width = frame.shape
    darkening = ndimage.gaussian_filter(background - frame, sigma=1.0)
    labels, count = ndimage.label(darkening > threshold)
    feats = np.zeros((cfg.max_detections, cfg.feature_dim), np.float32)
    pos = np.zeros((cfg.max_detections, 2), np.float32)
    mask = np.zeros((cfg.max_detections,), bool)
    if count == 0:
        return feats, pos, mask
    index = np.arange(1, count + 1)
    area = ndimage.sum_labels(np.ones_like(frame), labels, index)
    keep = np.nonzero(area >= min_area)[0]
    keep = keep[np.argsort(-area[keep])][:cfg.max_detections]
    centers = ndimage.center_of_mass(darkening, labels, index[keep])
    darkness = ndimage.mean(darkening, labels, index[keep])
    boxes = ndimage.find_objects(labels)
    for row, (label_idx, (cy, cx), dark) in enumerate(
            zip(keep, centers, darkness)):
        box = boxes[label_idx]
        pos[row] = (cx / width, cy / height)
        feats[row] = (cx / width, cy / height, math.log(area[label_idx]), dark,
                      (box[1].stop - box[1].start) / width,
                      (box[0].stop - box[0].start) / height)
        mask[row] = True
    return feats, pos, mask
