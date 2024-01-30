# 24-01-30

import numpy as np


def randomize_blocks(
    data: np.ndarray, shuffled: np.ndarray, block_size=100, sample_rate=0.1
) -> np.ndarray:

    if data is None:
        raise Exception("data is None")

    pseudo_shape = np.array([3600, 1800])
    shuffled.ravel()[:] = data.ravel()[:]

    blocks = (
        np.array(data.shape) if type(data) == np.ndarray else pseudo_shape
    ) // block_size
    num = np.product(blocks)
    rand_list = np.random.choice(np.arange(num), int(num * sample_rate))

    def _1d(block_id):
        start = block_id * block_size
        end = np.min([start + block_size, data.shape[0]])
        _ = np.copy(data[start:end])
        np.random.shuffle(_)
        shuffled[start:end] = _[:]

    def _2d(block_id):
        dbg_flag = False

        _, bX = blocks
        y, x = block_id // bX, block_id % bX
        ystart = y * block_size
        yend = np.min([ystart + block_size, data.shape[0]])
        xstart = x * block_size
        xend = np.min([xstart + block_size, data.shape[1]])
        _ = np.copy(data[ystart:yend, xstart:xend])
        np.random.shuffle(_)
        shuffled[ystart:yend, xstart:xend] = _[:, :]

    def _3d(block_id):
        _, bY, bX = blocks
        z, y, x = (block_id // bX) // bY, (block_id // bX) % bY, block_id % bX

        zstart = z * block_size
        zend = np.min([zstart + block_size, data.shape[0]])
        ystart = y * block_size
        yend = np.min([ystart + block_size, data.shape[1]])
        xstart = x * block_size
        xend = np.min([xstart + block_size, data.shape[2]])
        _ = np.copy(data[zstart:zend, ystart:yend, xstart:xend])
        np.random.shuffle(_)
        shuffled[zstart:zend, ystart:yend, xstart:xend] = _[:, :, :]

    for i, block_id in enumerate(rand_list):

        if len(data.shape) == 1:
            _1d(block_id)
        elif len(data.shape) == 2:
            _2d(block_id)
        elif len(data.shape) == 3:
            _3d(block_id)
