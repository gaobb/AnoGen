import math

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch

"""The scripts here are copied from DRAEM: https://github.com/VitjanZ/DRAEM"""


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


def rand_perlin_2d_np(
    shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3
):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: cv2.resize(
        np.repeat(
            np.repeat(
                gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]], d[0], axis=0
            ),
            d[1],
            axis=1,
        ),
        dsize=(shape[1], shape[0]),
    )
    dot = lambda grad, shift: (
        np.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            axis=-1,
        )
        * grad[: shape[0], : shape[1]]
    ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * lerp_np(
        lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1]
    )


rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


def perlin_noise(image, dtd_image, aug_prob=1.0):
    image = np.array(image, dtype=np.float32)
    dtd_image = np.array(dtd_image, dtype=np.float32)
    shape = image.shape[:2]
    min_perlin_scale, max_perlin_scale = 0, 6
    t_x = torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0]
    t_y = torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0]
    perlin_scalex, perlin_scaley = 2**t_x, 2**t_y

    perlin_noise = rand_perlin_2d_np(shape, (perlin_scalex, perlin_scaley))

    perlin_noise = rot(images=perlin_noise)
    perlin_noise = np.expand_dims(perlin_noise, axis=2)
    threshold = 0.5
    perlin_thr = np.where(
        perlin_noise > threshold,
        np.ones_like(perlin_noise),
        np.zeros_like(perlin_noise),
    )

    img_thr = dtd_image * perlin_thr / 255.0
    image = image / 255.0

    beta = torch.rand(1).numpy()[0] * 0.8
    image_aug = (
        image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
    )
    image_aug = image_aug.astype(np.float32)

    no_anomaly = torch.rand(1).numpy()[0]

    if no_anomaly > aug_prob:
        return image, np.zeros_like(perlin_thr)
    else:
        msk = (perlin_thr).astype(np.float32)
        msk = msk.transpose(2, 0, 1)

        return image_aug, msk
