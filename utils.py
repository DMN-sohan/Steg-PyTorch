import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import random
import itertools

def random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line):
    N = N_blur
    coords = torch.stack(torch.meshgrid(torch.arange(N_blur), torch.arange(N_blur)), -1).float() - (.5 * (N-1))
    manhat = torch.sum(torch.abs(coords), -1)

    # nothing, default
    vals_nothing = (manhat < .5).float()

    # gauss
    sig_gauss = torch.empty(1).uniform_(sigrange_gauss[0], sigrange_gauss[1])
    vals_gauss = torch.exp(-torch.sum(coords**2, -1)/2./sig_gauss**2)

    # line
    theta = torch.empty(1).uniform_(0, 2.*np.pi)
    v = torch.tensor([torch.cos(theta), torch.sin(theta)])
    dists = torch.sum(coords * v, -1)

    sig_line = torch.empty(1).uniform_(sigrange_line[0], sigrange_line[1])
    w_line = torch.empty(1).uniform_(wmin_line, .5 * (N-1) + .1)

    vals_line = torch.exp(-dists**2/2./sig_line**2) * (manhat < w_line).float()

    t = torch.rand(1)
    vals = vals_nothing
    vals = torch.where(t < probs[0]+probs[1], vals_line, vals)
    vals = torch.where(t < probs[0], vals_gauss, vals)

    v = vals / torch.sum(vals)
    z = torch.zeros_like(v)
    f = torch.stack([v,z,z, z,v,z, z,z,v],-1).reshape(N,N,3,3)

    return f

def get_rand_transform_matrix(image_size, d, batch_size):
    Ms = torch.zeros((batch_size, 2, 8))

    for i in range(batch_size):
        tl_x = random.uniform(-d, d)     # Top left corner, top
        tl_y = random.uniform(-d, d)    # Top left corner, left
        bl_x = random.uniform(-d, d)  # Bot left corner, bot
        bl_y = random.uniform(-d, d)    # Bot left corner, left
        tr_x = random.uniform(-d, d)     # Top right corner, top
        tr_y = random.uniform(-d, d)   # Top right corner, right
        br_x = random.uniform(-d, d)  # Bot right corner, bot
        br_y = random.uniform(-d, d)   # Bot right corner, right

        rect = np.array([
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y +  image_size]], dtype = "float32")

        dst = np.array([
            [0, 0],
            [image_size, 0],
            [image_size, image_size],
            [0, image_size]], dtype = "float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        M_inv = np.linalg.inv(M)
        Ms[i,0,:] = torch.from_numpy(M_inv.flatten()[:8])
        Ms[i,1,:] = torch.from_numpy(M.flatten()[:8])
    return Ms

def get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size):
    rnd_hue = torch.rand((batch_size,1,1,3)) * 2 * rnd_hue - rnd_hue
    rnd_brightness = torch.rand((batch_size,1,1,1)) * 2 * rnd_bri - rnd_bri
    return rnd_hue + rnd_brightness

def rgb_to_ycbcr(image):
    matrix = torch.tensor(
        [[65.481, 128.553, 24.966],
         [-37.797, -74.203, 112.],
         [112., -93.786, -18.214]],
        dtype=torch.float32).t() / 255
    shift = torch.tensor([16., 128., 128.])

    result = torch.tensordot(image, matrix, dims=1) + shift
    return result

def rgb_to_ycbcr_jpeg(image):
    matrix = torch.tensor(
        [[0.299, 0.587, 0.114],
         [-0.168736, -0.331264, 0.5],
         [0.5, -0.418688, -0.081312]],
        dtype=torch.float32).t()
    shift = torch.tensor([0., 128., 128.])

    result = torch.tensordot(image, matrix, dims=1) + shift
    return result

def downsampling_420(image):
    y, cb, cr = torch.split(image, 1, dim=3)
    cb = F.avg_pool2d(cb, kernel_size=2, stride=2)
    cr = F.avg_pool2d(cr, kernel_size=2, stride=2)
    return (y.squeeze(3), cb.squeeze(3), cr.squeeze(3))

def image_to_patches(image):
    k = 8
    batch_size, height, width = image.shape
    return image.view(batch_size, height // k, k, -1, k).permute(0, 1, 3, 2, 4).reshape(batch_size, -1, k, k)

def dct_8x8(image):
    image = image - 128
    tensor = torch.zeros((8, 8, 8, 8), dtype=torch.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = torch.cos(torch.tensor((2 * x + 1) * u * np.pi / 16)) * torch.cos(torch.tensor((2 * y + 1) * v * np.pi / 16))
    alpha = torch.tensor([1. / np.sqrt(2)] + [1] * 7)
    scale = torch.outer(alpha, alpha) * 0.25
    result = scale * torch.tensordot(image, tensor, dims=2)
    return result

y_table = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=torch.float32).t()

c_table = torch.full((8, 8), 99, dtype=torch.float32)
c_table[:4, :4] = torch.tensor([
    [17, 18, 24, 47],
    [18, 21, 26, 66],
    [24, 26, 56, 99],
    [47, 66, 99, 99]
]).t()

def y_quantize(image, rounding, factor=1):
    image = image / (y_table * factor)
    image = rounding(image)
    return image

def c_quantize(image, rounding, factor=1):
    image = image / (c_table * factor)
    image = rounding(image)
    return image

def y_dequantize(image, factor=1):
    return image * (y_table * factor)

def c_dequantize(image, factor=1):
    return image * (c_table * factor)

def idct_8x8(image):
    alpha = torch.tensor([1. / np.sqrt(2)] + [1] * 7)
    alpha = torch.outer(alpha, alpha)
    image = image * alpha

    tensor = torch.zeros((8, 8, 8, 8), dtype=torch.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = torch.cos(torch.tensor((2 * u + 1) * x * np.pi / 16)) * torch.cos(torch.tensor((2 * v + 1) * y * np.pi / 16))
    result = 0.25 * torch.tensordot(image, tensor, dims=2) + 128
    return result

def patches_to_image(patches, height, width):
    k = 8
    batch_size = patches.shape[0]
    image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    return image_transposed.reshape(batch_size, height, width)

def upsampling_420(y, cb, cr):
    def repeat(x, k=2):
        height, width = x.shape[1:3]
        x = x.unsqueeze(-1)
        x = x.repeat(1, 1, k, k)
        x = x.view(-1, height * k, width * k)
        return x

    cb = repeat(cb)
    cr = repeat(cr)
    return torch.stack((y, cb, cr), dim=-1)

def ycbcr_to_rgb(image):
    matrix = torch.tensor([
        [298.082, 0, 408.583],
        [298.082, -100.291, -208.120],
        [298.082, 516.412, 0]
    ], dtype=torch.float32).t() / 256
    shift = torch.tensor([-222.921, 135.576, -276.836])

    result = torch.tensordot(image, matrix, dims=1) + shift
    return result

def ycbcr_to_rgb_jpeg(image):
    matrix = torch.tensor([
        [1., 0., 1.402],
        [1, -0.344136, -0.714136],
        [1, 1.772, 0]
    ], dtype=torch.float32).t()
    shift = torch.tensor([0, -128, -128])

    result = torch.tensordot(image + shift, matrix, dims=1)
    return result

def diff_round(x):
    return torch.round(x) + (x - torch.round(x))**3

def round_only_at_0(x):
    cond = (torch.abs(x) < 0.5).float()
    return cond * (x ** 3) + (1 - cond) * x

def jpeg_compress_decompress(image, downsample_c=True, rounding=diff_round, factor=1):
    image *= 255
    height, width = image.shape[1:3]
    orig_height, orig_width = height, width
    if height % 16 != 0 or width % 16 != 0:
        height = ((height - 1) // 16 + 1) * 16
        width = ((width - 1) // 16 + 1) * 16

        vpad = height - orig_height
        wpad = width - orig_width

        image = F.pad(image, (0, wpad, 0, vpad), mode='reflect')

    # "Compression"
    image = rgb_to_ycbcr_jpeg(image)
    if downsample_c:
        y, cb, cr = downsampling_420(image)
    else:
        y, cb, cr = torch.split(image, 1, dim=3)
    components = {'y': y, 'cb': cb, 'cr': cr}
    for k in components.keys():
        comp = components[k]
        comp = image_to_patches(comp)
        comp = dct_8x8(comp)
        comp = c_quantize(comp, rounding, factor) if k in ('cb', 'cr') else y_quantize(comp, rounding, factor)
        components[k] = comp

    # "Decompression"
    for k in components.keys():
        comp = components[k]
        comp = c_dequantize(comp, factor) if k in ('cb', 'cr') else y_dequantize(comp, factor)
        comp = idct_8x8(comp)
        if k in ('cb', 'cr'):
            if downsample_c:
                comp = patches_to_image(comp, height//2, width//2)
            else:
                comp = patches_to_image(comp, height, width)
        else:
            comp = patches_to_image(comp, height, width)
        components[k] = comp

    y, cb, cr = components['y'], components['cb'], components['cr']
    if downsample_c:
        image = upsampling_420(y, cb, cr)
    else:
        image = torch.stack((y, cb, cr), dim=-1)
    image = ycbcr_to_rgb_jpeg(image)

    # Crop to original size
    if orig_height != height or orig_width != width:
        image = image[:, :orig_height, :orig_width]

    image = torch.clamp(image, 0, 255)
    image /= 255

    return image

def quality_to_factor(quality):
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality*2
    return quality / 100.
