import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from models import vgg
from utils import gaussian_utils
from utils import id_mrf


def reconstruction_loss(y_true, y_pred):
    diff = K.abs(y_pred - y_true)
    l1 = K.mean(diff, axis=[1, 2, 3])
    return l1


def wasserstein_loss(y_true, y_pred, wgan_loss_weight=1.0):
    return wgan_loss_weight * K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


def confidence_reconstruction_loss(y_true, y_pred, mask, num_steps, gaussian_kernel_size,
                                   gaussian_kernel_std):
    mask_blurred = gaussian_utils.blur_mask(mask, num_steps, gaussian_kernel_size,
                                            gaussian_kernel_std)
    valid_mask = 1 - mask
    diff = K.abs(y_true - y_pred)
    l1 = K.mean(diff * valid_mask + diff * mask_blurred)
    return l1


def id_mrf_loss(y_true, y_pred, mask, nn_stretch_sigma, batch_size, vgg_16_layers,
                id_mrf_style_weight, id_mrf_content_weight, id_mrf_loss_weight=1.0,
                use_original_vgg_shape=False):
    vgg_model = vgg.build_vgg16(y_pred, use_original_vgg_shape, vgg_16_layers)

    y_pred_vgg = vgg_model(y_pred)
    y_true_vgg = vgg_model(y_true)
    content_layers = [0]
    style_layers = [1, 2]
    id_mrf_config = dict()
    id_mrf_config['crop_quarters'] = False
    id_mrf_config['max_sampling_1d_size'] = 65
    id_mrf_config['nn_stretch_sigma'] = nn_stretch_sigma
    id_mrf_style_loss = id_mrf.id_mrf_loss_sum_for_layers(y_true_vgg, y_pred_vgg, mask,
                                                          style_layers, id_mrf_config,
                                                          batch_size)

    id_mrf_content_loss = id_mrf.id_mrf_loss_sum_for_layers(y_true_vgg, y_pred_vgg, mask,
                                                            content_layers, id_mrf_config,
                                                            batch_size)

    id_mrf_loss_total = id_mrf_style_loss * id_mrf_style_weight + id_mrf_content_loss * id_mrf_content_weight
    return id_mrf_loss_weight * id_mrf_loss_total
