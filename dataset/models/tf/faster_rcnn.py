"""
Ren S. et al "`Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
<https://arxiv.org/abs/1506.01497>`_"
"""
import tensorflow as tf

from . import TFModel, VGG7
from .layers import conv_block


def rpn_loss(reg, clsf, true_reg, true_cls, anchor_batch):
    with tf.variable_scope('rpn_loss'):
        cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_cls, logits=clsf)

        anchor_batch_size_norm = tf.expand_dims(1.0 / tf.reduce_sum(anchor_batch, axis=-1), axis=0)

        cls_loss = tf.matmul(anchor_batch_size_norm, cls_loss * anchor_batch)   
        cls_loss = cls_loss / tf.cast(tf.shape(clsf)[0], dtype=tf.float32)
        cls_loss = tf.reduce_sum(cls_loss, name='cls_loss')

        sums = tf.reduce_sum((true_reg - reg) ** 2, axis=-1)

        reg_loss = sums * true_cls * anchor_batch
        reg_loss = tf.reduce_mean(reg_loss, axis=-1)
        reg_loss = tf.reduce_mean(reg_loss, name='reg_loss')

        loss = reg_loss * 100 + cls_loss
    return loss

def rcn_loss(clsf, true_cls):
    with tf.variable_scope('rcn_loss'):
        true_cls = tf.one_hot(true_cls, clsf.get_shape().as_list()[-1])
        cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels=true_cls, logits=clsf)

        #anchor_batch_size_norm = tf.expand_dims(1.0 / tf.reduce_sum(anchor_batch, axis=-1), axis=0)

        #cls_loss = tf.matmul(anchor_batch_size_norm, cls_loss * anchor_batch)
        #cls_loss = cls_loss / tf.cast(tf.shape(clsf)[0], dtype=tf.float32)
        #cls_loss = tf.reduce_sum(cls_loss, name='cls_loss')

        loss = tf.reduce_mean(cls_loss)
    return loss

def parametrize(inputs, base):
    with tf.variable_scope('parametrize'):
        y = (inputs[:, :, 0] - base[:, 0]) * (1.0 / base[:, 2])
        x = (inputs[:, :, 1] - base[:, 1]) * (1.0 / base[:, 3])
        h = tf.log(inputs[:, :, 2] * (1.0 / base[:, 2]))
        w = tf.log(inputs[:, :, 3] * (1.0 / base[:, 3]))
        output = tf.stack((y, x, h, w), axis=-1)
    return output

def unparametrize(inputs, base):
    with tf.variable_scope('parametrize'):
        y = inputs[:, :, 0] * base[:, 2] + base[:, 0]
        x = inputs[:, :, 1] * base[:, 3] + base[:, 1]
        h = tf.exp(inputs[:, :, 2]) * base[:, 2]
        w = tf.exp(inputs[:, :, 3]) * base[:, 3]
        res = tf.stack((y, x, h, w), axis=-1) 
    return res

def rpn(inputs, **kwargs):
    with tf.variable_scope('rpn'):
        net = conv_block(inputs, 'ca', filters=512, kernel_size=3, name='conv', **kwargs)
        rpn_reg = conv_block(net, 'c', filters=4*9, kernel_size=1, name='reg', **kwargs)
        rpn_cls = conv_block(net, 'c', filters=1*9, kernel_size=1, name='cls', **kwargs)
        
        spatial_net_shape = net.get_shape().as_list()[1:3]
        n_anchors = spatial_net_shape[0] * spatial_net_shape[1] * 9
        
        rpn_reg = tf.reshape(rpn_reg, [-1, n_anchors, 4])
        rpn_cls = tf.reshape(rpn_cls, [-1, n_anchors])
    return rpn_reg, rpn_cls, n_anchors

def filter_tensor(inputs, cond, *args):
    with tf.variable_scope('filter_tensor'):
        if not callable(cond):
            callable_cond = lambda x: x > cond
        else:
            callable_cond = cond
        indices = tf.where(callable_cond(inputs))
        output = (indices, *[tf.gather_nd(x, indices) for x in [inputs, *args]])
    return output

def non_max_suppression(inputs, scores, batch_size, max_output_size, score_threshold=0.7, iou_threshold=0.7):
    with tf.variable_scope('nms'):
        ix = tf.constant(0)
        filtered_rois = tf.TensorArray(dtype=tf.int32, size=batch_size)
        loop_cond = lambda ix, filtered_rois: tf.less(ix, batch_size)
        def loop_body(ix, filtered_rois): 
            indices, score, roi = filter_tensor(scores[ix], score_threshold, inputs[ix])
            roi_corners = tf.concat([roi[:, :2], roi[:, :2]+roi[:, 2:]], axis=-1)
            roi_after_nms = tf.image.non_max_suppression(roi_corners, score, max_output_size, iou_threshold)
            filtered_rois = filtered_rois.write(ix, tf.cast(tf.gather(indices, roi_after_nms), dtype=tf.int32))
            return [ix+1, filtered_rois]
        _, res = tf.while_loop(loop_cond, loop_body, [ix, filtered_rois])
        res = array_to_tuple(res, batch_size)
    return res

def array_to_tuple(inputs, size):
    with tf.variable_scope('array_to_tuple'):
        output = tf.tuple([inputs.read(i) for i in range(size)])
    return output

def get_rois_and_labels(rois, labels, indices, batch_size):
    with tf.variable_scope('get_rois_and_labels'):
        output_rois = tf.TensorArray(dtype=tf.float32, size=batch_size)
        output_labels = tf.TensorArray(dtype=tf.int32, size=batch_size)
        for i, index in enumerate(indices):
            output_rois = output_rois.write(i, tf.gather_nd(rois[i], index))
            output_labels = output_labels.write(i, tf.gather_nd(labels[i], index))
        output_rois = array_to_tuple(output_rois, batch_size)
        output_labels = array_to_tuple(output_labels, batch_size)
    return output_rois, output_labels


def roi_pooling_layer(inputs, rois, labels, factor=(1,1), shape=(7,7), name=None):
    with tf.variable_scope('roi-pooling'):
        image_index = tf.constant(0)
        output_tensor = tf.TensorArray(dtype=tf.float32, size=len(rois))
        cond_images = lambda image_index, output_tensor: tf.less(image_index, len(rois))

        for image_index in range(len(rois)):
            image = inputs[image_index]
            image_rois = rois[image_index]
            cropped_regions = tf.TensorArray(dtype=tf.float32, size=tf.shape(image_rois)[0])
            roi_index = tf.constant(0)

            cond_rois = lambda roi_index, cropped_regions: tf.less(roi_index, tf.shape(image_rois)[0])

            def roi_body(roi_index, cropped_regions):
                with tf.variable_scope('crop-from-image-{}'.format(image_index)):
                    roi = image_rois[roi_index]

                    spatial_start = roi[:2] * factor
                    spatial_size = roi[2:] * factor

                    spatial_start = tf.cast(tf.ceil(spatial_start), dtype=tf.int32)
                    spatial_size = tf.cast(tf.ceil(spatial_size), dtype=tf.int32)
                    
                    spatial_start = tf.maximum(tf.constant((0, 0)), spatial_start)            
                    spatial_start = tf.minimum(tf.shape(image)[:2]-1, spatial_start)

                    spatial_size = tf.maximum(tf.constant((0, 0)), spatial_size)
                    spatial_size = tf.minimum(tf.shape(image)[:2]-spatial_start, spatial_size)

                    start = tf.concat([spatial_start, tf.constant((0,))] , axis=0)
                    end = tf.concat([spatial_size, (tf.shape(image)[-1], )], axis=0)
                    
                    cropped = tf.slice(image, start, end)
                    cropped = tf.image.resize_images(cropped, shape)
                    cropped_regions = cropped_regions.write(roi_index, cropped)
                return [roi_index+1, cropped_regions]

            _, res = tf.while_loop(cond_rois, roi_body, [roi_index, cropped_regions])
            res = res.stack()
            output_tensor = output_tensor.write(image_index, res)

        res = array_to_tuple(output_tensor, len(rois))
        res = tf.concat(res, axis=0)
        res.set_shape([None, *shape, inputs.get_shape().as_list()[-1]])
        labels = tf.concat(labels, axis=0)
    return res, labels
