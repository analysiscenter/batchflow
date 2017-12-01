#pylint:disable=cell-var-from-loop

"""
Ren S. et al "`Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
<https://arxiv.org/abs/1506.01497>`_"
"""
import tensorflow as tf
import numpy as np

from . import TFModel, VGG7
from .layers import conv_block

_IOU_LOW = 0.3
_IOU_HIGH = 0.7


def rpn_loss(reg, clsf, true_reg, true_cls, anchor_batch):
    """ Mixed MSE+CE Loss for RPN. """
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
    """ CE loss for RCN. """
    with tf.variable_scope('rcn_loss'):
        true_cls = tf.one_hot(true_cls, clsf.get_shape().as_list()[-1])
        cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels=true_cls, logits=clsf)

        #anchor_batch_size_norm = tf.expand_dims(1.0 / tf.reduce_sum(anchor_batch, axis=-1), axis=0)

        #cls_loss = tf.matmul(anchor_batch_size_norm, cls_loss * anchor_batch)
        #cls_loss = cls_loss / tf.cast(tf.shape(clsf)[0], dtype=tf.float32)
        #cls_loss = tf.reduce_sum(cls_loss, name='cls_loss')

        loss = tf.reduce_mean(cls_loss)
    return loss

def _parametrize(inputs, base):
    with tf.variable_scope('parametrize'):
        y = (inputs[:, :, 0] - base[:, 0]) * (1.0 / base[:, 2])
        x = (inputs[:, :, 1] - base[:, 1]) * (1.0 / base[:, 3])
        height = tf.log(inputs[:, :, 2] * (1.0 / base[:, 2]))
        width = tf.log(inputs[:, :, 3] * (1.0 / base[:, 3]))
        output = tf.stack((y, x, height, width), axis=-1)
    return output

def _unparametrize(inputs, base):
    with tf.variable_scope('parametrize'):
        y = inputs[:, :, 0] * base[:, 2] + base[:, 0]
        x = inputs[:, :, 1] * base[:, 3] + base[:, 1]
        height = tf.exp(inputs[:, :, 2]) * base[:, 2]
        width = tf.exp(inputs[:, :, 3]) * base[:, 3]
        res = tf.stack((y, x, height, width), axis=-1)
    return res

def _rpn(inputs, **kwargs):
    with tf.variable_scope('rpn'):
        net = conv_block(inputs, 'ca', filters=512, kernel_size=3, name='conv', **kwargs)
        rpn_reg = conv_block(net, 'c', filters=4*9, kernel_size=1, name='reg', **kwargs)
        rpn_cls = conv_block(net, 'c', filters=1*9, kernel_size=1, name='cls', **kwargs)

        spatial_net_shape = net.get_shape().as_list()[1:3]
        n_anchors = spatial_net_shape[0] * spatial_net_shape[1] * 9

        rpn_reg = tf.reshape(rpn_reg, [-1, n_anchors, 4])
        rpn_cls = tf.reshape(rpn_cls, [-1, n_anchors])
    return rpn_reg, rpn_cls, n_anchors

def _filter_tensor(inputs, cond, *args):
    with tf.variable_scope('filter_tensor'):
        if not callable(cond):
            callable_cond = lambda x: x > cond
        else:
            callable_cond = cond
        indices = tf.where(callable_cond(inputs))
        output = (indices, *[tf.gather_nd(x, indices) for x in [inputs, *args]])
    return output

def non_max_suppression(inputs, scores, batch_size, max_output_size,
                        score_threshold=0.7, iou_threshold=0.7, nonempty=False):
    """ Perform NMS on batch of images. """
    with tf.variable_scope('nms'):
        ix = tf.constant(0)
        filtered_rois = tf.TensorArray(dtype=tf.int32, size=batch_size, infer_shape=False)
        loop_cond = lambda ix, filtered_rois: tf.less(ix, batch_size)
        def _loop_body(ix, filtered_rois):
            indices, score, roi = _filter_tensor(scores[ix], score_threshold, inputs[ix]) # pylint: disable=unbalanced-tuple-unpacking
            roi_corners = tf.concat([roi[:, :2], roi[:, :2]+roi[:, 2:]], axis=-1)
            roi_after_nms = tf.image.non_max_suppression(roi_corners, score, max_output_size, iou_threshold)
            if nonempty:
                is_not_empty = lambda: filtered_rois.write(ix,
                                                           tf.cast(tf.gather(indices, roi_after_nms),
                                                                   dtype=tf.int32))
                is_empty = lambda: filtered_rois.write(ix, tf.constant([[0]]))
                filtered_rois = tf.cond(tf.not_equal(tf.shape(indices)[0], 0), is_not_empty, is_empty)
            else:
                filtered_rois = filtered_rois.write(ix, tf.cast(tf.gather(indices, roi_after_nms), dtype=tf.int32))
            return [ix+1, filtered_rois]
        _, res = tf.while_loop(loop_cond, _loop_body, [ix, filtered_rois])
        res = _array_to_tuple(res, batch_size, [-1, 1])
    return res

def _array_to_tuple(inputs, size, shape=None):
    with tf.variable_scope('array_to_tuple'):
        if shape is None:
            output = tf.tuple([inputs.read(i) for i in range(size)])
        else:
            output = tf.tuple([tf.reshape(inputs.read(i), shape) for i in range(size)])
    return output

def _get_rois_and_labels(rois, labels, indices, batch_size):
    with tf.variable_scope('get_rois_and_labels'):
        output_rois = tf.TensorArray(dtype=tf.float32, size=batch_size)
        output_labels = tf.TensorArray(dtype=tf.int32, size=batch_size)
        for i, index in enumerate(indices):
            output_rois = output_rois.write(i, tf.gather_nd(rois[i], index))
            output_labels = output_labels.write(i, tf.gather_nd(labels[i], index))
        output_rois = _array_to_tuple(output_rois, batch_size)
        output_labels = _array_to_tuple(output_labels, batch_size)
    return output_rois, output_labels


def roi_pooling_layer(inputs, rois, factor=(1, 1), shape=(7, 7), name='roi-pooling'):
    """ ROI pooling layer with resize. """
    with tf.variable_scope(name):
        image_index = tf.constant(0)
        output_tuple = tf.TensorArray(dtype=tf.float32, size=len(rois))

        for image_index, image_rois in enumerate(rois):
            image = inputs[image_index]
            cropped_regions = tf.TensorArray(dtype=tf.float32, size=tf.shape(image_rois)[0])
            roi_index = tf.constant(0)

            cond_rois = lambda roi_index, cropped_regions: tf.less(roi_index, tf.shape(image_rois)[0])

            def _roi_body(roi_index, cropped_regions):
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

                    start = tf.concat([spatial_start, tf.constant((0,))], axis=0)
                    end = tf.concat([spatial_size, (tf.shape(image)[-1], )], axis=0)

                    cropped = tf.slice(image, start, end)
                    cropped = tf.image.resize_images(cropped, shape)
                    cropped.set_shape([*shape, image.get_shape().as_list()[-1]])
                    cropped_regions = cropped_regions.write(roi_index, cropped)
                return [roi_index+1, cropped_regions]

            _, res = tf.while_loop(cond_rois, _roi_body, [roi_index, cropped_regions])
            res = res.stack()
            output_tuple = output_tuple.write(image_index, res)
        res = _array_to_tuple(output_tuple, len(rois))
    return res


def _stack_tuple(inputs, *args):
    tuple_size = len(inputs)
    tensor_sizes = [tf.shape(inputs[i])[0] for i in range(tuple_size)]
    outputs = [tf.concat(x, axis=0) for x in [inputs, *args]]
    return (tensor_sizes, *outputs)

def _unstack_tuple(inputs, tensor_sizes):
    size = len(tensor_sizes)
    start_position = tf.constant(0)
    output = []
    for i in range(size):
        output.append(tf.slice(inputs, begin=[start_position, 0], size=[tensor_sizes[i], -1]))
        start_position = start_position + tensor_sizes[i]
    return tf.tuple(output)


class RPN(TFModel):
    """ Region Propoasl Network """
    MAP_SHAPE = None

    @classmethod
    def default_config(cls):
        config = TFModel.default_config()

        config['output']['prefix'] = ['reg', 'cls']

        return config

    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['map_shape'] = config['map_shape']
        config['head']['image_shape'] = np.array(config['inputs']['images']['shape'][:2])
        return config

    @classmethod
    def input_block(cls, inputs, name='input_block', **kwargs):
        return VGG7.body(inputs, **kwargs)

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        return conv_block(inputs, 'ca', filters=512, kernel_size=3, name='feature_maps', **kwargs)

    @classmethod
    def head(cls, inputs, name='head', **kwargs):
        _ = name
        map_shape = np.array(inputs.get_shape().as_list()[1:3])
        n_anchors = map_shape[0] * map_shape[1] * 9
        kwargs['map_shape'].append(map_shape)
        kwargs['map_shape'].append(n_anchors)

        tf.placeholder(tf.float32, shape=[n_anchors, 4], name='anchors')
        tf.placeholder(tf.float32, shape=[None, n_anchors], name='anchor_batch')
        tf.placeholder(tf.float32, shape=[None, n_anchors], name='anchor_clsf')
        tf.placeholder(tf.int32, shape=[None, n_anchors], name='anchor_labels')
        tf.placeholder(tf.float32, shape=[None, n_anchors, 4], name='anchor_reg')

        train_mode = tf.placeholder(tf.bool, shape=(), name='train_mode')

        rpn_reg, rpn_clsf, loss1 = cls._rpn_head(inputs, **kwargs)
        _, loss2 = cls._rcn_head([inputs, rpn_reg, rpn_clsf], **kwargs)

        loss = tf.cond(train_mode, lambda: loss1, lambda: loss2)

        tf.losses.add_loss(loss)

        return rpn_reg, rpn_clsf


    @classmethod
    def _rpn_head(cls, inputs, name='rpn_head', **kwargs):
        n_anchors = kwargs['map_shape'][1]

        scope_name = tf.get_default_graph().get_name_scope()
        anchors = tf.get_default_graph().get_tensor_by_name(scope_name+'/anchors:0')
        anchor_reg = tf.get_default_graph().get_tensor_by_name(scope_name+'/anchor_reg:0')
        anchor_clsf = tf.get_default_graph().get_tensor_by_name(scope_name+'/anchor_clsf:0')
        anchor_batch = tf.get_default_graph().get_tensor_by_name(scope_name+'/anchor_batch:0')

        with tf.variable_scope(name):

            rpn_reg = conv_block(inputs, 'c', filters=4*9, kernel_size=1, name='conv_reg', **kwargs)
            rpn_clsf = conv_block(inputs, 'c', filters=1*9, kernel_size=1, name='conv_clsf', **kwargs)
            rpn_reg = tf.reshape(rpn_reg, [-1, n_anchors, 4])
            rpn_clsf = tf.reshape(rpn_clsf, [-1, n_anchors])
            anchor_reg_param = _parametrize(anchor_reg, anchors)

            loss = rpn_loss(rpn_reg, rpn_clsf, anchor_reg_param, anchor_clsf, anchor_batch)
            loss = tf.identity(loss, 'loss')

            rpn_reg = tf.identity(_unparametrize(rpn_reg, anchors), 'reg')
            rpn_clsf = tf.sigmoid(rpn_clsf, 'clsf')

        return rpn_reg, rpn_clsf, loss

    @classmethod
    def _rcn_head(cls, inputs, name='rcn_head', **kwargs):
        scope_name = tf.get_default_graph().get_name_scope()
        anchors_labels = tf.get_default_graph().get_tensor_by_name(scope_name+'/anchor_labels:0')

        feature_maps, rpn_reg, rpn_cls = inputs
        map_shape = kwargs['map_shape'][0]
        image_shape = kwargs['image_shape']
        n_anchors = map_shape[0] * map_shape[1] * 9

        batch_size = 16 #rpn_reg.get_shape().as_list()[0] # ??????????????????

        with tf.variable_scope(name):
            rcn_input_indices = non_max_suppression(rpn_reg, rpn_cls, batch_size, n_anchors,
                                                    iou_threshold=0.4, score_threshold=0.7, nonempty=True)

            rcn_input_rois, rcn_input_labels = _get_rois_and_labels(rpn_reg, anchors_labels, 
                                                                    rcn_input_indices, batch_size)
            roi_factor = np.array(map_shape/image_shape)

            #rcn_input_rois = stop_gradient_tuple(rcn_input_rois)
            #rcn_input_labels = stop_gradient_tuple(rcn_input_labels)

            roi_cropped = roi_pooling_layer(feature_maps, rcn_input_rois, factor=roi_factor, shape=(7, 7))
            indices, roi_cropped, rcn_input_labels = _stack_tuple(roi_cropped, rcn_input_labels) # pylint: disable=unbalanced-tuple-unpacking
            rcn_cls = conv_block(roi_cropped, 'f', units=10, name='output_conv')

            loss = rcn_loss(rcn_cls, rcn_input_labels)

            rcn_cls = _unstack_tuple(rcn_cls, indices)
            rcn_cls = tf.tuple(rcn_cls, name='clsf')
            loss = tf.identity(loss, 'loss')

        return rcn_cls, loss


    def _fill_feed_dict(self, feed_dict=None, is_training=True):

        bboxes = feed_dict.pop('bboxes')
        labels = feed_dict.pop('labels')
        map_shape = self.config['map_shape'][0]
        image_shape = feed_dict['images'].shape[1:3]

        feed_dict = super()._fill_feed_dict(feed_dict, is_training)

        anchors = self.create_anchors(image_shape, map_shape)
        anchor_reg, anchor_clsf, anchor_labels = self.create_rpn_inputs(anchors, bboxes, labels)
        anchor_batch = self.create_batch(anchor_clsf)
        anchor_clsf = np.array(anchor_clsf == 1, dtype=np.int32)

        feed_dict = {**feed_dict,
                     self._map_name('RPN/anchors'): anchors,
                     self._map_name('RPN/anchor_reg'): anchor_reg,
                     self._map_name('RPN/anchor_clsf'): anchor_clsf,
                     self._map_name('RPN/anchor_labels'): anchor_labels,
                     self._map_name('RPN/anchor_batch'): anchor_batch}


        return feed_dict

    @classmethod
    def create_anchors(cls, image_shape, map_shape, scales=(4, 8, 16), ratio=2):
        """ Create anchors for image_shape depending on output_map_shape. """
        ratios = ((np.sqrt(ratio), 1/np.sqrt(ratio)),
                  (1, 1),
                  (1/np.sqrt(ratio), np.sqrt(ratio)))

        anchors = []
        for scale in scales:
            for ratio in ratios:
                image_height, image_width = image_shape
                map_height, map_width = map_shape
                n = map_height * map_width

                j = np.array(list(range(map_height)))
                j = np.expand_dims(j, 1)
                j = np.tile(j, (1, map_width))
                j = j.reshape((-1))

                i = np.array(list(range(map_width)))
                i = np.expand_dims(i, 0)
                i = np.tile(i, (map_height, 1))
                i = i.reshape((-1))

                s = np.ones((n)) * scale
                ratio0 = np.ones((n)) * ratio[0]
                ratio1 = np.ones((n)) * ratio[1]

                height = s * ratio0
                width = s * ratio1
                y = (j + 0.5) * image_height / map_height - height * 0.5
                x = (i + 0.5) * image_width / map_width - width * 0.5

                y, x = [np.maximum(vector, np.zeros((n))) for vector in [y, x]]
                height = np.minimum(height, image_height-y)
                width = np.minimum(width, image_width-x)

                cur_anchors = [np.expand_dims(vector, 1) for vector in [y, x, height, width]]
                cur_anchors = np.concatenate(cur_anchors, axis=1)
                anchors.append(np.array(cur_anchors, np.int32))

        anchors = np.array(anchors).transpose(1, 0, 2).reshape(-1, 4)
        return anchors

    @classmethod
    def create_rpn_inputs(cls, anchors, bboxes, labels):
        """ Create reg and clsf targets of RPN. """
        anchor_reg = []
        anchor_clsf = []
        anchor_labels = []
        for ind, image_bboxes in enumerate(bboxes): # TODO: for -> np
            image_labels = labels[ind]

            n = anchors.shape[0]
            k = image_bboxes.shape[0]

            # Compute the IoUs of the anchors and ground truth boxes
            tiled_anchors = np.tile(np.expand_dims(anchors, 1), (1, k, 1))
            tiled_bboxes = np.tile(np.expand_dims(image_bboxes, 0), (n, 1, 1))

            tiled_anchors = tiled_anchors.reshape((-1, 4))
            tiled_bboxes = tiled_bboxes.reshape((-1, 4))

            ious = cls.iou_bbox(tiled_anchors, tiled_bboxes)[0]
            ious = ious.reshape(n, k)

            # Label each anchor based on its max IoU
            max_ious = np.max(ious, axis=1)
            best_bbox_for_anchor = np.argmax(ious, axis=1)

            anchor_reg.append(image_bboxes[best_bbox_for_anchor])
            anchor_labels.append(image_labels[best_bbox_for_anchor].reshape(-1))

            # anchor has at least one gt-bbox with IoU >_IOU_HIGH
            image_clsf = np.array(max_ious > _IOU_HIGH, dtype=np.int32)

            # anchor intersects with at least one bbox
            best_anchor_for_bbox = np.argmax(ious, axis=0)
            image_clsf[best_anchor_for_bbox] = 1

            # max IoU for anchor < _IOU_LOW
            image_clsf[np.logical_and(max_ious < _IOU_LOW, image_clsf == 0)] = -1
            anchor_clsf.append(image_clsf)
        return np.array(anchor_reg), np.array(anchor_clsf), np.array(anchor_labels)

    @classmethod
    def iou_bbox(cls, bboxes1, bboxes2):
        """ Compute the IoUs between bounding boxes. """
        bboxes1 = np.array(bboxes1, np.float32)
        bboxes2 = np.array(bboxes2, np.float32)

        intersection_min_y = np.maximum(bboxes1[:, 0], bboxes2[:, 0])
        intersection_max_y = np.minimum(bboxes1[:, 0] + bboxes1[:, 2] - 1, bboxes2[:, 0] + bboxes2[:, 2] - 1)
        intersection_height = np.maximum(intersection_max_y - intersection_min_y + 1, np.zeros_like(bboxes1[:, 0]))

        intersection_min_x = np.maximum(bboxes1[:, 1], bboxes2[:, 1])
        intersection_max_x = np.minimum(bboxes1[:, 1] + bboxes1[:, 3] - 1, bboxes2[:, 1] + bboxes2[:, 3] - 1)
        intersection_width = np.maximum(intersection_max_x - intersection_min_x + 1, np.zeros_like(bboxes1[:, 1]))

        area_intersection = intersection_height * intersection_width
        area_first = bboxes1[:, 2] * bboxes1[:, 3]
        area_second = bboxes2[:, 2] * bboxes2[:, 3]
        area_union = area_first + area_second - area_intersection

        iou = area_intersection * 1.0 / area_union
        iof = area_intersection * 1.0 / area_first
        ios = area_intersection * 1.0 / area_second

        return iou, iof, ios

    @classmethod
    def create_batch(cls, anchor_clsf, batch_size=64):
        """ Create batch indices for anchors. """
        anchor_batch = []
        for clsf in anchor_clsf:
            batch_size = min(batch_size, len(clsf))
            positive = clsf == 1
            negative = clsf == -1
            if sum(positive) + sum(negative) < batch_size:
                batch_size = sum(positive) + sum(negative)
            if sum(positive) < batch_size / 2:
                positive_batch_size = sum(positive)
                negative_batch_size = batch_size - sum(positive)
            elif sum(negative) < batch_size / 2:
                positive_batch_size = batch_size - sum(negative)
                negative_batch_size = sum(negative)
            else:
                positive_batch_size = batch_size // 2
                negative_batch_size = batch_size // 2

            p = positive / sum(positive)
            positive_batch = np.random.choice(len(clsf), size=positive_batch_size, replace=False, p=p)
            p = negative / sum(negative)
            negative_batch = np.random.choice(len(clsf), size=negative_batch_size, replace=False, p=p)
            image_anchor_batch = np.array([False]*len(clsf))
            image_anchor_batch[positive_batch] = True
            image_anchor_batch[negative_batch] = True
            anchor_batch.append(image_anchor_batch)
        return np.array(anchor_batch)
