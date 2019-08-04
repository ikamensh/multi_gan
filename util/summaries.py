import os


# import tensorflow as tf
#
# summary_dir = 'tmp/summaries'
# summary_writer = tf.summary.create_file_writer('tmp/summaries')
#
# with summary_writer.as_default():
#   tf.summary.scalar('loss', 0.1, step=42)
#   tf.summary.scalar('loss', 0.2, step=43)
#   tf.summary.scalar('loss', 0.3, step=44)
#   tf.summary.scalar('loss', 0.4, step=45)


from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.framework import tensor_util

def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)

from config import generated_dir
summary_dir = os.path.join(generated_dir, 'summaries')
from ilya_ezplot import Metric, ez_plot

def to_metrics(path):

    metrics = {}
    # ctr = 0

    for event in my_summary_iterator(path):
        for value in event.summary.value:

            tag = value.tag
            m: Metric = metrics.setdefault(tag, Metric('step', tag))
            t = tensor_util.MakeNdarray(value.tensor)
            m.add_record(event.step, float(t))

        # ctr +=1
        # if ctr > 300000:
        #     break

    return metrics


filename = os.listdir(summary_dir)[0]
path = os.path.join(summary_dir, filename)
ms = to_metrics(path)

for tag, metric in ms.items():
    ez_plot(metric, 'plots_sm2')


# for filename in os.listdir(summary_dir):
#     path = os.path.join(summary_dir, filename)
#     for event in my_summary_iterator(path):
#         for value in event.summary.value:
#             t = tensor_util.MakeNdarray(value.tensor)
#             print(value.tag, event.step, t)

