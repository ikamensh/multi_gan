import os
from config import generated_dir
from tf_reader import records
from ilya_ezplot import Metric, plot_group


def to_metrics(path):

    metrics = {}

    for tag, step, value in records(path):
        m: Metric = metrics.setdefault(tag, Metric('step', tag))
        m.add_record(step, value)

    return metrics


summary_dir = os.path.join(generated_dir, 'summaries')

from discriminator.discriminator import GanMetrics

losses = (GanMetrics.real_loss, GanMetrics.fake_loss, GanMetrics.aux_loss)
accuracies = (GanMetrics.real_acc, GanMetrics.fake_acc)

def main():
    filename = os.listdir(summary_dir)[0]
    path = os.path.join(summary_dir, filename)
    ms = to_metrics(path)

    plot_group({k:v for k,v in ms.items() if k in losses}, os.path.join(generated_dir, 'plots'), 'losses')
    plot_group({k:v for k,v in ms.items() if k in accuracies}, os.path.join(generated_dir, 'plots'), 'accuracies')

if __name__ == '__main__':
    main()

