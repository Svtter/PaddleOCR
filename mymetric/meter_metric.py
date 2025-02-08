from metermetrics.counter import Counter

from ppocr.metrics.rec_metric import Levenshtein, RecMetric


class MyCounter(Counter):
    def reset(self):
        self.pred_list = []
        self.true_list = []
        self.err_list = []
        self.cnt = 0


class MeterRecMetric(RecMetric):
    def __init__(self, main_indicator="acc", **kwargs):
        super().__init__(main_indicator, **kwargs)
        # 初始化 Counter
        self.counter = Counter()
        self.reset()

    def reset(self):
        """重置指标"""
        super().reset()
        self.counter.reset()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            self.counter.add_val(pred, target)
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis

    def get_metric(self):
        """返回评估指标结果"""
        metrics = super().get_metric()

        metrics["meter_acc"] = self.counter.get_acc()
        metrics["meter_sacc"] = self.counter.get_strict_acc()
        metrics["meter_mse"] = self.counter.get_mse()
        metrics["meter_mae"] = self.counter.get_mae()
        metrics["meter_rmse"] = self.counter.get_rmse()
        metrics["meter_max_error"] = self.counter.get_max_error()
        metrics["meter_max_abs_err"] = self.counter.get_max_abs_err()

        return metrics
