import pandas as pd
from mmseg.core import EvalHook, DistEvalHook
import warnings
import datetime
import os


class EvalHookNew(EvalHook):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_list = []
        timestamp = datetime.datetime.now().__str__().split('.')[0][:-3].replace(' ', '_').replace(':', '-')
        print(f'Timestamp: {timestamp}')
        self.data_log_dir = f'/gpfs/work5/0/prjs0790/data/run_outputs/evaluation/vitada_run_{timestamp}/'
        os.makedirs(self.data_log_dir)
    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import single_gpu_test
        results = single_gpu_test(
            runner.model, self.dataloader, show=False, pre_eval=self.pre_eval)
        runner.log_buffer.clear()

        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)

    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)

        data_dic = {}
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
            data_dic[name] = val
        runner.log_buffer.ready = True
        self.data_list.append(data_dic)
        pd.DataFrame(self.data_list).to_csv(self.data_log_dir + 'metrics_log.csv', index=False)

        if self.save_best is not None:
            # If the performance of model is pool, the `eval_res` may be an
            # empty dict and it will raise exception when `self.save_best` is
            # not None. More details at
            # https://github.com/open-mmlab/mmdetection/issues/6265.
            if not eval_res:
                warnings.warn(
                    'Since `eval_res` is an empty dict, the behavior to save '
                    'the best checkpoint will be skipped in this evaluation.')
                return None

            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None