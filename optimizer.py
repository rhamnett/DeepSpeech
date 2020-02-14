#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import sys
import statistics

import optuna
import absl.app

from DeepSpeech import create_model, try_loading
from evaluate import evaluate
from util.config import initialize_globals
from util.flags import create_flags, FLAGS
from util.logging import log_error


def objective(trial):
    FLAGS.lm_alpha = trial.suggest_uniform('lm_alpha', 0, FLAGS.lm_alpha_max)
    FLAGS.lm_beta = trial.suggest_uniform('lm_beta', 0, FLAGS.lm_beta_max)

    samples = evaluate(FLAGS.test_files.split(','), create_model, try_loading)

    return statistics.mean([sample.wer for sample in samples])

def main(_):
    initialize_globals()

    if not FLAGS.test_files:
        log_error('You need to specify what files to use for evaluation via '
                  'the --test_files flag.')
        sys.exit(1)

    study = optuna.create_study()
    study.optimize(objective, n_jobs=1, n_trials=FLAGS.n_trials)
    print('Best params: lm_alpha={} and lm_beta={} with WER={}'.format(study.best_params['lm_alpha'],
                                                                       study.best_params['lm_beta'],
                                                                       study.best_value))


if __name__ == '__main__':
    create_flags()
    absl.app.run(main)
