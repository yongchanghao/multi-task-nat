# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy

import torch

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from fairseq.tasks.translation_lev import TranslationLevenshteinTask
from fairseq.utils import new_arange


@register_task('translation_mt')
class TranslationSemiAutoRegressiveTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete', 'random_mask', 'no_noise', 'full_mask'])
        parser.add_argument(
            '--generator',
            default="none",
            choices=["at", "nat", "none"]
        )
        parser.add_argument(
            '--mode-switch-updates', default=0, type=int,
            help='after how many steps to switch at/nat criterion, 0 for no switches'
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = self.args.generator

    def decide_at_or_nat(self, steps):
        if self.args.mode_switch_updates > 0:
            return "at" if (steps // self.args.mode_switch_updates) % 2 else "nat"
        else:
            return None

    def build_generator(self, args):
        if self.generator == "at":
            if getattr(args, 'score_reference', False):
                from fairseq.sequence_scorer import SequenceScorer
                return SequenceScorer(self.target_dictionary)
            else:
                from fairseq.sequence_generator import SequenceGenerator, SequenceGeneratorWithAlignment
                if getattr(args, 'print_alignment', False):
                    seq_gen_cls = SequenceGeneratorWithAlignment
                else:
                    seq_gen_cls = SequenceGenerator
                return seq_gen_cls(
                    self.target_dictionary,
                    beam_size=getattr(args, 'beam', 5),
                    max_len_a=getattr(args, 'max_len_a', 0),
                    max_len_b=getattr(args, 'max_len_b', 200),
                    min_len=getattr(args, 'min_len', 1),
                    normalize_scores=(not getattr(args, 'unnormalized', False)),
                    len_penalty=getattr(args, 'lenpen', 1),
                    unk_penalty=getattr(args, 'unkpen', 0),
                    sampling=getattr(args, 'sampling', False),
                    sampling_topk=getattr(args, 'sampling_topk', -1),
                    sampling_topp=getattr(args, 'sampling_topp', -1.0),
                    temperature=getattr(args, 'temperature', 1.),
                    diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                    diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
                    match_source_len=getattr(args, 'match_source_len', False),
                    no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
                )
        elif self.generator == "nat":
            from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
            return IterativeRefinementGenerator(
                self.target_dictionary,
                eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
                max_iter=getattr(args, 'iter_decode_max_iter', 10),
                beam_size=getattr(args, 'iter_decode_with_beam', 1),
                reranking=getattr(args, 'iter_decode_with_external_reranker', False),
                decoding_format=getattr(args, 'decoding_format', None),
                adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
                retain_history=getattr(args, 'retain_iter_history', False))
        else:
            return None
            raise NotImplementedError("Please specify generator type by using '--generator"
                                      " option")
    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            if self.args.generator == "at":
                models = [m.at_model for m in models]
            elif self.args.generator == "nat":
                models = [m.nat_model for m in models]
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)


    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(os.pathsep)
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            src_prepend_bos=True, tgt_prepend_bos=True,
        )

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True)

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = 2 + ((target_length - 2) * target_score.new_zeros(
                target_score.size(0), 1).uniform_()).long()
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = target_tokens.gather(
                1, target_rank).masked_fill_(target_cutoff, pad).gather(
                    1,
                    target_rank.masked_fill_(target_cutoff,
                                             max_len).sort(1)[1])
            prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.
                                                    ne(pad).sum(1).max()]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = target_tokens.ne(pad) & \
                           target_tokens.ne(bos) & \
                           target_tokens.ne(eos)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk)
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = target_tokens.eq(bos) | target_tokens.eq(
                eos) | target_tokens.eq(pad)
            return target_tokens.masked_fill(~target_mask, unk)

        if self.args.noise == 'random_delete':
            return _random_delete(target_tokens)
        elif self.args.noise == 'random_mask':
            return _random_mask(target_tokens)
        elif self.args.noise == 'full_mask':
            return _full_mask(target_tokens)
        elif self.args.noise == 'no_noise':
            return target_tokens
        else:
            raise NotImplementedError



    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   ignore_grad=False, steps=None):
        model.train()
        at_sample = deepcopy(sample)
        nat_sample = sample
        nat_sample['prev_target'] = self.inject_noise(nat_sample['target'])
        loss, sample_size, logging_output = criterion(model, at_sample, nat_sample, self.decide_at_or_nat(steps), steps=steps)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            at_sample = deepcopy(sample)
            nat_sample = sample
            # bsz = at_sample['nsentences']
            # at_sample['ntokens'] -= bsz
            # at_sample['net_input']['prev_output_tokens'] = torch.cat([
            #     at_sample['net_input']['prev_output_tokens'][:, :1], at_sample['net_input']['prev_output_tokens'][:, 2:]
            # ], dim=1)
            # at_sample['target'] = at_sample['target'][:, 1:]
            nat_sample['prev_target'] = self.inject_noise(nat_sample['target'])
            loss, sample_size, logging_output = criterion(model, at_sample, nat_sample)
        return loss, sample_size, logging_output
