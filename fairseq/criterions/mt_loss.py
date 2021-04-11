import math
import torch
from copy import deepcopy

from . import FairseqCriterion, register_criterion
from .label_smoothed_cross_entropy import label_smoothed_nll_loss
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor


def merge_dict(origin, new, prefix):
    for key, value in new.items():
        origin[prefix + key] = value
    return origin


def get_loss(loss, sample_size):
    return loss / sample_size / math.log(2) if sample_size > 0 else 0.0

sampler = torch.distributions.Uniform(0, 1)
def lucky(prob):
    return sampler.sample().item() < prob

def _compute_loss(outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0):
    """
	outputs: batch x len x d_model
	targets: batch x len
	masks:   batch x len

	policy_logprob: if there is some policy
	    depends on the likelihood score as rewards.
    """

    def mean_ds(x: Tensor, dim=None) -> Tensor:
        return (
            x.float().mean().type_as(x)
            if dim is None
            else x.float().mean(dim).type_as(x)
        )
    if masks is not None:
        outputs, targets = outputs[masks], targets[masks]

    if masks is not None and not masks.any():
        nll_loss = torch.tensor(0)
        loss = nll_loss
    else:
        logits = F.log_softmax(outputs, dim=-1)
        if targets.dim() == 1:
            losses = F.nll_loss(logits, targets.to(logits.device), reduction='none')
        else:  # soft-labels
            losses = F.kl_div(logits, targets.to(logits.device), reduction='none')
            losses = losses.sum(-1)

        nll_loss = mean_ds(losses)
        if label_smoothing > 0:
            loss = nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
        else:
            loss = nll_loss

    loss = loss * factor
    return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

@register_criterion('mt_loss')
class LabelSmoothedMixedCrossEntropyCriterion(FairseqCriterion):
    def loss_fn(self, at_loss, nat_loss, at_sample_size, nat_sample_size, batch_sample_size):
        return self.args.at_weight * at_loss / at_sample_size * batch_sample_size + self.args.nat_weight * nat_loss / nat_sample_size * batch_sample_size

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--label-smoothing',
            default=0.1,
            type=float,
            metavar='D',
            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument(
                '--at-weight', type=float, default=0.5)
        parser.add_argument(
                '--nat-weight', type=float, default=0.5)
        parser.add_argument(
                '--at-drop-rate', type=float, default=0.5)
        parser.add_argument(
                '--nat-drop-rate', type=float, default=0.0)
        parser.add_argument(
                '--selection-criterion', type=str, default='all')
        parser.add_argument(
                '--online-kd', action='store_true', default=False)

    def __init__(self, args, task):
        super().__init__(args, task)
        # self.nat_criterion = LabelSmoothedDualImitationCriterion(args, task)
        # self.at_criterion = LabelSmoothedCrossEntropyCriterion(args, task)
        self.eps = args.label_smoothing


        self.at_last_loss = None
        self.nat_last_loss = None


    def forward(self, model, at_sample, nat_sample, at_or_nat=None, reduce=True, steps=None):
        at_model = model.at_model
        nat_model = model.nat_model

        def gen_randperm(bsz, droprate):
            nbsz = max(1, int((1.0 - droprate) * bsz))
            return torch.randperm(bsz)[:nbsz]

        def drop_sentences_(sample, rate, indexes=None):
            bsz = sample['nsentences']
            if indexes is None:
                indexes = gen_randperm(bsz, rate)
            nbsz = indexes.size(0)
            for k, v in sample['net_input'].items():
                if isinstance(v, torch.Tensor):
                    sample['net_input'][k] = v[indexes]
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    sample[k] = v[indexes]
            sample['ntokens'] = sample['ntokens'] * nbsz // bsz
            sample['nsentences'] = nbsz
            return sample
        #if steps is not None:
        #    self.args.at_weight = (300000 - steps) / 300000
        #    self.args.nat_weight = steps / 300000
        logging_output = dict()
        ntokens = at_sample['ntokens']
        nsentences = at_sample['nsentences']
        sample_size = nsentences if self.args.sentence_avg else ntokens
        logging_output['ntokens'] = ntokens
        logging_output['nsentences'] = nsentences
        logging_output['sample_size'] = sample_size
        logging_output['selection-criterion'] = self.args.selection_criterion if self.args.selection_criterion else "all"
        if at_or_nat is None or at_or_nat == "at":
            at_sample = drop_sentences_(at_sample, self.args.at_drop_rate)
            at_sample['net_input']['encoder_out'] = at_model.encoder(**at_sample['net_input'])
            at_net_output = at_model.decoder(**at_sample['net_input'])
            at_lprobs = at_model.get_normalized_probs(at_net_output, log_probs=True)
            at_target = at_model.get_targets(at_sample, at_net_output)

            at_loss, at_nll_loss = label_smoothed_nll_loss(
                at_lprobs.view(-1, at_lprobs.size(-1)), at_target.view(-1, 1), self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
            at_sample_size = at_sample['target'].size(0) if self.args.sentence_avg else at_sample['ntokens']
            at_logging_output = {
                'loss': utils.item(at_loss.data) if reduce else at_loss.data,
                'nll_loss': utils.item(at_nll_loss.data) if reduce else at_nll_loss.data,
                'ntokens': at_sample['ntokens'],
                'nsentences': at_sample['target'].size(0),
                'sample_size': at_sample_size,
            }
            logging_output = merge_dict(logging_output, at_logging_output, "at-")
            if at_or_nat == "at":
                logging_output['at-weight'] = 1.0
                logging_output['nat-weight'] = 0.0
                return at_loss / at_sample_size * sample_size, sample_size, logging_output

        if at_or_nat is None or at_or_nat == "nat":
            nat_sample['net_input']['encoder_out'] = nat_model.encoder(**nat_sample['net_input'])
            tgt_tokens, prev_output_tokens = nat_sample["target"], nat_sample["prev_target"]

            length_out = nat_model.decoder.forward_length(normalize=False, encoder_out=nat_sample['net_input']['encoder_out'])
            length_tgt = nat_model.decoder.forward_length_prediction(length_out, nat_sample['net_input']['encoder_out'], tgt_tokens)
            word_ins_out = nat_model.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=nat_sample['net_input']['encoder_out'])
            word_ins_mask = prev_output_tokens.eq(nat_model.unk)
            outputs = {
                "word_ins": {
                    "out": word_ins_out, "tgt": tgt_tokens,
                    "mask": word_ins_mask, "ls": self.eps,
                    "nll_loss": True
                },
                "length": {
                    "out": length_out, "tgt": length_tgt,
                    "factor": nat_model.decoder.length_loss_factor
                }
            }
            losses, nll_losses = [], []
            for obj in outputs:
                if outputs[obj].get("loss", None) is None:
                    _losses = _compute_loss(
                        outputs[obj].get("out"),
                        outputs[obj].get("tgt"),
                        outputs[obj].get("mask", None),
                        outputs[obj].get("ls", 0.0),
                        name=obj + '-loss',
                        factor=outputs[obj].get("factor", 1.0)
                    )
                else:
                    assert False

                losses += [_losses]
                if outputs[obj].get("nll_loss", False):
                    nll_losses += [_losses.get("nll_loss", 0.0)]

            nat_loss = sum(l["loss"] for l in losses)
            nat_nll_loss = sum(l for l in nll_losses) if len(nll_losses) > 0 \
                else loss.new_tensor(0)
            nat_nll_loss = nat_nll_loss

            # NOTE:
            # we don't need to use sample_size as denominator for the gradient
            # here sample_size is just used for logging
            nat_sample_size = 1
            nat_logging_output = {
                "loss": utils.item(nat_loss.data) if reduce else nat_loss.data,
                "nll_loss": utils.item(nat_nll_loss.data) if reduce else nat_nll_loss.data,
                "ntokens": nat_sample["ntokens"],
                "nsentences": nat_sample["nsentences"],
                "sample_size": nat_sample_size,
            }

            for l in losses:
                nat_logging_output[l["name"]] = (
                    utils.item(l["loss"].data / l["factor"])
                    if reduce
                    else l[["loss"]].data / l["factor"]
                )
            logging_output = merge_dict(logging_output, nat_logging_output, "nat-")
            if at_or_nat == "nat":
                logging_output['nat-weight'] = 1.0
                logging_output['at-weight'] = 0.0
                return nat_loss / nat_sample_size * sample_size, sample_size, logging_output
        if steps is not None:
            setattr(self.args, 'nat_weight', (steps/self.args.max_update))
            setattr(self.args, 'at_weight', (1 - steps/self.args.max_update))
        logging_output['nat-weight'] = self.args.nat_weight
        logging_output['at-weight'] = self.args.at_weight
        return self.loss_fn(at_loss, nat_loss, at_sample_size, nat_sample_size, sample_size), sample_size, logging_output
    

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        at_ntokens = sum(log.get("at-ntokens", 0) for log in logging_outputs)
        at_nsentences = sum(log.get("at-nsentences", 0) for log in logging_outputs)
        at_sample_size = sum(log.get("at-sample_size", 0) for log in logging_outputs)
        at_loss = sum(log.get("at-loss", 0) for log in logging_outputs)
        at_nll_loss = sum(log.get("at-nll_loss", 0) for log in logging_outputs)
        at_weights = [log.get("at-weight", 0) for log in logging_outputs]
        at_weight = max(at_weights)

        nat_ntokens = sum(log.get("nat-ntokens", 0) for log in logging_outputs)
        nat_nsentences = sum(log.get("nat-nsentences", 0) for log in logging_outputs)
        nat_sample_size = sum(log.get("nat-sample_size", 0) for log in logging_outputs)
        nat_loss = sum(log.get("nat-loss", 0) for log in logging_outputs)
        nat_nll_loss = sum(log.get("nat-nll_loss", 0) for log in logging_outputs)
        nat_weights = [log.get("nat-weight", 0) for log in logging_outputs]
        nat_weight = max(nat_weights)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)

        loss = get_loss((at_weight * at_loss / at_sample_size * sample_size if at_sample_size > 0 else 0.0) + (nat_weight * nat_loss / nat_sample_size * sample_size if nat_sample_size > 0 else 0.0), sample_size)
        nll_loss = get_loss((at_weight * at_nll_loss / at_sample_size * sample_size if at_sample_size > 0 else 0.0) + (nat_weight * nat_nll_loss / nat_sample_size * sample_size if nat_sample_size > 0 else 0.0), sample_size)
        #criteria = set(log.get('selection-criterion') for log in logging_outputs)
        #assert len(criteria) == 1
        criteria = "nat"
        at_loss = get_loss(at_loss, at_sample_size)
        nat_loss = get_loss(nat_loss, nat_sample_size)
        at_nll_loss = get_loss(at_nll_loss, at_sample_size)
        nat_nll_loss = get_loss(nat_nll_loss, nat_sample_size)
        return {
            "loss": at_loss if criteria == "at" else (nat_loss if criteria == "nat" else (max(at_loss, nat_loss) if criteria == "max" else loss)),
            "nll_loss": at_nll_loss if criteria == "at" else (nat_nll_loss if criteria == "nat" else (max(at_nll_loss, nat_nll_loss) if criteria == "max" else nll_loss)),
            "at_nat_loss": loss,
            "at_loss": at_loss,
            "nat_loss": nat_loss,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "at_weight": at_weight,
            "nat_weight": nat_weight,
            "at_sample_size": at_sample_size,
            "nat_sample_size": nat_sample_size,
        }
