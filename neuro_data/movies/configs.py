from collections import OrderedDict
from itertools import product, count
from pprint import pformat

from attorch.dataloaders import RepeatsBatchSampler
from torch.utils.data.sampler import SubsetRandomSampler

from .data_schemas import MovieMultiDataset
from .transforms import Subsample, Normalizer, ToTensor, Subsequence
from ..utils.sampler import SubsetSequentialSampler, BalancedSubsetSampler
from ..utils.config import ConfigBase
import datajoint as dj
from .. import logger as log
from .schema_bridge import experiment, anatomy

import numpy as np
from torch.utils.data import DataLoader

schema = dj.schema('neurodata_movie_configs', locals())


class StimulusTypeMixin:

    def add_transforms(self, key, datasets, tier, exclude=None, normalize=True, cuda=False):
        log.info('Adding transforms'.ljust(80, '-'))
        if exclude is not None:
            log.info('Excluding "' + '", "'.join(exclude) + '" from normalization')
        for k, dataset in datasets.items():
            for ex in exclude:
                assert ex in dataset.data_groups, '{} not in data_groups'.format(ex)
            transforms = []
            if tier == 'train':
                transforms.append(Subsequence(key['train_seq_len']))
            if normalize:
                log.info('Using normalization={}'.format(normalize))
                transforms.append(Normalizer(dataset, stats_source=key['stats_source'], exclude=exclude))
            transforms.append(ToTensor(cuda=cuda))
            dataset.transforms = transforms

        return datasets

    def get_constraint(self, dataset, stimulus_type, tier=None):
        constraint = np.zeros(len(dataset.types), dtype=bool)
        for const in map(lambda s: s.strip(), stimulus_type.split('|')):
            if const.startswith('~'):
                tmp = (dataset.types != const[1:])
            else:
                tmp = (dataset.types == const)
            constraint = constraint | tmp
        if tier is not None:
            constraint = constraint & (dataset.tiers == tier)
        return constraint

    def get_loaders(self, datasets, tier, batch_size, stimulus_types, balanced=False,
                    merge_noise_types=True, shrink_to_same_size=False):
        log.info('Assembling loaders'.ljust(80, '-'))
        if not isinstance(stimulus_types, list):
            log.info('Using ' + stimulus_types + ' as stimulus type for all datasets')
            stimulus_types = len(datasets) * [stimulus_types]
        elif len(stimulus_types) == 1 and len(datasets) > 1:
            log.info('Using ' + stimulus_types[0] + ' as stimulus type for all datasets')
            stimulus_types = len(datasets) * stimulus_types
        else:
            assert len(stimulus_types) == len(datasets), \
                'Number of requested types does not match number of datasets. You need to choose a different group'

        log.info('Stimulus sources: ' + ', '.join(stimulus_types))

        loaders = OrderedDict()
        constraints = [self.get_constraint(dataset, stimulus_type, tier=tier)
                       for dataset, stimulus_type in zip(datasets.values(), stimulus_types)]
        if shrink_to_same_size:
            if tier is None or tier in ('train', 'validation'):
                min_n = np.min([e.sum() for e in constraints])
                new_con = []
                for i, c, st in zip(count(), constraints, stimulus_types):
                    if not '|' in st:
                        c2 = c & (c.cumsum() <= min_n)
                    else:
                        c2 = c
                    untouched = c == c2
                    for c3 in constraints[i + 1:]:
                        c3 &= untouched
                    new_con.append(c2)
                constraints = new_con
                log.info('Shrinking each type in {} sets to same size of {}'.format(tier, min_n))

        for (k, dataset), stimulus_type, constraint in zip(datasets.items(), stimulus_types, constraints):
            log.info('Selecting trials from ' + stimulus_type + ' and tier=' + repr(tier))
            ix = np.where(constraint)[0]
            log.info('Found {} active trials'.format(constraint.sum()))
            if tier == 'train':
                if not balanced:
                    log.info("Configuring random subset sampler for " + k)
                    loaders[k] = DataLoader(dataset, sampler=SubsetRandomSampler(ix), batch_size=batch_size)
                else:
                    log.info("Configuring balanced random subset sampler for" + k)
                    if merge_noise_types:
                        log.info("Balancing Clip vs. Rest")
                        types = np.array([('Clip' if t == 'stimulus.Clip' else 'Noise') for t in dataset.types])
                    loaders[k] = DataLoader(dataset, sampler=BalancedSubsetSampler(ix, types), batch_size=batch_size)
                    log.info('Number of samples in the loader will be {}'.format(len(loaders[k].sampler)))
            else:
                log.info("Configuring sequential subset sampler for " + k)
                loaders[k] = DataLoader(dataset, sampler=SubsetSequentialSampler(ix), batch_size=batch_size)
                log.info('Number of samples in the loader will be {}'.format(len(loaders[k].sampler)))
            log.info('Number of batches in the loader will be {}'.format(
                int(np.ceil(len(loaders[k].sampler) / loaders[k].batch_size))))

        return loaders

    def load_data(self, key, stimulus_types, tier=None, batch_size=1, key_order=None,
                  normalize=True, exclude_from_normalization=None,
                  balanced=False, shrink_to_same_size=False, cuda=False):
        log.info('Loading {} datasets with tier {}'.format(pformat(stimulus_types, indent=20), tier))
        datasets = MovieMultiDataset().fetch_data(key, key_order=key_order)
        log.info('Adding stats source'.ljust(80, '-'))
        for k, dat in datasets.items():
            if 'stats_source' in key:
                log.info('Adding stats_source "{stats_source}" to dataset '.format(**key))
                dat.stats_source = key['stats_source']

        log.info('Using statistics source ' + key['stats_source'])
        datasets = self.add_transforms(key, datasets, tier, exclude=exclude_from_normalization,
                                       normalize=normalize, cuda=cuda)
        loaders = self.get_loaders(datasets, tier, batch_size, stimulus_types=stimulus_types,
                                   balanced=balanced, shrink_to_same_size=shrink_to_same_size)
        return datasets, loaders


class AreaLayerMixin(StimulusTypeMixin):
    def load_data(self, key, tier=None, batch_size=1, key_order=None, cuda=False, **kwargs):
        log.info('Ignoring {} when loading {}'.format(pformat(kwargs, indent=20), self.__class__.__name__))
        shrink = key.pop('shrink', False)
        balanced = key.pop('balanced', False)

        datasets, loaders = super().load_data(key, key.pop('stimulus_type').split(','),
                                              tier, batch_size, key_order,
                                              exclude_from_normalization=key.pop('exclude').split(','),
                                              normalize=key.pop('normalize'),
                                              balanced=balanced,
                                              shrink_to_same_size=shrink,
                                              cuda=cuda)

        log.info('Subsampling to layer "{layer}" and area "{brain_area}"'.format(**key))
        for readout_key, dataset in datasets.items():
            layers = dataset.neurons.layer
            areas = dataset.neurons.area
            idx = np.where((layers == key['layer']) & (areas == key['brain_area']))[0]
            dataset.transforms.insert(-1, Subsample(idx))
        return datasets, loaders



@schema
class DataConfig(ConfigBase, dj.Lookup):
    _config_type = 'data'

    def data_key(self, key):
        return dict(key, **self.parameters(key))

    def load_data(self, key, oracle=False, keep_transforms=False, **kwargs):
        assert not keep_transforms or oracle, 'keep_transforms should only be true when oracle is true'
        data_key = self.data_key(key)
        Data = getattr(self, data_key.pop('data_type'))
        datasets, loaders = Data().load_data(data_key, **kwargs)

        if oracle:
            log.info('Placing oracle data samplers')
            for readout_key, loader in loaders.items():
                ix = loader.sampler.indices
                condition_hashes = datasets[readout_key].condition_hashes
                log.info('Replacing ' + loader.sampler.__class__.__name__ + ' with RepeatsBatchSampler')
                Loader = loader.__class__
                loaders[readout_key] = Loader(loader.dataset,
                                              batch_sampler=RepeatsBatchSampler(condition_hashes, subset_index=ix))

                if not keep_transforms:
                    datasets[readout_key].transforms = \
                        [tr for tr in datasets[readout_key].transforms if isinstance(tr, (Subsample, ToTensor))]
        return datasets, loaders

    class AreaLayer(dj.Part, AreaLayerMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        stimulus_type           : varchar(512)  # type of stimulus
        exclude                 : varchar(512) # what inputs to exclude from normalization
        normalize               : bool         # whether to use a normalize or not
        train_seq_len           : smallint     # training sequence length in frames
        -> experiment.Layer
        -> anatomy.Area
        """
        _exclude_from_normalization = ['inputs', 'responses']

        def describe(self, key):
            return "{brain_area} {layer} on {stimulus_type}. normalize={normalize} on {stats_source} (except '{exclude}')".format(
                **key)

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Clip', '~stimulus.Clip'],
                             ['inputs,responses'],
                             [True],
                             [30 * 5],
                             ['L2/3'],
                             ['V1']):
                yield dict(zip(self.heading.dependent_attributes, p))

    class AreaLayerMultiSource(dj.Part, AreaLayerMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        stimulus_type           : varchar(512)  # type of stimulus
        exclude                 : varchar(512) # what inputs to exclude from normalization
        normalize               : bool         # whether to use a normalize or not
        train_seq_len           : smallint     # training sequence length in frames
        balanced                : bool         # whether to use balanced samplers
        shrink                  : bool         # whether to shrink all datasets to the same number of trial
        -> experiment.Layer
        -> anatomy.Area
        """
        _exclude_from_normalization = ['inputs', 'responses']

        def describe(self, key):
            return "{brain_area} {layer} on {stimulus_type}. normalize={normalize} on {stats_source} (except '{exclude}'). balanced={balanced} and shrink={shrink}".format(
                **key)

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Clip,~stimulus.Clip,stimulus.Clip|~stimulus.Clip'],
                             ['inputs,responses'],
                             [True],
                             [30 * 5],
                             [True],
                             [True, False],
                             ['L2/3'],
                             ['V1']):
                yield dict(zip(self.heading.dependent_attributes, p))
