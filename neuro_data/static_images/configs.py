from collections import OrderedDict
from itertools import product, count
from attorch.dataloaders import RepeatsBatchSampler
from torch.utils.data.sampler import SubsetRandomSampler

from .data_schemas import StaticMultiDataset
from .transforms import Subsample, Normalizer, ToTensor
from ..utils.sampler import SubsetSequentialSampler
from ..utils.config import ConfigBase
import datajoint as dj
from .. import logger as log

import numpy as np
from torch.utils.data import DataLoader

experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')

schema = dj.schema('neurodata_static_configs', locals())


class StimulusTypeMixin:
    _stimulus_type = None

    #TODO: add normalize option
    def add_transforms(self, key, datasets, exclude=None):
        if exclude is not None:
            log.info('Excluding "{}" from normalization'.format('", "'.join(exclude)))
        for k, dataset in datasets.items():
            transforms = []
            transforms.extend([Normalizer(dataset, stats_source=key['stats_source'], exclude=exclude), ToTensor()])
            dataset.transforms = transforms

        return datasets

    def get_constraint(self, dataset, stimulus_type, tier=None):
        constraint = np.zeros(len(dataset.types), dtype=bool)
        for const in map(lambda s: s.strip(), stimulus_type.split('|')):
            if const.startswith('~'):
                log.info('Using all trial but from {}'.format(const[1:]))
                tmp = (dataset.types != const[1:])
            else:
                log.info('Using all trial from {}'.format(const))
                tmp = (dataset.types == const)
            constraint = constraint | tmp
        if tier is not None:
            constraint = constraint & (dataset.tiers == tier)
        return constraint

    def get_loaders(self, datasets, tier, batch_size, stimulus_types):
        if not isinstance(stimulus_types, list):
            log.info('Using {} as stimulus type for all datasets'.format(stimulus_types))
            stimulus_types = len(datasets) * [stimulus_types]

        log.info('Stimulus sources: "{}"'.format('","'.join(stimulus_types)))

        loaders = OrderedDict()
        constraints = [self.get_constraint(dataset, stimulus_type, tier=tier)
                       for dataset, stimulus_type in zip(datasets.values(), stimulus_types)]

        for (k, dataset), stimulus_type, constraint in zip(datasets.items(), stimulus_types, constraints):
            log.info('Selecting trials from {} and tier={}'.format(stimulus_type, tier))
            ix = np.where(constraint)[0]
            log.info('Found {} active trials'.format(constraint.sum()))
            if tier == 'train':
                log.info("Configuring random subset sampler for " + k)
                loaders[k] = DataLoader(dataset, sampler=SubsetRandomSampler(ix), batch_size=batch_size)
            else:
                log.info("Configuring sequential subset sampler for " + k)
                loaders[k] = DataLoader(dataset, sampler=SubsetSequentialSampler(ix), batch_size=batch_size)
                log.info('Number of samples in the loader will be {}'.format(len(loaders[k].sampler)))
                log.info('Number of batches in the loader will be {}'.format(
                    int(np.ceil(len(loaders[k].sampler) / loaders[k].batch_size))))
        return loaders

    def load_data(self, key, tier=None, batch_size=1, key_order=None,
                  exclude_from_normalization=None, stimulus_types=None):
        log.info('Loading {} dataset with tier={}'.format(self._stimulus_type, tier))
        datasets = StaticMultiDataset().fetch_data(key, key_order=key_order)
        for k, dat in datasets.items():
            if 'stats_source' in key:
                log.info('Adding stats_source "{stats_source}" to dataset'.format(**key))
                dat.stats_source = key['stats_source']

        log.info('Using statistics source ' +  key['stats_source'])
        datasets = self.add_transforms(key, datasets, exclude=exclude_from_normalization)
        loaders = self.get_loaders(datasets, tier, batch_size, stimulus_types=stimulus_types)
        return datasets, loaders


class AreaLayerRawMixin(StimulusTypeMixin):
    def load_data(self, key, tier=None, batch_size=1, key_order=None, stimulus_types=None, **kwargs):
        log.info('Ignoring input arguments: "' + '", "'.join(kwargs.keys()) + '"' + 'when creating datasets')
        exclude = key.pop('exclude').split(',')
        stimulus_types = key.pop('stimulus_type')
        datasets, loaders = super().load_data(key, tier, batch_size, key_order,
                                              exclude_from_normalization=exclude,
                                              stimulus_types=stimulus_types)

        log.info('Subsampling to layer "{layer}" and area "{brain_area}"'.format(**key))
        for readout_key, dataset in datasets.items():
            layers = dataset.neurons.layer
            areas = dataset.neurons.area
            idx = np.where((layers == key['layer']) & (areas == key['brain_area']))[0]
            if len(idx) == 0:
                log.warning('Empty set of neurons. Deleting this key')
                del datasets[readout_key]
                del loaders[readout_key]
            else:
                dataset.transforms.insert(-1, Subsample(idx))
        return datasets, loaders


@schema
class DataConfig(ConfigBase, dj.Lookup):
    _config_type = 'data'

    def data_key(self, key):
        return dict(key, **self.parameters(key))

    def load_data(self, key, cuda=False, oracle=False, **kwargs):
        data_key = self.data_key(key)
        Data = getattr(self, data_key.pop('data_type'))
        datasets, loaders = Data().load_data(data_key, **kwargs)

        if oracle:
            log.info('Placing oracle data samplers')
            for k, loader in loaders.items():
                ix = loader.sampler.indices
                condition_hashes = datasets[k].condition_hashes
                log.info('Replacing {} with RepeatsBatchSampler'.format(loader.sampler.__class__.__name__))
                loader.sampler = None

                removed = []
                keep = []
                for tr in datasets[k].transforms:
                    if isinstance(tr, (Subsample, ToTensor)):
                        keep.append(tr)
                    else:
                        removed.append(tr.__class__.__name__)
                datasets[k].transforms = keep
                log.warning('Removed the following transforms: "{}"'.format('", "'.join(removed)))
                loader.batch_sampler = RepeatsBatchSampler(condition_hashes, subset_index=ix)

        log.info('Setting cuda={}'.format(cuda))
        for dat in datasets.values():
            for tr in dat.transforms:
                if isinstance(tr, ToTensor):
                    tr.cuda = cuda

        return datasets, loaders

    class AreaLayer(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)   # normalization source
        stimulus_type           : varchar(50)   # type of stimulus
        exclude                 : varchar(512)  # what inputs to exclude from normalization
        normalize               : bool          # whether to use a normalize or not
        -> experiment.Layer
        -> anatomy.Area
        """

        def describe(self, key):
            return "{brain_area} {layer} on {stimulus_type}. normalize={normalize} on {stats_source} (except '{exclude}')".format(**key)

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Frame', '~stimulus.Frame'],
                             ['images,responses',''],
                             [True],
                             ['L4', 'L2/3'],
                             ['V1', 'LM']):
                yield dict(zip(self.heading.dependent_attributes, p))


    class AreaLayerPercentOracle(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)   # normalization source
        stimulus_type           : varchar(50)   # type of stimulus
        exclude                 : varchar(512)  # what inputs to exclude from normalization
        normalize               : bool          # whether to use a normalize or not
        (oracle_source) -> master 
        -> experiment.Layer
        -> anatomy.Area
        percent_low                 : tinyint       # percent oracle lower cutoff 
        percent_high                 : tinyint      # percent oracle upper cutoff 
        """

        def describe(self, key):
            return "Like AreaLayer but only {percent_low}-{percent_high} percent best oracle neurons computed on {oracle_source}".format(**key)

        @property
        def content(self):
            for p in product(['all'],
                             ['stimulus.Frame', '~stimulus.Frame'],
                             ['images,responses'],
                             [True],
                             list((DataConfig.AreaLayer() & dict(brain_area='V1', layer='L2/3',
                                                            normalize=True, stats_source='all',
                                                            stimulus_type='~stimulus.Frame',
                                                            exclude='images,responses')).fetch('data_hash')),
                             ['L2/3'],
                             ['V1'],
                             [25],
                             [75]):
                yield dict(zip(self.heading.dependent_attributes, p))
            for p in product(['all'],
                             ['stimulus.Frame', '~stimulus.Frame'],
                             ['images,responses'],
                             [True],
                             list((DataConfig.AreaLayer() & dict(brain_area='V1', layer='L2/3',
                                                            normalize=True, stats_source='all',
                                                            stimulus_type='~stimulus.Frame',
                                                            exclude='images,responses')).fetch('data_hash')),
                             ['L2/3'],
                             ['V1'],
                             [75],
                             [100]):
                yield dict(zip(self.heading.dependent_attributes, p))

        def load_data(self, key, tier=None, batch_size=1, key_order=None, stimulus_types=None):
            from .oracle import Pearson
            datasets, loaders = super().load_data(key, tier=tier, batch_size=batch_size,
                                                  key_order=key_order, stimulus_types=stimulus_types)
            for rok, dataset in datasets.items():
                member_key = (StaticMultiDataset.Member() & key & dict(name=rok)).fetch1(dj.key)

                okey = dict(key, **member_key)
                okey['data_hash'] = okey.pop('oracle_source')
                units, pearson = (Pearson.UnitScores() & okey).fetch('unit_id', 'pearson')
                assert len(pearson) > 0, 'You forgot to populate oracle for data_hash="{}"'.format(key['oracle_source'])
                assert len(units) == len(dataset.neurons.unit_ids), 'Number of neurons has changed'
                assert np.all(units == dataset.neurons.unit_ids), 'order of neurons has changed'

                low, high = np.percentile(pearson, [key['percent_low'], key['percent_high']])
                selection = (pearson >= low) & (pearson <= high)
                log.info('Subsampling to {} neurons above {:.2f} and below {} oracle'.format(selection.sum(), low, high))
                dataset.transforms.insert(-1, Subsample(np.where(selection)[0]))

                assert np.all(dataset.neurons.unit_ids == units[selection]), 'Units are inconsistent'
            return datasets, loaders