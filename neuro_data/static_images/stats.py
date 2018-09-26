import datajoint as dj
from tqdm import tqdm
import json
import numpy as np
from scipy import stats

from neuro_data.utils.measures import corr
from .configs import DataConfig

from .data_schemas import StaticMultiDataset, StaticScan

from .. import logger as log

schema = dj.schema('neurodata_static_stats', locals())
data_schemas = dj.create_virtual_module('data_schemas', 'neurodata_static')


@schema
class Oracle(dj.Computed):
    definition = """
    # oracle computation for static images

    -> StaticMultiDataset
    -> DataConfig
    ---
    """

    @property
    def key_source(self):
        return StaticMultiDataset() * DataConfig()

    class Scores(dj.Part):
        definition = """
        -> master
        -> StaticMultiDataset.Member
        ---
        pearson           : float     # mean test correlation
        """

    class UnitScores(dj.Part):
        definition = """
        -> master.Scores
        -> StaticScan.Unit
        ---
        pearson           : float     # mean test correlation
        """

    def make(self, key):
        # --- load data
        testsets, testloaders = DataConfig().load_data(key, tier='test', oracle=True)

        self.insert1(dict(key))
        for readout_key, loader in testloaders.items():
            log.info('Computing oracle for ' + readout_key)
            oracles, data = [], []
            for inputs, *_, outputs in loader:
                inputs = inputs.numpy()
                outputs = outputs.numpy()
                assert np.all(np.abs(np.diff(inputs, axis=0)) == 0), \
                    'Images of oracle trials does not match'
                r, n = outputs.shape  # responses X neurons
                log.info('\t    {} responses for {} neurons'.format(r, n))
                assert r > 4, 'need more than 4 trials for oracle computation'
                mu = outputs.mean(axis=0, keepdims=True)
                oracle = (mu - outputs / r) * r / (r - 1)
                oracles.append(oracle)
                data.append(outputs)
            if len(data) == 0:
                log.error('Found no oracle trials! Skipping ...')
                return
            pearson = corr(np.vstack(data), np.vstack(oracles), axis=0)

            member_key = (StaticMultiDataset.Member() & key &
                          dict(name=readout_key)).fetch1(dj.key)
            member_key = dict(member_key, **key)
            self.Scores().insert1(dict(member_key, pearson=np.mean(pearson)), ignore_extra_fields=True)
            unit_ids = testsets[readout_key].neurons.unit_ids
            assert len(unit_ids) == len(
                pearson) == outputs.shape[-1], 'Neuron numbers do not add up'
            self.UnitScores().insert(
                [dict(member_key, pearson=c, unit_id=u)
                 for u, c in tqdm(zip(unit_ids, pearson), total=len(unit_ids))],
                ignore_extra_fields=True)


def load_dataset(key):
    from .data_schemas import InputResponse, Eye, Treadmill
    from .datasets import StaticImageSet
    for k in InputResponse.heading.primary_key:
        assert k in key
    include_behavior = bool(Eye.proj() * Treadmill().proj() & key)
    data_names = ['images', 'responses'] if not include_behavior \
        else ['images',
              'behavior',
              'pupil_center',
              'responses']
    h5filename = InputResponse().get_filename(key)
    return StaticImageSet(h5filename, *data_names)


@schema
class OracleStims(dj.Computed):
    definition = """
    -> data_schemas.InputResponse
    ---
    stimulus_type           : varchar(64)   # {stimulus.Frame, ~stimulus.Frame, stimulus.Frame|~stimulus.Frame}
    frame_image_ids         : longblob      # Array of frame_iamge_ids that has at least 4 (Arbitary) repeats
    condition_hashes_json   : varchar(8000) # Json (list) of condition_hashes that has at least 4 (Arbitary) repeats
    num_oracle_stims        : int           # num of unique stimuli that have >= 4 repeat presentations
    min_trial_repeats       : int           # The min_num_of_occurances in the condition_hashes array
    """

    @property
    def key_source(self):
        from .data_schemas import StaticMultiDataset, InputResponse
        return InputResponse & StaticMultiDataset.Member

    def make(self, key):
        min_num_of_repeats = 4  # Arbitary requirment

        dataset = load_dataset(key)

        # Get all frame_image_ids for repeated stimulus.image that repeates more than 4
        all_frame_ids, all_frame_ids_counts = np.unique(
            dataset.info.frame_image_id[dataset.types == 'stimulus.Frame'], return_counts=True)
        frame_id_counts_mask = all_frame_ids_counts >= min_num_of_repeats
        frame_image_ids = all_frame_ids[frame_id_counts_mask]

        # Get all condition_hash for repeated ~stimulus.image that repeates more than 4
        all_cond_hashes, all_cond_hash_counts = np.unique(
            dataset.condition_hashes[dataset.types != 'stimulus.Frame'], return_counts=True)
        cond_hash_counts_mask = all_cond_hash_counts >= min_num_of_repeats
        condition_hashes = all_cond_hashes[cond_hash_counts_mask]

        # Compute min_trial_repeats for both natural images and noise, also determine stimulus.type
        min_trial_repeats = []
        stim_types = []
        if len(frame_image_ids) > 0:
            min_trial_repeats.append(
                all_frame_ids_counts[frame_id_counts_mask].min())
            stim_types.append('stimulus.Frame')
        if len(condition_hashes) > 0:
            min_trial_repeats.append(
                all_cond_hash_counts[cond_hash_counts_mask].min())
            stim_types.append('~stimulus.Frame')

        if len(min_trial_repeats) == 0:
            raise Exception('Dataset does not contain trial repeats')

        chashes_json = json.dumps(condition_hashes.tolist())
        assert len(
            chashes_json) < 8000, 'condition hashes exceeds 8000 characters'
        key['stimulus_type'] = '|'.join(stim_types)
        key['frame_image_ids'] = frame_image_ids
        key['condition_hashes_json'] = chashes_json
        key['num_oracle_stims'] = frame_image_ids.size + condition_hashes.size
        key['min_trial_repeats'] = np.min(min_trial_repeats)
        self.insert1(key)


@schema
class BootstrapOracleSeed(dj.Lookup):
    definition = """
    oracle_bootstrap_seed                 :  int # random seed
    ---
    """

    @property
    def contents(self):
        for seed in list(range(100)):
            yield (seed,)


@schema
class BootstrapOracle(dj.Computed):
    definition = """
    -> OracleStims
    -> BootstrapOracleSeed
    ---
    """

    class Score(dj.Part):
        definition = """
        -> master
        ---
        boostrap_score_true			    : float
        boostrap_score_null			    : float
        """

    class UnitScore(dj.Part):
        definition = """
        -> master
        -> StaticScan.Unit
        ---
        boostrap_unit_score_true		: float
        boostrap_unit_score_null		: float
        """

    def sample_from_id_or_hash(self, id_or_hash, dataset, sample_size):
        return np.random.choice(np.where(dataset == id_or_hash)[0], sample_size, replace=False)

    def compute_oracle(self, outputs):
        r = outputs.shape[0]
        mu = outputs.mean(axis=0, keepdims=True)
        oracles = (mu - outputs / r) * r / (r - 1)
        return oracles

    def check_input(self, inputs):
        assert np.all(np.abs(np.diff(inputs, axis=0)) ==
                      0), 'Images of oracle trials do not match'

    def sample_and_compute_oracle(self, dataset, frame_image_ids, condition_hashes, sample_size):
        num_natim = len(frame_image_ids)
        num_noise = len(condition_hashes)
        total_imgs = num_natim + num_noise

        # Consturct boolean matrix for null sampling
        is_natim_list = [True] * num_natim
        is_natim_list += [False] * num_noise

        # Save into memeory to save time from pulling it every time
        responses = dataset.responses
        dataset_frame_image_id = dataset.info.frame_image_id
        dataset_condition_hashes = dataset.condition_hashes
        dataset_images = dataset.images

        # True oracle computation
        # Matrices to store results
        true_responses = np.empty(
            shape=[total_imgs, sample_size, responses.shape[1]])
        true_oracles = np.empty(
            shape=[total_imgs, sample_size, responses.shape[1]])

        for i in range(0, num_natim):
            true_target_index = self.sample_from_id_or_hash(
                frame_image_ids[i], dataset_frame_image_id, sample_size)
            self.check_input(dataset_images[true_target_index])

            response_matrix = responses[true_target_index]
            true_responses[i] = response_matrix
            true_oracles[i] = self.compute_oracle(response_matrix)

        for i in range(0, num_noise):
            true_target_index = self.sample_from_id_or_hash(
                condition_hashes[i], dataset_condition_hashes, sample_size)
            self.check_input(dataset_images[true_target_index])

            response_matrix = responses[true_target_index]
            true_responses[i + num_natim] = response_matrix
            true_oracles[i + num_natim] = self.compute_oracle(response_matrix)

        # corr(response_matrices, oracle_matrices, axis=0)
        # Null oracle computation
        null_responses = np.empty(
            shape=[total_imgs, sample_size, responses.shape[1]])
        null_oracles = np.empty(
            shape=[total_imgs, sample_size, responses.shape[1]])

        for i in range(0, total_imgs):
            target_indices = np.random.choice(
                len(is_natim_list), sample_size, replace=False)
            null_target_indices = np.empty(
                shape=[len(target_indices)], dtype='int_')

            for j, null_index in enumerate(target_indices):
                # Determine if it is in natrual images or not
                if null_index < num_natim:
                    id_or_hash = frame_image_ids[null_index]
                else:
                    id_or_hash = condition_hashes[null_index - num_natim]

                # Sample from respective datasets
                if is_natim_list[null_index]:
                    null_target_indices[j] = self.sample_from_id_or_hash(
                        id_or_hash, dataset_frame_image_id, 1)
                else:
                    null_target_indices[j] = self.sample_from_id_or_hash(
                        id_or_hash, dataset_condition_hashes, 1)

            response_matrix = responses[null_target_indices]
            null_responses[i] = response_matrix
            null_oracles[i] = self.compute_oracle(response_matrix)

        true_responses = true_responses.reshape([-1, responses.shape[1]])
        true_oracles = true_oracles.reshape([-1, responses.shape[1]])
        null_responses = null_responses.reshape([-1, responses.shape[1]])
        null_oracles = null_oracles.reshape([-1, responses.shape[1]])

        return corr(true_responses, true_oracles, axis=0), corr(null_responses, null_oracles, axis=0)

    def make(self, key):
        dataset = load_dataset(key)

        stim_tup = OracleStims & key
        frame_image_ids = stim_tup.fetch1('frame_image_ids')
        condition_hashes = json.loads(stim_tup.fetch1('condition_hashes_json'))
        sample_size = min(
            *stim_tup.fetch1('num_oracle_stims', 'min_trial_repeats'))

        np.random.seed(key['oracle_bootstrap_seed'])

        true_pearson, null_pearson = self.sample_and_compute_oracle(
            dataset, frame_image_ids, condition_hashes, sample_size)

        self.insert1(key)
        # Inserting pearson mean scores to Score table
        self.Score().insert1(dict(key, boostrap_score_true=true_pearson.mean(),
                                  boostrap_score_null=null_pearson.mean()))
        # Inserting unit pearson scores
        self.UnitScore().insert([dict(key, unit_id=u, boostrap_unit_score_true=t, boostrap_unit_score_null=n)
                                 for u, t, n in zip(dataset.neurons.unit_ids, true_pearson, null_pearson)])
