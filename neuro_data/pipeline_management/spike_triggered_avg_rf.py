import pandas as pd
import datajoint as dj
import numpy as np
import matplotlib.pyplot as plt
import datajoint as dj

from neuro_data.static_images import data_schemas

stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')

# raw data storage
dj.config['extnernal'] = dict(protocol='file',
                              location='/external/')

# dj.config['external-data'] = {'protocol': 'file', 'location': '/external/'}

schema = dj.schema('neurodata_static')

def check_none(np_array):
    """
    check if array's elements are all Nonetype or not
    Args:
        np_array (np.array): numpy array
    
    Return:
        bool: True if every element in an array is None. False otherwise.
    """
    return all(v is None for v in np_array)

@schema
class StaticSpikeTriggeredAverageRF(dj.Computed):
    definition = """
    # spike trigerred average using static image (e.g. imagenet)
    -> data_schemas.InputResponse
    """
    class Unit(dj.Part):
        definition = """ # frame for a single unit
        -> master
        -> data_schemas.InputResponse.ResponseKeys
        ---
        sta_rf      :   external               # STA RF 
        """

    def make(self, key):
        # Get data
        print('Loading data')
        response_block = (data_schemas.InputResponse.ResponseBlock & key).fetch1('responses')
        unit_keys = (data_schemas.InputResponse.ResponseKeys & key).fetch('KEY', order_by='col_id')
        frames = (data_schemas.Frame * data_schemas.InputResponse.Input & key).fetch('frame', order_by='row_id')
        frames = np.stack(frames).astype(np.float32)

        # Iterate over each unit getting the frame
        print('Iterating over units')
        self.insert1(key)
        for responses, unit_key in zip(response_block.T, unit_keys):
            sta_rf = np.average(frames, weights=responses/responses.sum(), axis=0)
            self.Unit.insert1({**unit_key, 'sta_rf': sta_rf})

    #TODO Finish this method
    @staticmethod
    def plot_sta_rf(key):
        """
        Plot Spike Triggered Average Receptive Field.
        If multiple channels exist (i.e. color stimulus), then it looks for channel
        info and plots based on which color was presented.
        For example, if color info was [2, 3, None] in stimulus table,
        then it fills 0 for red, and blue/green are mapped as they are (r=1, g=2, b=3).
        If color info was [1, None, 2], then it fills 0 for blue and red/green are
        mapped as they are.

        If stimulus was in grayscale, then it plots grayscale STA RF.

        Args:
            key (dict): scan with the following keys
                animal_id
                session
                scan_idx
                unit_id
            num_neurons: number of neurons to plot. Default at 50
        """
        
        stim_type = np.unique((stimulus.Condition & (stimulus.Trial & key)).fetch('stimulus_type'))

        if len(stim_type) >1:
            msg = """There are more than 1 stimulus type ({}) for this scan!
                    You need to either check your scan or customize the plotting
                    for multple stimulus types!""".format(stim_type)
            raise ValueError(msg)
        
        stim_type = stim_type[0].split('.')[1]

        if stim_type == 'ColorFrameProjector':
            stim_table = (stimulus.Trial & key) * (dj.U('condition_hash',
                                                        'projector_config_id'
                                                        'channel_1',
                                                        'channel_2',
                                                        'channel_3') & getattr(stimulus,stim_type))
            
            # config_id = stim_table.fetch('projector_config_id')

            # # make sure that the config id doesnt change in the middle of recording
            # if any(config_id):
            #     config_id = np.unique(config_id)[0]
            #     channels = (experiment.ProjectorConfig & 'projector_config_id = {}'.format(config_id)).fetch1('channel_1','channel_2','channel_3')

            #     # check which channels were used (wrt projector)
            #     valid_channel_inds = np.where(np.array(channels) != 'none')[0]



            # else:
            #     msg = """Projector configuration must stay the same during recording!
            #             But config id was set to {}""".format(np.unique(config_id))
            #     raise ValueError(msg)

            channels = stim_table.fetch('channel_1','channel_2','channel_3')
            
            # check every channel has the same value. In theory, it should since
            # the projector configuration doesnt change during trial.
            # Also if they are all the same, record which channel to use (wrt image, not projector)
            valid_colors = []
            for ind, channel in enumerate(channels):
                if not any(channel):
                    if not check_none(channel):
                        raise ValueError('channel_{} value changed during scan!'.format(ind+1))
                else:
                    valid_colors.append(np.unique(channel)[0]-1)

            sta_rf = (StaticSpikeTriggeredAverageRF.Unit & key).fetch1('sta_rf')

        pass

# Adopted from Erick Cobost's static_pilot.analyses.py

def get_traces(key):
    """ Get spike traces for all cells in these scan (along with their times in stimulus
    clock).

    Arguments:
        key (dict): Key for a scan (or field).

    Returns:
        traces (np.array): A (num_units x num_scan_frames) array with all spike traces.
            Traces are restricted to those classified as soma and ordered by unit_id.
        unit_ids (list): A (num_units) list of unit_ids in traces.
        trace_times (np.array): A (num_units x num_scan_frames) array with the time (in
            seconds) for each unit's trace in stimulus clock (same clock as times in
            stimulus.Trial).

    Note: On notation
        What is called a frametime in stimulus.Sync and stimulus.Trial is actually the
        time each depth of scanning started. So for a scan with 1000 frames and four
        depths per frame/volume, there will be 4000 "frametimes".

    Note 2:
        For a scan with 10 depths, a frame i is considered complete if all 10 depths were
        recorded and saved in the tiff file, frame_times however save the starting time of
        each depth independently (for instance if 15 depths were recorded there will be
        one scan frame but 15 frame times, the last 5 have to be ignored).
    """
    # Pick right pipeline for this scan (reso or meso)
    pipe_name = (fuse.ScanDone & key).fetch1('pipe')
    pipe = reso if pipe_name == 'reso' else meso

    # Get traces
    units = pipe.ScanSet.Unit() & key & (pipe.MaskClassification.Type & {'type': 'soma'})
    spikes = pipe.Activity.Trace() * pipe.ScanSet.UnitInfo() & units.proj()
    unit_ids, traces, ms_delays = spikes.fetch('unit_id', 'trace', 'ms_delay',
                                               order_by='unit_id')

    # Get time of each scan frame for this scan (in stimulus clock; same as in Trial)
    depth_times = (stimulus.Sync & key).fetch1('frame_times')
    num_frames = (pipe.ScanInfo & key).fetch1('nframes')
    num_depths = len(dj.U('z') & (pipe.ScanInfo.Field.proj('z', nomatch='field') & key))
    if len(depth_times) / num_depths < num_frames or (len(depth_times) / num_depths >
                                                      num_frames + 1):
        raise ValueError('Mismatch between frame times and tiff frames')
    frame_times = depth_times[:num_depths * num_frames:num_depths]  # one per frame

    # Add per-cell delay to each frame_time
    trace_times = np.add.outer(ms_delays / 1000, frame_times)  # num_traces x num_frames

    return np.stack(traces), np.stack(unit_ids), trace_times


def trapezoid_integration(x, y, x0, xf):
    """ Integrate y (recorded at points x) from x0 to xf.

    Arguments:
        x (np.array): Timepoints (num_timepoints) when y was recorded.
        y (np.array): Signal (num_timepoints).
        x0 (float or np.array): Starting point(s). Could be a 1-d array (num_samples).
        xf (float or np.array): Final point. Same shape as x0.

    Returns:
        Integrated signal from x0 to xf:
            a 0-d array (i.e., float) if x0 and xf are floats
            a 1-d array (num_samples) if x0 and xf are 1-d arrays
    """
    # Basic checks
    if np.any(xf <= x0):
        raise ValueError('xf has to be higher than x0')
    if np.any(x0 < x[0]) or np.any(xf > x[-1]):
        raise ValueError('Cannot integrate outside the original range x of the signal.')

    # Compute area under each trapezoid
    trapzs = np.diff(x) * (
                y[:-1] + y[1:]) / 2  # index i is trapezoid from point i to point i + 1

    # Find timepoints right before x0 and xf
    idx_before_x0 = np.searchsorted(x, x0) - 1
    idx_before_xf = np.searchsorted(x, xf) - 1

    # Compute y at the x0 and xf points
    slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])  # index i is slope from p_i to p_{i+1}
    y0 = y[idx_before_x0] + slopes[idx_before_x0] * (x0 - x[idx_before_x0])
    yf = y[idx_before_xf] + slopes[idx_before_xf] * (xf - x[idx_before_xf])

    # Sum area of all interior trapezoids
    indices = np.stack([idx_before_x0 + 1, idx_before_xf],
                       axis=-1).ravel()  # interleaved x0 and xf for all samples
    integral = np.add.reduceat(trapzs, indices, axis=-1)[::2].squeeze()

    # Add area of edge trapezoids (ones that go from x0 to first_x_sample and from last_x_sample to xf)
    integral += (x[idx_before_x0 + 1] - x0) * (y0 + y[idx_before_x0 + 1]) / 2
    integral += (xf - x[idx_before_xf]) * (y[idx_before_xf] + yf) / 2

    # Deal with edge case where both x0 and xf are in the same trapezoid
    same_trapezoid = idx_before_x0 == idx_before_xf
    integral[same_trapezoid] = ((xf - x0) * (y0 + yf) / 2)[same_trapezoid]

    return integral


# TODO: WAIT UNTIL ERICK MAKE A PR ON NEURO_DATA PACKAGE..
# @schema
# class NewResponses(dj.Computed):
#     definition = """ # responses from each cell to all (relevant) trials

#     -> Scan
#     ---
#     image_resps:        longblob        # responses to all images (num_trials x num_cells)
#     blank_resps:        longblob        # responses to blanks before each image (num_trials x num_cells)
#     """

#     class Trial(dj.Part):
#         definition = """ # a single trial in the response block
#         -> master
#         row_id      :int
#         ---
#         -> Scan.Frame
#         -> stimulus.Trial
#         """

#     class Unit(dj.Part):
#         definition = """ # a single unit in the response block
#         -> master
#         col_id              : int
#         ---
#         -> Scan.Unit
#         """

#     def make(self, key):
#         # Get all traces for this scan
#         print('Getting traces...')
#         traces, unit_ids, trace_times = get_traces(key)

#         # Get trial times for frames in Scan.Frame (excluding bad trials)
#         print('Getting onset and offset times for each image (and blank)...')
#         trials_rel = stimulus.Trial * Scan.Frame - data_schemas.ExcludedTrial & key
#         flip_times, trial_ids, cond_hashes = trials_rel.fetch('flip_times', 'trial_idx',
#                                                               'condition_hash',
#                                                               order_by='trial_idx',
#                                                               squeeze=True)
#         if any([len(ft) < 2 or len(ft) > 3 for ft in flip_times]):
#             raise ValueError('Only works for frames with 2 or 3 flips')

#         # Find start and duration of blank and image frames
#         monitor_fps = 60
#         blank_onset = np.stack(
#             [ft[0] for ft in flip_times]) - 1 / monitor_fps  # start of blank period
#         image_onset = np.stack(
#             [ft[1] for ft in flip_times]) + 1 / monitor_fps  # start of image
#         blank_duration = image_onset + 1 / monitor_fps - blank_onset
#         image_duration = 0.5  # np.stack([ft[2] for ft in flip_times]) - image_onset
#         """
#         Each trial is a stimulus.Frame.
#         A single stimulus.Frame is composed of a flip (1/60 secs), a blanking period (0.3 
#         - 0.5 secs), another flip, the image (0.5 secs) and another flip. During flips 
#         screen is gray (as during blanking) so I count the flips before and after the 
#         blanking as part of the blanking. There is also another flip after the image and 
#         some time between trials (t / 60, t > 0, usually 1) that could be counted as part 
#         of the blanking; I ignore those.
#         """

#         # Add a shift to the onset times to account for the time it takes for the image to
#         # travel from the retina to V1
#         image_onset += 0.03
#         blank_onset += 0.03
#         # Wiskott, L. How does our visual system achieve shift and size invariance?. Problems in Systems Neuroscience, 2003.

#         # Sample responses (trace by trace) with a rectangular window
#         print('Sampling responses...')
#         image_resps = np.stack([trapezoid_integration(tt, t, image_onset, image_onset +
#                                                       image_duration) / image_duration for
#                                 tt, t in zip(trace_times, traces)], axis=-1)
#         blank_resps = np.stack([trapezoid_integration(tt, t, blank_onset, blank_onset +
#                                                       blank_duration) / blank_duration for
#                                 tt, t in zip(trace_times, traces)], axis=-1)

#         # Insert
#         print('Inserting...')
#         self.insert1({**key, 'image_resps': image_resps.astype(np.float32),
#                       'blank_resps': blank_resps.astype(np.float32)})
#         self.Unit.insert([{**key, 'unit_id': unit_id, 'col_id': i} for i, unit_id in
#                           enumerate(unit_ids)])
#         self.Trial.insert([{**key, 'trial_idx': trial_idx, 'condition_hash': cond_hash,
#                             'row_id': i} for i, (trial_idx, cond_hash) in enumerate(zip(
#             trial_ids, cond_hashes))])

