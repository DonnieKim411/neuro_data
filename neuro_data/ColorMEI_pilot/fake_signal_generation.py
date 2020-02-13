import datajoint as dj
dj.config['external'] = dict(protocol='file',
                              location='/external/')

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
import pandas as pd 
import os, shutil
import h5py, json, cv2
from copy import deepcopy

from neuro_data.static_images.data_schemas import * # bad practice
from staticnet_experiments.models import Model
from neuro_data.static_images.data_schemas import StaticMultiDataset

stimulus = dj.create_virtual_module('stimulus','pipeline_stimulus')
schema = dj.schema('neurodata_static')

from .dj_parent import DJTableBase

def pixel_to_luminance():

    # CHANGE BASED ON MAX BIT ON HARDWARE
    color_level = 2**8; 
    pixelValues = np.arange(0,255)
    normalizedPixelValues = pixelValues/255
    gammaValue = 1.9
    lookuptableValues = normalizedPixelValues**gammaValue
    lookuptable = np.repeat(lookuptableValues, 3,1)

    return lookuptable


def add_noise(img, mean, var):
    """
    Add gaussian white noise to the image
    
    input:
        img (numpy array): N dimensional image in shape of ch, row, col
        mean (float): mean value for the guassian noise to be centered at
        var (float): variance value for the guassian noise. 
    
    return:
        noisy_img (numpy array): N dimensional image in shape of ch, row, col after noise added
    
    """
    ch, row, col = img.shape
    sigma = var**0.5
    gauss_noise = np.random.normal(mean, sigma, (1,row,col))
    gauss_noise = np.repeat(gauss_noise, ch, axis=0)

    return img + gauss


def find_valid_trials_from_behavior(key):
    """
    Using the same logic from data_schemas.InputResponse.compute_data line 614,
    Find valid trial indices from behavior data
    
    Input:
        key(dict): animal_id, session, scan_idx

    Return:
        invalid_trial_indices(ndarray): valid trial indices 
    """

    pupil, dpupil, pupil_center, valid_eye = (Eye & key).fetch1('pupil', 'dpupil', 'center', 'valid')
    pupil_center = pupil_center.T
    treadmill, valid_treadmill = (Treadmill & key).fetch1('treadmill', 'valid')
    valid = valid_eye & valid_treadmill

    return valid


@schema
class GaussianNoise(dj.Lookup, DJTableBase):
    definition = """
    # Parameters for Gaussian noise
    noise_hash      : char(32)              # MD5 in base64
    ---
    mean            : float                 # mean value for the gaussian noise to be centered at
    var             : float                 # variance value for the gaussian noise
    """

    @classmethod
    def fill(cls, mean, var):
        noise_hash = cls.generate_md5_hash(dict(mean=mean,var=var))
        cls.insert1([noise_hash, mean, var])


@schema
class NoisyGaussianKernels(dj.Manual, DJTableBase):
    definition = """
    # Gaussian kernels with noise added. Katrin generated the kernels
    kernel_hash   : char(32)              # MD5 in base64
    ---
    noise_type          : varchar(128)          # noise type name
    noise_params        : varchar(256)          # dictionary containing noise params
    original_file_name  : varchar(128)          # katrin's file name
    kernels             : external              # noisy gaussian rfs
    """
    
    @staticmethod
    def fill(path_to_gaussian_rf_file, noise_params):
        """
        Fill in the table 
        Input:
            path_to_gaussian_rf_file (str): absolute path for gaussian_rf_file
            noise_params (dict): parameters for noise to be added.
                                 noise_type(str): name of the noise table (e.g. GaussianNoise)
                                 noise_params: parameters used for the noise_type table. ex) if GaussianNoise table
                                               was used, then mean and var must be provided.
        Return:
            None
        """                      

        # katrin's gaussian RF generation
        filename = os.path.basename(path_to_gaussian_rf_file)

        with h5py.File(path_to_gaussian_rf_file, 'r') as f:
            # List all groups
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]

            # Get the data
            data = list(f[a_group_key])

        gaussian_rfs = np.array(data).transpose(3,2,0,1)

        # obtain noise table and its params
        noise_table = eval(noise_params['noise_type'])

        # Until finding a better way to restrict with floats, Lookup the noise table first,
        # then restrict by the hash.
        mean_val, var_val = (noise_table & "noise_hash ='{}'".format(noise_params['noise_hash'])).fetch1('mean','var')

        kernels = np.zeros(shape=gaussian_rfs.shape)
        new_noise_params = dict(mean=mean_val, var=var_val)

        kernel_hash = DJTableBase.generate_md5_hash(dict(new_noise_params,
                                                         original_file_name=filename,
                                                         noise_type=noise_params['noise_type']))
        
        # add noise to the kernel and insert
        for ind,rf in enumerate(gaussian_rfs):
            kernels[ind,:,:,:]=add_noise(img=rf, mean = mean_val, var=var_val)
        
        NoisyGaussianKernels.insert1(dict(kernel_hash=kernel_hash,
                                          noise_type=noise_params['noise_type'],
                                          noise_params=json.dumps(new_noise_params),
                                          original_file_name=filename,
                                          kernels=kernels.transpose(0,2,3,1)))


@schema
class GaborKernels(dj.Manual, DJTableBase):
    definition = """
    # Gabor kernels that Manolis generated
    kernel_hash   : char(32)                    # MD5 in base64
    ---
    original_file_name  : varchar(128)          # katrin's file name
    kernels             : external              # garbor kernels
    """

    @classmethod
    def fill(cls, path_to_gabor_file):
        """
        Fill in the table 
        Input:
            path_to_gabor_file (str): absolute path for gaussian_rf_file
            
        Return:
            None
        """                      

        # Manolis's garbor generation
        filename = os.path.basename(path_to_gabor_file)
        extension = os.path.splitext(filename)[-1]
        
        if extension == '.mat': # matfile
            kernels = loadmat(path_to_gabor_file)['A'].reshape(16,16,-1, order='C').transpose(2,0,1)
        elif extension == '.h5':
            kernels = []
            with h5py.File(path_to_gabor_file) as fh:
                for channel in fh.keys():
                    kernels.append(fh[channel][:])

            kernels = np.stack(kernels)

        kernel_hash = DJTableBase.generate_md5_hash(dict(original_file_name=filename))
               
        cls.insert1(dict(kernel_hash=kernel_hash, original_file_name=filename, kernels=kernels))

        return None



@schema
class FakeResponseFromImages(dj.Manual, DJTableBase):
    definition = """
    # fake signals generated by convolving input images with kernels
    ->experiment.Scan
    response_hash       : char(32)              # MD5 in base64
    ---
    input_image_class   : varchar(128)          # stimulus image class
    kernel_type         : varchar(128)          # kernel type
    kernel_hash         : char(32)              # MD5 in base64
    response            : external              # averaged fake response
    """

    @staticmethod
    def fill(key, kernel_params):
        """
        fill in FakeResponseFromImages table
        Input:
            key(dict): animal_id, session, scan_idx
            kernel_params(dict): 
                kernel_type: table name for the kernel.
                kernel_hash: kernel_hash to uniquely identify which kernel to use from kernel_type table
        Return:
            None
        """
        
        # find the key's stimulus type and its table, then fetch image data
        stim_type = np.unique((stimulus.Condition & (stimulus.Trial & key)).fetch('stimulus_type'))

        if len(stim_type) > 1:
            raise ValueError("There must be only 1 stimulus type! Currently there are stimuli of : {}".format(stim_type))
        else:
            stim_type = stim_type[0].split('.')[1]
        stim_table = getattr(stimulus, stim_type)

        id_table = ((stimulus.Trial & key) * stim_table).proj('image_id','channel_1','channel_2','channel_3')
        img_table = (stimulus.StaticImage.Image & 'image_class = "imagenet_v2_rgb"') & id_table

        image_class, images = (id_table * img_table).fetch('image_class','image')
        image_class = np.unique(image_class)

        if len(image_class) > 1:
            raise ValueError("There must be only 1 image class! Currently there are : {} classes".format(image_class))
        else:
            image_class = image_class[0]

        # find which colors were used for imagenet images
        colors = np.array((id_table & 'trial_idx = 0').fetch1('channel_1','channel_2','channel_3'))
        
        valid_colors =(colors[colors != None] - 1).astype('int')

        # obtain the kernels
        kernel_table = eval(kernel_params['kernel_type']) & 'kernel_hash = "{}"'.format(kernel_params['kernel_hash'])
        kernels = kernel_table.fetch1('kernels')

        # number of neurons = number of kernels
        num_neurons = kernels.shape[0]

        # generate fake values
        response_block = np.zeros(shape=(len(images),len(kernels)))

        for img_ind, img in enumerate(images):
            print(img_ind)
            #TODO color_ind logic can be improved. For now, hardcoded for kernel channel config
            restricted_img = np.zeros(shape=kernels.shape[1:])
            
            for channel_ind, color_ind in enumerate(valid_colors):
                restricted_img[:,:,channel_ind] = img[:,:,color_ind]  
            
            convolved = np.multiply(restricted_img, kernels)
            
            # introduce nonlinearity
            convolved[convolved<0] = 0 

            signals = convolved.reshape(num_neurons,-1).mean(-1)
                
            response_block[img_ind,:] = signals

        # generate response_hash
        response_hash = DJTableBase.generate_md5_hash(dict(kernel_params, input_image_class=image_class))
    
        #update our key
        key.update(kernel_params)
        FakeResponseFromImages.insert1(dict(key,
                                            response_hash=response_hash,        
                                            input_image_class=image_class,
                                            response=response_block))

    @staticmethod
    def generate_h5_with_fake_signal(key, orig_h5_path):
        """
        From the original h5 file, replace responeses with fake data and 
        create a new h5 file.
        
        Input:
            key(dict): animal_id, session, scan_idx
            orig_h5_path(str): path to the original h5 for the key
                
        Return:
            None
        
        """
        
        #ensure Fake data has been generated first
        if len(FakeResponseFromImages & key) !=1:
            raise ValueError("Either the fake data hasn't been populated or there are more than 1 fake data!")
        else:
            fake_response = (FakeResponseFromImages & key).fetch1('response')

        # copy the h5 file with new name if it doesnt exist
        f_name = 'fake_responses'
        for val in key.values():
            f_name +=str(val) + '-'
        f_name = f_name[:-1] + '.h5'

        fake_file_path = os.path.join(os.path.dirname(orig_h5_path), f_name)

        if not os.path.exists(fake_file_path):
            shutil.copyfile(orig_h5_path, fake_file_path)
        else:
            raise ValueError("fake data already exists at: {}! It might be corrupted!".format(fake_file_path))

        # load the fake h5 file. Upto this pt, fake data = original data. We need to update our responses
        fake_h5 = h5py.File(fake_file_path)

        valid_trials = find_valid_trials_from_behavior(key)
        
        # only retain valid response
        fake_response = fake_response[valid_trials]

        # now delete responses
        del fake_h5['responses']

        # save new responses
        fake_h5.create_dataset('responses', data=fake_response)
        fake_h5.close()

        # now test if the fake response is the same as the one we just saved
        fake_reloaded = h5py.File(fake_file_path, 'r')
        assert np.allclose(fake_reloaded['responses'].value, fake_response)


    @staticmethod
    def get_fake_data_path(key):
        # Assume that fake data is in the same path as the original one
        orig_h5_path = (InputResponse & target_scans[0]).get_filename()

        f_name = 'fake_responses'
        for val in key.values():
            f_name +=str(val) + '-'
        f_name = f_name[:-1] + '.h5'

        return os.path.join(os.path.dirname(orig_h5_path), f_name)


@schema
class FakeStaticSpikeTriggeredAverageRF(dj.Manual):
    definition = """
    # spike trigerred average using static image (e.g. imagenet)
    -> FakeResponseFromImages
    ---
    sta_rf                 : external              # STA using fake response
    """

    @classmethod
    def fill(cls, key):
        # Get data
        print('Loading data')
        response_block = (FakeResponseFromImages & key).fetch1('response')
        frames = (Frame * (stimulus.Trial & key)).fetch('frame')
        frames = np.stack(frames).astype(np.float32)

        print('Iterating over units')
        sta_rf = np.zeros(shape=(response_block.shape[-1],*frames.shape[1:]))

        for ind, response in enumerate(response_block.T):
            sta_rf[ind] = np.average(frames, weights=response/response.sum(), axis=0)
            
        cls.insert1(dict(key, sta_rf=sta_rf))
        
    @classmethod
    def plot_sta(cls, key, num_to_plot=10):

        sta_rf = (cls & key).fetch1('sta_rf')
        # frames = (Frame * (stimulus.Trial & key)).fetch('frame', limit=num_to_plot)
        
        fig, ax_list = plt.subplots(num_to_plot, 4, figsize=(10,20))
        
        ax_list[0][0].set_title('STA RF')
        ax_list[0][1].set_title('STA RF blue')
        ax_list[0][2].set_title('STA RF UV')
        ax_list[0][3].set_title('STA RF blue - UV')

        for ind, ax in enumerate(ax_list):

            temp_img = np.zeros((*(sta_rf[ind,:].shape[:-1]), 3))
            temp_img[:,:,:2] = sta_rf[ind,:]
            for i in range(2):
                temp_img[:,:,i] = (temp_img[:,:,i] - temp_img[:,:,i].min())/\
                                (temp_img[:,:,i].max()- temp_img[:,:,i].min())
            ax[0].imshow(temp_img)
            ax[0].set_axis_off()
            ax[1].imshow(sta_rf[ind,:,:,0])
            ax[1].set_axis_off()
            ax[2].imshow(sta_rf[ind,:,:,1])
            ax[2].set_axis_off()
            ax[3].imshow(sta_rf[ind,:,:,0] - sta_rf[ind,:,:,1])
            ax[3].set_axis_off()

        return fig


@schema
class FakeResponseFromImagesMethod2(dj.Manual, DJTableBase):
    definition = """
    # fake signals generated by convolving input images with kernels
    ->experiment.Scan
    response_hash       : char(32)              # MD5 in base64
    ---
    input_image_class   : varchar(128)          # stimulus image class
    kernel_type         : varchar(128)          # kernel type
    kernel_hash         : char(32)              # MD5 in base64
    response            : external              # averaged fake response
    """

    @classmethod
    def fill(cls,key, kernel_params):
        """
        fill in FakeResponseFromImages table
        Input:
            key(dict): animal_id, session, scan_idx
            kernel_params(dict): 
                kernel_type: table name for the kernel.
                kernel_hash: kernel_hash to uniquely identify which kernel to use from kernel_type table
        Return:
            None
        """
        
        # find the key's stimulus type and its table, then fetch image data
        stim_type = np.unique((stimulus.Condition & (stimulus.Trial & key)).fetch('stimulus_type'))

        if len(stim_type) > 1:
            raise ValueError("There must be only 1 stimulus type! Currently there are stimuli of : {}".format(stim_type))
        else:
            stim_type = stim_type[0].split('.')[1]
        stim_table = getattr(stimulus, stim_type)

        id_table = ((stimulus.Trial & key) * stim_table).proj('image_id','channel_1','channel_2','channel_3')
        img_table = (stimulus.StaticImage.Image & 'image_class = "imagenet_v2_rgb"') & id_table

        image_class, images = (id_table * img_table).fetch('image_class','image')
        image_class = np.unique(image_class)

        if len(image_class) > 1:
            raise ValueError("There must be only 1 image class! Currently there are : {} classes".format(image_class))
        else:
            image_class = image_class[0]

        # find which colors were used for imagenet images
        colors = np.array((id_table & 'trial_idx = 0').fetch1('channel_1','channel_2','channel_3'))
        
        valid_colors =(colors[colors != None] - 1).astype('int')

        # obtain the kernels
        kernel_table = eval(kernel_params['kernel_type']) & 'kernel_hash = "{}"'.format(kernel_params['kernel_hash'])
        kernels = np.repeat(np.expand_dims(kernel_table.fetch1('kernels'),axis=3),repeats=2,axis=3)

        # number of neurons = number of kernels
        num_neurons = kernels.shape[0]

        # generate fake values
        response_block = np.zeros(shape=(len(images), num_neurons, 2))

        for img_ind, img in enumerate(images):
            print(img_ind)
            #TODO color_ind logic can be improved. For now, hardcoded for kernel channel config
            restricted_img = np.zeros(shape=(*img.shape[:-1],2))

            for channel_ind, color_ind in enumerate(valid_colors):
                restricted_img[:,:,channel_ind] = img[:,:,color_ind]  

            resized_img = cv2.resize(restricted_img,(32,18),interpolation=cv2.INTER_AREA)[1:-1,8:-8]

            response_block[img_ind,:] = np.multiply(resized_img,kernels).reshape(num_neurons,-1,2).mean(1)
        
        # generate response_hash
        response_hash = DJTableBase.generate_md5_hash(dict(kernel_params, input_image_class=image_class))
    
        # update our key
        key.update(kernel_params)
        cls.insert1(dict(key, response_hash=response_hash,        
                              input_image_class=image_class,
                              response=response_block))

    @staticmethod
    def generate_h5_with_fake_signal(key, orig_h5_path):
        """
        From the original h5 file, replace responeses with fake data and 
        create a new h5 file.
        
        Input:
            key(dict): animal_id, session, scan_idx
            orig_h5_path(str): path to the original h5 for the key
                
        Return:
            None
        
        """
        
        #ensure Fake data has been generated first
        if len(FakeResponseFromImages & key) !=1:
            raise ValueError("Either the fake data hasn't been populated or there are more than 1 fake data!")
        else:
            fake_response = (FakeResponseFromImages & key).fetch1('response')

        # copy the h5 file with new name if it doesnt exist
        f_name = 'fake_responses'
        for val in key.values():
            f_name +=str(val) + '-'
        f_name = f_name[:-1] + '.h5'

        fake_file_path = os.path.join(os.path.dirname(orig_h5_path), f_name)

        if not os.path.exists(fake_file_path):
            shutil.copyfile(orig_h5_path, fake_file_path)
        else:
            raise ValueError("fake data already exists at: {}! It might be corrupted!".format(fake_file_path))

        # load the fake h5 file. Upto this pt, fake data = original data. We need to update our responses
        fake_h5 = h5py.File(fake_file_path)

        valid_trials = find_valid_trials_from_behavior(key)
        
        # only retain valid response
        fake_response = fake_response[valid_trials]

        # now delete responses
        del fake_h5['responses']

        # save new responses
        fake_h5.create_dataset('responses', data=fake_response)
        fake_h5.close()

        # now test if the fake response is the same as the one we just saved
        fake_reloaded = h5py.File(fake_file_path, 'r')
        assert np.allclose(fake_reloaded['responses'].value, fake_response)


    @staticmethod
    def get_fake_data_path(key):
        # Assume that fake data is in the same path as the original one
        orig_h5_path = (InputResponse & target_scans[0]).get_filename()

        f_name = 'fake_responses'
        for val in key.values():
            f_name +=str(val) + '-'
        f_name = f_name[:-1] + '.h5'

        return os.path.join(os.path.dirname(orig_h5_path), f_name)


@schema
class FakeStaticSpikeTriggeredAverageRFMethod2(dj.Manual):
    definition = """
    # spike trigerred average using static image (e.g. imagenet)
    -> FakeResponseFromImagesMethod2
    ---
    sta_rf                 : external              # STA using fake response
    """

    @classmethod
    def fill(cls, key):
        # Get data
        print('Loading data')
        # find the key's stimulus type and its table, then fetch image data
        stim_type = np.unique((stimulus.Condition & (stimulus.Trial & key)).fetch('stimulus_type'))

        if len(stim_type) > 1:
            raise ValueError("There must be only 1 stimulus type! Currently there are stimuli of : {}".format(stim_type))
        else:
            stim_type = stim_type[0].split('.')[1]
        stim_table = getattr(stimulus, stim_type)

        id_table = ((stimulus.Trial & key) * stim_table).proj('image_id','channel_1','channel_2','channel_3')
        img_table = (stimulus.StaticImage.Image & 'image_class = "imagenet_v2_rgb"') & id_table

        image_class, images = (id_table * img_table).fetch('image_class','image')
        image_class = np.unique(image_class)

        if len(image_class) > 1:
            raise ValueError("There must be only 1 image class! Currently there are : {} classes".format(image_class))
        else:
            image_class = image_class[0]

        #resize images to 16 by 16
        images = np.stack(images)[:,:,:,1:] #select only green and blue channels

        images_resized = np.zeros(shape=(len(images),16,16,2))

        for ind, img in enumerate(images):
            images_resized[ind] = cv2.resize(img,(32,18),interpolation=cv2.INTER_AREA)[1:-1,8:-8]

        images_resized = images_resized.transpose(3,1,2,0)

        response_block = (FakeResponseFromImagesMethod2 & key).fetch1('response').transpose(2,1,0)

        print('Iterating over units')
        sta_rf = np.zeros(shape=(response_block.shape[1]*2, *frames.shape[1:]))

        for channel_ind, (channel_response, channel_img)  in enumerate(zip(response_block, images_resized)):
    
            for ind, response in enumerate(channel_response):
                
                positive_relu = deepcopy(response)
                positive_relu[positive_relu<0] = 0
                
                negative_relu = deepcopy(response) * -1
                negative_relu[negative_relu<0] = 0
                
                
                sta_rf[:response_block.shape[1],:,:,channel_ind] = np.average(channel_img, weights=positive_relu/positive_relu.sum(),axis=-1)
                sta_rf[response_block.shape[1]:,:,:,channel_ind] = np.average(channel_img, weights=negative_relu/negative_relu.sum(),axis=-1)

        #     print(channel_response.shape, channel_img.shape)
            
        cls.insert1(dict(key, sta_rf=sta_rf))
        
    @classmethod
    def plot_sta(cls, key, num_to_plot=10):

        sta_rf = (cls & key).fetch1('sta_rf')
        # frames = (Frame * (stimulus.Trial & key)).fetch('frame', limit=num_to_plot)
        
        fig, ax_list = plt.subplots(num_to_plot, 4, figsize=(10,20))
        
        ax_list[0][0].set_title('STA RF')
        ax_list[0][1].set_title('STA RF blue')
        ax_list[0][2].set_title('STA RF UV')
        ax_list[0][3].set_title('STA RF blue - UV')

        for ind, ax in enumerate(ax_list):

            temp_img = np.zeros((*(sta_rf[ind,:].shape[:-1]), 3))
            temp_img[:,:,:2] = sta_rf[ind,:]
            for i in range(2):
                temp_img[:,:,i] = (temp_img[:,:,i] - temp_img[:,:,i].min())/\
                                (temp_img[:,:,i].max()- temp_img[:,:,i].min())
            ax[0].imshow(temp_img)
            ax[0].set_axis_off()
            ax[1].imshow(sta_rf[ind,:,:,0])
            ax[1].set_axis_off()
            ax[2].imshow(sta_rf[ind,:,:,1])
            ax[2].set_axis_off()
            ax[3].imshow(sta_rf[ind,:,:,0] - sta_rf[ind,:,:,1])
            ax[3].set_axis_off()

        return fig


@schema
class TransformedImagenetImages(dj.Manual, DJTableBase):
    definition = """
    # Given an album num, generate 4 cases: cor + biased, uncor + biased, cor + biased, uncor + unbiased
    -> Frame
    response_hash       : char(32)              # MD5 in base64
    ---
    input_image_class   : varchar(128)          # stimulus image class
    kernel_type         : varchar(128)          # kernel type
    kernel_hash         : char(32)              # MD5 in base64
    response            : external              # averaged fake response
    """