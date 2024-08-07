#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2020
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

import json
import numpy as np
from models.setup import *
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import scipy
import scipy.signal
import librosa
from tqdm import tqdm
sys.path.append("..")
# from preprocessing.audio_preprocessing import extract_features
import warnings
warnings.filterwarnings("ignore")

scipy_windows = {
    'hamming': scipy.signal.hamming,
    'hann': scipy.signal.hann, 
    'blackman': scipy.signal.blackman,
    'bartlett': scipy.signal.bartlett
    }

categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148, 149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]

def preemphasis(signal,coeff=0.97):  
    # function adapted from https://github.com/dharwath
    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class ImageAudioDatawithSampling(Dataset):
    def __init__(self, image_base_path, dataset_json_file, lookup_file, args, rank):

        with open(dataset_json_file, 'r') as fp:
            data = json.load(fp)
        self.image_base_path = Path(image_base_path).absolute()

        self.data = data
        self.n = 10

        lookups = np.load(lookup_file, allow_pickle=True)
        self.pos_masks_to_images = lookups['lookup'].item()
        self.neg_masks_to_images = lookups['neg_lookup'].item()
        if rank ==0: 
            print(f'\rRead in data paths from:')
            printDirectory(dataset_json_file)
            print(f'\n\rRead in {len(self.data)} data points')
        
        self.audio_conf = args["audio_config"]
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.padval = self.audio_conf.get('padval', 0)
        self.image_conf = args["image_config"]
        self.crop_size = self.image_conf.get('crop_size')
        center_crop = self.image_conf.get('center_crop')
        RGB_mean = self.image_conf.get('RGB_mean')
        RGB_std = self.image_conf.get('RGB_std')

        if center_crop:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.CenterCrop(224), transforms.ToTensor()])
        else:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.RandomResizedCrop(self.crop_size), transforms.ToTensor()])

        self.resize = transforms.Resize((256, 256))
        self.to_tensor = transforms.ToTensor()
        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

    def _myRandomCrop(self, im):

        im = self.resize(im)
        im = self.to_tensor(im)
        return im

    def _LoadImage(self, impath):

        img = Image.open(impath).convert('RGB')
        img = self._myRandomCrop(img)
        img = self.image_normalize(img)
        return img

    def _SampleNeg(self, im_name, english_wav_name, dutch_wav_name, im_path, id):
        n = self.n + 1
        negatives = []

        neg_ids = np.random.choice(list(self.neg_masks_to_images[id]['english'].keys()), size=n)

        for i in range(n):

            temp = self.neg_masks_to_images[id]['english'][neg_ids[i]].copy()

            neg_eng_fn = np.random.choice(temp, size=1)[0]
            neg_eng_name = str(Path(neg_eng_fn).stem)
            neg_eng_fn = Path('data/english') / str(neg_eng_name + '.npz')
            if english_wav_name == neg_eng_name or neg_ids[i] == id: print("Audio negative sampling went wrong.")
            # data_point = np.load(neg_eng_fn)
            english_audio_feat1 = np.load(neg_eng_fn)["audio_feat"]
            english_audio_feat1, english_nframes1 = self._PadFeat(english_audio_feat1)



            temp = self.neg_masks_to_images[id]['dutch'][neg_ids[i]].copy()

            neg_dutch_fn = np.random.choice(temp, size=1)[0]
            neg_dutch_name = str(Path(neg_dutch_fn).stem)
            neg_dutch_fn = Path('data/dutch') / str(neg_dutch_name + '.npz')
            if dutch_wav_name == neg_dutch_name: print("Audio negative sampling went wrong.")
            dutch_audio_feat1 = np.load(neg_dutch_fn)["audio_feat"]
            dutch_audio_feat1, dutch_nframes1 = self._PadFeat(dutch_audio_feat1)


            temp = self.neg_masks_to_images[id]['images'][neg_ids[i]].copy()

            neg_img_fn = np.random.choice(temp, size=1)[0]
            neg_img_name = str(Path(neg_img_fn).stem)
            if im_name == neg_img_name or neg_ids[i] == id:  print("Image negative sampling went wrong.")
            imgpath = self.image_base_path / neg_img_fn
            img1 = Image.open(imgpath).convert('RGB')
            img1 = self.resize(img1)
            
            img1 = self._myRandomCrop(img1)
            img1 = self.image_normalize(img1)

            negatives.append({
                "neg_english": english_audio_feat1, 
                "neg_dutch": dutch_audio_feat1, 
                "neg_image": img1, 
                "neg_id": neg_ids[i]
                })

        return negatives

    def _PadFeat(self, feat):
        
        nframes = feat.shape[1]
        pad = self.target_length - nframes
        
        if pad > 0:
            feat = np.pad(feat, ((0, 0), (0, pad)), 'constant',
                constant_values=(self.padval, self.padval))
        elif pad < 0:
            nframes = self.target_length
            feat = feat[:, 0: pad]

        return feat, nframes

    def __getitem__(self, index):

        num_pos = 5
        
        data_point = np.load(self.data[index] + ".npz", allow_pickle=True)
        ids = list(data_point['ids'])
        id = np.random.choice(ids, 1)[0]

        english_audio_feat = data_point["audio_feat"]
        if len(english_audio_feat.shape) != 2: print(id, '\n')
        english_audio_feat, english_nframes = self._PadFeat(english_audio_feat)
        wav_name = str(Path(self.data[index]).stem)

        # id_lookup = self.pos_masks_to_images[id]['images'].copy()
        names = list(self.pos_masks_to_images[id]['images'].keys())
        image_name = np.random.choice(names, size=num_pos+1, replace=False)
        imgpath = np.random.choice(self.pos_masks_to_images[id]['images'][image_name[0]], size=1)[0]
        imgpath = self.image_base_path / str(imgpath)
        image = self._LoadImage(imgpath)


        names = list(self.pos_masks_to_images[id]["dutch"].keys())
        dutch_name = np.random.choice(names, size=num_pos+1, replace=False)
        dutchpath = np.random.choice(self.pos_masks_to_images[id]["dutch"][dutch_name[0]], size=1)[0]
        dutchpath = Path('data/dutch') / Path(dutchpath.stem + '.npz')
        dutch_audio_feat1 = np.load(dutchpath)["audio_feat"]
        dutch_audio_feat1, dutch_nframes1 = self._PadFeat(dutch_audio_feat1)


        positives = []
        for p in range(num_pos):
            imgpath = np.random.choice(self.pos_masks_to_images[id]['images'][image_name[p+1]], size=1)[0]
            imgpath = self.image_base_path / str(imgpath)
            pos_image1 = self._LoadImage(imgpath)


            id_lookup = self.pos_masks_to_images[id]['english'].copy()
            names = list(id_lookup.keys())
            names.remove(wav_name)
            pos_wav_name = np.random.choice(names, size=1)[0]
            pos_wav_path = Path('data/english') / str(pos_wav_name + '.npz')
            pos_english_audio_feat1 = np.load(pos_wav_path)["audio_feat"]
            if len(pos_english_audio_feat1.shape) != 2: print("positive ", p, id, '\n')
            pos_english_audio_feat1, pos_english_nframes1 = self._PadFeat(pos_english_audio_feat1)


            pos_dutch_path = Path('data/dutch') / str(dutch_name[p+1] + '.npz')
            pos_dutch_audio_feat1 = np.load(pos_dutch_path)["audio_feat"]
            pos_dutch_audio_feat1, pos_dutch_nframes1 = self._PadFeat(pos_dutch_audio_feat1)
            
            positives.append({
                "pos_english": pos_english_audio_feat1, 
                "pos_dutch": pos_dutch_audio_feat1,
                "pos_image": pos_image1})

        # pos_english_audio_feat1, pos_english_nframes1, pos_image1, pos_english_audio_feat2, pos_english_nframes2, pos_image2, pos_english_audio_feat3, pos_english_nframes3, pos_image3 = self._SamplePosImage(image_name, self.data[index], id)
        # # neg_english_audio_feat1, neg_english_nframes1, neg_image1, neg_id1, neg_english_audio_feat2, neg_english_nframes2, neg_image2, neg_id2 = self._SampleNegImage(image_name, self.data[index], id)
        negatives = self._SampleNeg(image_name[0], wav_name, dutch_name[0], self.data[index], id)
        
        return {"image": image, 
        "english_feat": english_audio_feat, 
        "dutch_feat": dutch_audio_feat1,
        "positives": positives,
        "negatives": negatives
        }   

    def __len__(self):
        return len(self.data)


class ImageAudioDatawithSamplingVal(Dataset):
    def __init__(self, image_base_path, dataset_json_file, lookup_file, args, rank):

        with open(dataset_json_file, 'r') as fp:
            data = json.load(fp)
        self.image_base_path = Path(image_base_path).absolute()

        self.data = data
        self.n = 5

        lookups = np.load(lookup_file, allow_pickle=True)
        self.pos_masks_to_images = lookups['lookup'].item()
        self.neg_masks_to_images = lookups['neg_lookup'].item()
        if rank ==0: 
            print(f'\rRead in data paths from:')
            printDirectory(dataset_json_file)
            print(f'\n\rRead in {len(self.data)} data points')
        
        self.audio_conf = args["audio_config"]
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.padval = self.audio_conf.get('padval', 0)
        self.image_conf = args["image_config"]
        self.crop_size = self.image_conf.get('crop_size')
        center_crop = self.image_conf.get('center_crop')
        RGB_mean = self.image_conf.get('RGB_mean')
        RGB_std = self.image_conf.get('RGB_std')

        if center_crop:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.CenterCrop(224), transforms.ToTensor()])
        else:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.RandomResizedCrop(self.crop_size), transforms.ToTensor()])

        self.resize = transforms.Resize((256, 256))
        self.to_tensor = transforms.ToTensor()
        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

    def _myRandomCrop(self, im):

        im = self.resize(im)
        im = self.to_tensor(im)
        return im

    def _LoadImage(self, impath):

        img = Image.open(impath).convert('RGB')
        img = self._myRandomCrop(img)
        img = self.image_normalize(img)
        return img

    def _SampleNeg(self, im_name, wav_name, im_path, id):
        n = self.n
        negatives = []

        all_aud_ids = list(self.neg_masks_to_images[id]['english'].keys())
        neg_aud_id = np.random.choice(all_aud_ids, size=n)
        all_im_ids = list(self.neg_masks_to_images[id]['images'].keys())
        neg_im_id = np.random.choice(all_im_ids, size=n)

        for i in range(n):

            temp = self.neg_masks_to_images[id]['english'][neg_aud_id[i]].copy()

            neg_aud_fn = np.random.choice(temp, size=1)[0]
            neg_aud_name = str(Path(neg_aud_fn).stem)
            neg_aud_fn = Path('data/english') / str(neg_aud_name + '.npz')
            if wav_name == neg_aud_name or neg_aud_id[i] == id: print("Audio negative sampling went wrong.")
            data_point = np.load(neg_aud_fn)
            english_audio_feat1 = np.load(neg_aud_fn)["audio_feat"]
            english_audio_feat1, english_nframes1 = self._PadFeat(english_audio_feat1)

            temp = self.neg_masks_to_images[id]['images'][neg_im_id[i]].copy()

            neg_img_fn = np.random.choice(temp, size=1)[0]
            neg_img_name = str(Path(neg_img_fn).stem)
            if im_name == neg_img_name or neg_im_id[i] == id:  print("Image negative sampling went wrong.")
            imgpath = self.image_base_path / neg_img_fn
            img1 = Image.open(imgpath).convert('RGB')
            img1 = self.resize(img1)
            
            img1 = self._myRandomCrop(img1)
            img1 = self.image_normalize(img1)


            # neg_img_fn = np.random.choice(self.base_neg_imgs, size=1)[0]
            # neg_img_name = str(Path(neg_img_fn).stem)
            # if im_name == neg_img_name or neg_im_id[i] == id:  print("Image negative sampling went wrong.")
            # imgpath = self.image_base_path / neg_img_fn
            # img2 = Image.open(imgpath).convert('RGB')
            # img2 = self.resize(img2)
            
            # img2 = self._myRandomCrop(img2)
            # img2 = self.image_normalize(img2)

            negatives.append({
                "neg_audio": english_audio_feat1, 
                "neg_frames": english_nframes1, 
                "neg_image": img1, 
                "neg_img_id": neg_im_id[i], 
                "neg_aud_id": neg_aud_id[i],
                # "base_neg_image": img2
                })
        return negatives

    def _PadFeat(self, feat):
        nframes = feat.shape[1]
        pad = self.target_length - nframes
        
        if pad > 0:
            feat = np.pad(feat, ((0, 0), (0, pad)), 'constant',
                constant_values=(self.padval, self.padval))
        elif pad < 0:
            nframes = self.target_length
            feat = feat[:, 0: pad]

        return feat, nframes

    def __getitem__(self, index):

        num_pos = self.n

        data_point = np.load(self.data[index] + ".npz", allow_pickle=True)
        ids = list(data_point['ids'])
        id = np.random.choice(ids, 1)[0]

        english_audio_feat = data_point["audio_feat"]
        english_audio_feat, english_nframes = self._PadFeat(english_audio_feat)

        # id_lookup = self.pos_masks_to_images[id]['images'].copy()
        names = list(self.pos_masks_to_images[id]['images'].keys())
        image_name = np.random.choice(names, size=num_pos+1, replace=False)
        imgpath = np.random.choice(self.pos_masks_to_images[id]['images'][image_name[0]], size=1)[0]
        imgpath = self.image_base_path / str(imgpath)
        image = self._LoadImage(imgpath)

        positives = []
        for p in range(num_pos):
            imgpath = np.random.choice(self.pos_masks_to_images[id]['images'][image_name[p+1]], size=1)[0]
            imgpath = self.image_base_path / str(imgpath)
            pos_image1 = self._LoadImage(imgpath)

            id_lookup = self.pos_masks_to_images[id]['english'].copy()
            names = list(id_lookup.keys())
            wav_name = str(Path(self.data[index]).stem)
            names.remove(wav_name)
            pos_wav_name = np.random.choice(names, size=2)[0]
            pos_wav_path = Path('data/english') / str(pos_wav_name + '.npz')
            pos_english_audio_feat1 = np.load(pos_wav_path)["audio_feat"]
            pos_english_audio_feat1, pos_english_nframes1 = self._PadFeat(pos_english_audio_feat1)
            positives.append({"pos_audio": pos_english_audio_feat1, "pos_frames": pos_english_nframes1, "pos_image": pos_image1})

        # pos_english_audio_feat1, pos_english_nframes1, pos_image1, pos_english_audio_feat2, pos_english_nframes2, pos_image2, pos_english_audio_feat3, pos_english_nframes3, pos_image3 = self._SamplePosImage(image_name, self.data[index], id)
        # # neg_english_audio_feat1, neg_english_nframes1, neg_image1, neg_id1, neg_english_audio_feat2, neg_english_nframes2, neg_image2, neg_id2 = self._SampleNegImage(image_name, self.data[index], id)
        negatives = self._SampleNeg(image_name[0], wav_name, self.data[index], id)
        
        return {"image": image, "english_feat": english_audio_feat, "english_nframes": english_nframes,
        "positives": positives,
        "negatives": negatives
        }

    def __len__(self):
        return len(self.data)

class ImageAudioDataKWS(Dataset):
    def __init__(self, image_base_path, dataset_json_file, args, rank):

        with open(dataset_json_file, 'r') as fp:
            data = json.load(fp)
        self.image_base_path = Path(image_base_path).absolute()

        self.data = data
        if rank == 0: 
            print(f'\rRead in data paths from:')
            printDirectory(dataset_json_file)
            print(f'\n\rRead in {len(self.data)} data points')
        
        self.audio_conf = args["audio_config"]
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.padval = self.audio_conf.get('padval', 0)
        self.image_conf = args["image_config"]
        self.crop_size = self.image_conf.get('crop_size')
        center_crop = self.image_conf.get('center_crop')
        RGB_mean = self.image_conf.get('RGB_mean')
        RGB_std = self.image_conf.get('RGB_std')

        self.to_tensor = transforms.ToTensor()
        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

    def _myRandomCrop(self, im, seg, id):
        # print((seg == id).any())
        grid = np.where(seg == id)
        ind = np.random.randint(0, len(grid[0]))
        y_ind = grid[0][ind]
        x_ind = grid[1][ind]

        x_coord = np.random.randint(max(x_ind - self.crop_size + 1, 0), min(x_ind, im.size[1] - self.crop_size) + 1, size=1)
        y_coord = np.random.randint(max(y_ind - self.crop_size + 1, 0), min(y_ind, im.size[0] - self.crop_size) + 1, size=1)

        im = transforms.functional.crop(im, y_coord[0], x_coord[0], self.crop_size, self.crop_size)
        im = self.to_tensor(im)
        return im

    def _LoadImage(self, impath, imseg):
        ids = np.unique(imseg)
        id = np.random.choice(ids, 1)[0]

        img = Image.open(impath).convert('RGB')
        # img = self.image_resize_and_crop(img)
        img = self._myRandomCrop(img, imseg, id)
        img = self.image_normalize(img)
        return img

    def _PadFeat(self, feat):
        nframes = feat.shape[1]
        pad = self.target_length - nframes
        
        if pad > 0:
            feat = np.pad(feat, ((0, 0), (0, pad)), 'constant',
                constant_values=(self.padval, self.padval))
        elif pad < 0:
            nframes = self.target_length
            feat = feat[:, 0: pad]

        return feat, nframes

    def __getitem__(self, index):

        data_point = np.load(self.data[index] + ".npz")

        english_audio_feat = data_point["eng_audio_feat"]
        english_audio_feat, english_nframes = self._PadFeat(english_audio_feat)
        
        imgpath = self.image_base_path / str(data_point['image_fn'])
        image_seg = data_point['panoptic_segmentation']
        image = self._LoadImage(imgpath, image_seg)
        return image, english_audio_feat, english_nframes

    def __len__(self):
        return len(self.data)


class ImageAudioData(Dataset):
    def __init__(self, image_base_path, dataset_json_file, args, rank):

        with open(dataset_json_file, 'r') as fp:
            data = json.load(fp)
        self.image_base_path = Path(image_base_path).absolute()

        self.data = data[0:1000]
        if rank == 0: 
            print(f'\rRead in data paths from:')
            printDirectory(dataset_json_file)
            print(f'\n\rRead in {len(self.data)} data points')
        
        self.audio_conf = args["audio_config"]
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.padval = self.audio_conf.get('padval', 0)
        self.image_conf = args["image_config"]
        self.crop_size = self.image_conf.get('crop_size')
        center_crop = self.image_conf.get('center_crop')
        RGB_mean = self.image_conf.get('RGB_mean')
        RGB_std = self.image_conf.get('RGB_std')

        # if center_crop:
        self.image_resize_and_crop = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        # else:
        #     self.image_resize_and_crop = transforms.Compose(
        #         [transforms.RandomResizedCrop(self.crop_size), transforms.ToTensor()])
        self.to_tensor = transforms.ToTensor()
        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

    def _LoadImage(self, impath, imseg):
        ids = np.unique(imseg)
        id = np.random.choice(ids, 1)[0]

        img = Image.open(impath).convert('RGB')
        # img = self.image_resize_and_crop(img)
        img = self.image_resize_and_crop(img)
        img = self.image_normalize(img)

        h, w = imseg.shape
        w_crop = (w - self.crop_size) // 2
        h_crop = (h - self.crop_size) // 2
        ids = np.unique(imseg[h_crop:h_crop+self.crop_size, w_crop:w_crop+self.crop_size])
        all_ids = torch.zeros(categories[-1]+1)
        for an_id in list(ids): all_ids[an_id] = 1

        return img, id, all_ids

    def _PadFeat(self, feat):
        nframes = feat.shape[1]
        pad = self.target_length - nframes
        
        if pad > 0:
            feat = np.pad(feat, ((0, 0), (0, pad)), 'constant',
                constant_values=(self.padval, self.padval))
        elif pad < 0:
            nframes = self.target_length
            feat = feat[:, 0: pad]

        return feat, nframes

    def __getitem__(self, index):

        data_point = np.load(self.data[index] + ".npz")

        english_audio_feat = data_point["audio_feat"]
        english_audio_feat, english_nframes = self._PadFeat(english_audio_feat)
        
        imgpath = self.image_base_path / str(data_point['image'])
        image_seg = data_point['panoptic_segmentation']
        image, id, all_ids = self._LoadImage(imgpath, image_seg)
        return image, english_audio_feat, english_nframes, id, all_ids

    def __len__(self):
        return len(self.data)

class AudioData(Dataset):
    def __init__(self, root, args):

        self.root = root

        self.data = list(self.root.rglob("*.wav"))

        print(f'\n\rRead in {len(self.data)} data points')
        
        self.audio_conf = args["audio_config"]
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.padval = self.audio_conf.get('padval', 0)
    
    def _LoadAudio(self, path):

        audio_type = self.audio_conf.get('audio_type')
        if audio_type not in ['melspectrogram', 'spectrogram']:
            raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
        
        preemph_coef = self.audio_conf.get('preemph_coef')
        sample_rate = self.audio_conf.get('sample_rate')
        window_size = self.audio_conf.get('window_size')
        window_stride = self.audio_conf.get('window_stride')
        window_type = self.audio_conf.get('window_type')
        num_mel_bins = self.audio_conf.get('num_mel_bins')
        target_length = self.audio_conf.get('target_length')
        fmin = self.audio_conf.get('fmin')
        n_fft = self.audio_conf.get('n_fft', int(sample_rate * window_size))
        win_length = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)

        # load audio, subtract DC, preemphasis
        y, sr = librosa.load(path, sample_rate)
        if y.size == 0:
            y = np.zeros(target_length)
        y = y - y.mean()
        y = preemphasis(y, preemph_coef)

        # compute mel spectrogram / filterbanks
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=scipy_windows.get(window_type, scipy_windows['hamming']))
        spec = np.abs(stft)**2 # Power spectrum
        if audio_type == 'melspectrogram':
            mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
            melspec = np.dot(mel_basis, spec)
            logspec = librosa.power_to_db(melspec, ref=np.max)
        elif audio_type == 'spectrogram':
            logspec = librosa.power_to_db(spec, ref=np.max)
        # n_frames = logspec.shape[1]
        logspec = torch.FloatTensor(logspec)
        return torch.tensor(logspec)#, n_frames

    def _PadFeat(self, feat):
        nframes = feat.shape[1]
        pad = self.target_length - nframes
        
        if pad > 0:
            feat = np.pad(feat, ((0, 0), (0, pad)), 'constant',
                constant_values=(self.padval, self.padval))
        elif pad < 0:
            nframes = self.target_length
            feat = feat[:, 0: pad]

        return feat, nframes

    def __getitem__(self, index):

        data_path = self.data[index].with_suffix(".wav")

        audio_feat = self._LoadAudio(data_path)

        return audio_feat, str(self.data[index].stem)

    def __len__(self):
        return len(self.data)

class ImageCaptionDatasetWithPreprocessing(Dataset):
    def __init__(self, dataset_json_file, audio_conf=None, image_conf=None, add="train"):
        """
        Dataset that manages a set of paired images and audio recordings

        :param dataset_json_file
        :param audio_conf: Dictionary containing the sample rate, window and
        the window length/stride in seconds, and normalization to perform (optional)
        :param image_transform: torchvision transform to apply to the images (optional)
        """

        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        data = data_json['data']
        self.image_base_path = data_json['image_base_path']
        self.audio_base_path = data_json['audio_base_path']

        # if add == "train":
        #     with open("train_test.json", "r") as outfile:
        #         paths = json.load(outfile)

        #     self.data = []
        #     for entry in data:
        #         test = f'data/PlacesAudio_400k_distro+imagesPlaces205_resize/speaker_{entry["speaker"]}+{entry["wav"].split("/")[-1].split(".")[0]}+{entry["image"].split("/")[-1].split(".")[0]}'
        #         if test in paths: self.data.append(entry)
        #     print(len(self.data))
        # else:
        self.data = data
        
        if not audio_conf:
            self.audio_conf = {}
        else:
            self.audio_conf = audio_conf

        if not image_conf:
            self.image_conf = {}
        else:
            self.image_conf = image_conf

        crop_size = self.image_conf.get('crop_size', 224)
        center_crop = self.image_conf.get('center_crop', False)

        if center_crop:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        else:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.RandomResizedCrop(crop_size), transforms.ToTensor()])

        RGB_mean = self.image_conf.get('RGB_mean', [0.485, 0.456, 0.406])
        RGB_std = self.image_conf.get('RGB_std', [0.229, 0.224, 0.225])
        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

        self.windows = {'hamming': scipy.signal.hamming,
        'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

    def _LoadAudio(self, path):
        audio_type = self.audio_conf.get('audio_type', 'melspectrogram')
        if audio_type not in ['melspectrogram', 'spectrogram']:
            raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
        preemph_coef = self.audio_conf.get('preemph_coef', 0.97)
        sample_rate = self.audio_conf.get('sample_rate', 16000)
        window_size = self.audio_conf.get('window_size', 0.025)
        window_stride = self.audio_conf.get('window_stride', 0.01)
        window_type = self.audio_conf.get('window_type', 'hamming')
        num_mel_bins = self.audio_conf.get('num_mel_bins', 40)
        target_length = self.audio_conf.get('target_length', 1024)
        use_raw_length = self.audio_conf.get('use_raw_length', False)
        padval = self.audio_conf.get('padval', 0)
        fmin = self.audio_conf.get('fmin', 20)
        n_fft = self.audio_conf.get('n_fft', int(sample_rate * window_size))
        win_length = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)

        # load audio, subtract DC, preemphasis
        y, sr = librosa.load(path, sample_rate)
        if y.size == 0:
            y = np.zeros(200)
        y = y - y.mean()
        y = preemphasis(y, preemph_coef)
        # compute mel spectrogram
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length,
            window=self.windows.get(window_type, self.windows['hamming']))
        spec = np.abs(stft)**2
        if audio_type == 'melspectrogram':
            mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
            melspec = np.dot(mel_basis, spec)
            logspec = librosa.power_to_db(melspec, ref=np.max)
        elif audio_type == 'spectrogram':
            logspec = librosa.power_to_db(spec, ref=np.max)
        n_frames = logspec.shape[1]
        if use_raw_length:
            target_length = n_frames
        p = target_length - n_frames
        if p > 0:
            logspec = np.pad(logspec, ((0,0),(0,p)), 'constant',
                constant_values=(padval,padval))
        elif p < 0:
            logspec = logspec[:,0:p]
            n_frames = target_length
        logspec = torch.FloatTensor(logspec)
        return logspec, n_frames

    def _LoadImage(self, impath):
        img = Image.open(impath).convert('RGB')
        img = self.image_resize_and_crop(img)
        img = self.image_normalize(img)
        return img

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        datum = self.data[index]
        wavpath = os.path.join(self.audio_base_path, datum['wav'])
        imgpath = os.path.join(self.image_base_path, datum['image'])
        audio, nframes = self._LoadAudio(wavpath)
        image = self._LoadImage(imgpath)
        return image, audio, nframes

    def __len__(self):
        return len(self.data)

def get_english_speaker(entry):
    ID = entry.strip().split("/")[-1]
    return ID.split("+")[0].strip("_")[1].strip("-")[0]

def get_hindi_speaker(entry):
    ID = entry.strip().split("/")[-1]
    return ID.split("+")[1].strip("_")[1].strip("-")[0]

class ImageAudioDataWithCPC(Dataset):
    def __init__(self, image_base_path, dataset_json_file):

        with open(dataset_json_file, 'r') as fp:
            data = json.load(fp)
        self.image_base_path = Path(image_base_path).absolute()

        print(f'\n\rRead in {len(data)} data points')
        
        self.audio_conf = {
            "audio_type": "melspectrogram",
            "preemph_coef": 0.97,
            "sample_rate": 16000,
            "window_size": 0.025,
            "window_stride": 0.01,
            "window_type": "hamming",
            "num_mel_bins": 40,
            "target_length": 1024,
            "use_raw_length": False,
            "padval": 0,
            "fmin": 20
        }
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.padval = self.audio_conf.get('padval', 0)
        self.image_conf = {
            "crop_size": 224,
            "center_crop": False,
            "RGB_mean": [0.485, 0.456, 0.406],
            "RGB_std": [0.229, 0.224, 0.225]
        }
        crop_size = self.image_conf.get('crop_size')
        center_crop = self.image_conf.get('center_crop')
        RGB_mean = self.image_conf.get('RGB_mean')
        RGB_std = self.image_conf.get('RGB_std')

        if center_crop:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        else:
            self.image_resize_and_crop = transforms.Compose(
                [transforms.RandomResizedCrop(crop_size), transforms.ToTensor()])

        self.image_normalize = transforms.Normalize(mean=RGB_mean, std=RGB_std)

        self.n_sample_frames = 128 + 12
        self.n_utterances_per_speaker = 8

        self.english_speakers = []
        self.hindi_speakers = []

        for entry in data:
            english_speaker = get_english_speaker(entry)
            hindi_speaker = get_hindi_speaker(entry)
            self.english_speakers.append(english_speaker)
            self.hindi_speakers.append(hindi_speaker)
        self.english_speakers = sorted(self.english_speakers)
        self.hindi_speakers = sorted(self.hindi_speakers)

        metadata_by_speaker = dict()
        for entry in data:
            english_speaker = get_english_speaker(entry)
            hindi_speaker = get_hindi_speaker(entry)
            metadata_by_speaker.setdefault(english_speaker, []).append(entry)
            metadata_by_speaker.setdefault(hindi_speaker, []).append(entry)

        self.metadata = []
        for entry in data:
            english_speaker = get_english_speaker(entry)
            hindi_speaker = get_hindi_speaker(entry)
            if len(metadata_by_speaker[english_speaker]) >= self.n_utterances_per_speaker and len(metadata_by_speaker[hindi_speaker]) >= self.n_utterances_per_speaker:
                self.metadata.append((entry, metadata_by_speaker[english_speaker], metadata_by_speaker[hindi_speaker]))

    def _LoadImage(self, impath):
        img = Image.open(impath).convert('RGB')
        img = self.image_resize_and_crop(img)
        img = self.image_normalize(img)
        return img

    def _PadFeat(self, feat):
        nframes = feat.shape[1]
        pad = self.target_length - nframes
        
        if pad > 0:
            feat = np.pad(feat, ((0, 0), (0, pad)), 'constant',
                constant_values=(self.padval, self.padval))
        elif pad < 0:
            nframes = self.target_length
            feat = feat[:, 0: pad]

        return feat, nframes

    def _sample(self, pos_feat, pos_frames, pos_path, paths, key):

        mels = list()
        
        mel, nframes = self._PadFeat(pos_feat)
        pos = random.randint(0, pos_frames - self.n_sample_frames)
        mel = mel[:, pos:pos + self.n_sample_frames]
        mels.append(mel)
        
        temp = paths.copy()
        temp.remove(pos_path)
        
        paths = random.sample(paths, self.n_utterances_per_speaker-1)
        for path in temp:
            data_point = np.load(data_path + ".npz")
            mel, nframes = self._PadFeat(data_point[key])
            pos = random.randint(0, nframes - self.n_sample_frames)
            mel = mel[:, pos:pos + self.n_sample_frames]
            mels.append(mel)
        mels = np.stack(mels)

    def __getitem__(self, index):

        data_path, english_points, hindi_points = self.metadata[index]

        data_point = np.load(data_path + ".npz")

        english_audio_feat = data_point["eng_audio_feat"]
        english_audio_feat, english_nframes = self._PadFeat(english_audio_feat)
        english_cpc = self._sample(english_audio_feat, english_nframes, data_path, english_points, "eng_audio_feat")

        hindi_audio_feat = data_point["hindi_audio_feat"]
        hindi_audio_feat, hindi_nframes = self._PadFeat(hindi_audio_feat)
        hindi_cpc = self._sample(hindi_audio_feat, hindi_nframes, data_path, hindi_points, "hindi_audio_feat")
        
        # imgpath = self.image_base_path / str(data_point['image'])
        # image = self._LoadImage(imgpath)
        # return image, english_audio_feat, english_nframes, hindi_audio_feat, hindi_nframes, 
        return english_cpc, hindi_cpc

    def __len__(self):
        return len(self.metadata)

class AudioDataZeroSpeech2020(Dataset):
    def __init__(self, root, args):

        self.root = root

        self.data = list(self.root.rglob("*.wav"))

        print(f'\n\rRead in {len(self.data)} data points')
        
        self.audio_conf = args["audio_config"]
        self.target_length = self.audio_conf.get('target_length', 1024)
        self.padval = self.audio_conf.get('padval', 0)
    
    def _LoadAudio(self, path):

        audio_type = self.audio_conf.get('audio_type')
        if audio_type not in ['melspectrogram', 'spectrogram']:
            raise ValueError('Invalid audio_type specified in audio_conf. Must be one of [melspectrogram, spectrogram]')
        
        preemph_coef = self.audio_conf.get('preemph_coef')
        sample_rate = self.audio_conf.get('sample_rate')
        window_size = self.audio_conf.get('window_size')
        window_stride = self.audio_conf.get('window_stride')
        window_type = self.audio_conf.get('window_type')
        num_mel_bins = self.audio_conf.get('num_mel_bins')
        target_length = self.audio_conf.get('target_length')
        fmin = self.audio_conf.get('fmin')
        n_fft = self.audio_conf.get('n_fft', int(sample_rate * window_size))
        win_length = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)

        # load audio, subtract DC, preemphasis
        y, sr = librosa.load(path, sample_rate)
        if y.size == 0:
            y = np.zeros(target_length)
        y = y - y.mean()
        y = preemphasis(y, preemph_coef)

        # compute mel spectrogram / filterbanks
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=scipy_windows.get(window_type, scipy_windows['hamming']))
        spec = np.abs(stft)**2 # Power spectrum
        if audio_type == 'melspectrogram':
            mel_basis = librosa.filters.mel(sr, n_fft, n_mels=num_mel_bins, fmin=fmin)
            melspec = np.dot(mel_basis, spec)
            logspec = librosa.power_to_db(melspec, ref=np.max)
        elif audio_type == 'spectrogram':
            logspec = librosa.power_to_db(spec, ref=np.max)
        # n_frames = logspec.shape[1]
        logspec = torch.FloatTensor(logspec)
        return torch.tensor(logspec)#, n_frames

    def _PadFeat(self, feat):
        nframes = feat.shape[1]
        pad = self.target_length - nframes
        
        if pad > 0:
            feat = np.pad(feat, ((0, 0), (0, pad)), 'constant',
                constant_values=(self.padval, self.padval))
        elif pad < 0:
            nframes = self.target_length
            feat = feat[:, 0: pad]

        return feat, nframes

    def __getitem__(self, index):

        data_path = self.data[index].with_suffix(".wav")

        audio_feat = self._LoadAudio(data_path)
        # audio_feat, nframes = self._PadFeat(audio_feat)

        return audio_feat, str(self.data[index])

    def __len__(self):
        return len(self.data)