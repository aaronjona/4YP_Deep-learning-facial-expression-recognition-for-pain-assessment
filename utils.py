import os
import math
import random

from PIL import Image
import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

def get_transforms():
	return transforms.Compose([
        # Random horizontal flip
        transforms.RandomHorizontalFlip(),
        
        # Random rotation within [-25, 25] degrees
        transforms.RandomRotation(degrees=(-10, 10)),
        
        # Random affine transformation
        # For height and width shifts, we use the translate argument,
        # which specifies the maximum absolute fraction of height and width for horizontal and vertical translations, respectively.
        # For shear, we use degrees. For zoom, we scale between 1-0.1 (90%) and 1+0.1 (110%).
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        
        # Convert image to tensor
        transforms.ToTensor(),
        
        # Additional normalization or transformations can be added here if necessary
    ])

class DataGenerator(Dataset): 
	def __init__(self, paths, transform=None, standard_size = (25, 256, 256), training = False, n_classes = 5): 
		self.paths = paths
		self.standard_size = standard_size
		self.training = training
		self.transform = transform
		self.n_classes = n_classes
		if self.n_classes == 5:
			self.label_map = {'BL1': 0, 'PA1': 1, 'PA2': 2, 'PA3': 3, 'PA4': 4}
		else: 
			self.label_map = {'BL1':0, 'PA4':1}
		self.file_class_pair = self.get_file_class_pair()

		if self.training: 
			random.shuffle(self.file_class_pair)

	@staticmethod
	def get_class(path): 
		return path.split('-')[1]

	@staticmethod
	def read_file(path):
		try: 
			tensor = np.load(path, mmap_mode='r')
			return tensor
		except FileNotFoundError:
			print(f"File not found: {path}")
			return None
		except IOError:
			print(f"IOError occurred when reading: {path}")
			return None
		except ValueError:
			print(f"ValueError: Cannot reshape array of incorrect size in {path}")
			return None
		except Exception as e:
			print(f"An unexpected error occurred: {e}=================================")
			return None

	def get_file_class_pair(self):
		file_class_pair = []
		for path in self.paths: 
			label_name = self.get_class(path)
			file_class_pair.append((path, self.label_map[label_name]))
		return file_class_pair


	def __len__(self):
		return len(self.file_class_pair)

	def __getitem__(self, index):
		path, label = self.file_class_pair[index]
		tensor = self.read_file(path)  # Assuming this reads in a NumPy array
		if np.isnan(tensor).any():
			print(f"Warning: NaN values found in data at index {index}")
		tensor = np.nan_to_num(tensor)

		if tensor is not None and tensor.shape == self.standard_size:
			# Assuming tensor is normalized [0, 1], scale to [0, 255] and convert to uint8
			
			if self.transform is not None:
				tensor = (tensor * 255).astype(np.uint8)
				# Assuming tensor shape is (T, H, W), where T is temporal dimension
				transformed_frames = []
				for i in range(tensor.shape[0]):  # Iterate over time dimension
					frame = tensor[i]
					frame = Image.fromarray(frame)  # Convert to PIL Image
					frame = self.transform(frame)  # Apply transformation
					transformed_frames.append(frame)
				tensor = torch.stack(transformed_frames).squeeze()  # Stacks along a new dimension
			else: 
				tensor = torch.tensor(tensor, dtype=torch.float32)
				
			label = torch.tensor(label, dtype=torch.long)
			return tensor.unsqueeze(0), label
		else:
			print(f"Data Problem!!!! In __getitem__!!! {tensor.shape}")
			return None, None

class AverageMeter(object):
	def __init__(self):
		self.reset()
		
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
		
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
		
def calculate_accuracy(outputs, labels, n_classes):
	with torch.no_grad():		
		if n_classes == 5:
			_, predicted = torch.max(outputs, 1)
			correct_pred = (predicted == labels).sum().item()
			return correct_pred / labels.size(0)
		else: 
			predicted = (torch.sigmoid(outputs) >= 0.5).long()
			correct_pred = (predicted == labels).sum().item()
			return correct_pred / labels.size(0)

	

def load_predefined_paths(all_per_paths, n_classes=5):
	# removed_paths = set(['082315_w_60', '082414_m_64', '082909_m_47', '083009_w_42', '083013_w_47', '083109_m_60', '083114_w_55', '091914_m_46', '092009_m_54', 
	# 			  '092014_m_56', '092509_w_51', '092714_m_64', '100514_w_51', '100914_m_39', '101114_w_37', '101209_w_61', '101809_m_59', '101916_m_40', 
	# 			  '111313_m_64', '120614_w_61'])
	validation_pers = ['100914_m_39', '101114_w_37', '082315_w_60', '083114_w_55', '083109_m_60', '072514_m_27', '080309_m_29', '112016_m_25', '112310_m_20',
					'092813_w_24', '112809_w_23', '112909_w_20', '071313_m_41', '101309_m_48', '101609_m_36', '091809_w_43', '102214_w_36', '102316_w_50',
					'112009_w_43', '101814_m_58', '101908_m_61', '102309_m_61', '112209_m_51', '112610_w_60', '112914_w_51', '120514_w_56']
	
	training_paths = [path for path in all_per_paths if path.split('/')[-1] not in validation_pers]
	validation_paths = [path for path in all_per_paths if path.split('/')[-1] in validation_pers]
	# assert len(training_paths) == 61, 
	print(f'Length of training_paths: {len(training_paths)}; Length of validation_paths: {len(validation_paths)}')

	if n_classes == 5:
		labels = ['BL1', 'PA1', 'PA2', 'PA3', 'PA4']
	else: 
		labels = ['BL1', 'PA4']

	training_set = []
	for per in training_paths:
		per_set = glob.glob(os.path.join(per, '*.npy'))
		correct_set = [path for path in per_set if path.split('-')[1] in labels]
		training_set.extend(correct_set)

	valid_set = []
	for per in validation_paths:
		per_set = glob.glob(os.path.join(per, '*.npy'))
		correct_set = [path for path in per_set if path.split('-')[1] in labels]
		valid_set.extend(correct_set)

	return training_set, valid_set


def load_paths(paths):
	train_per, valid_per = train_test_split(paths, test_size = 0.3, shuffle = True, random_state = 1)
	training_set = []
	for per in train_per:
		per_set = glob.glob(os.path.join(per, '*.npy'))
		training_set.extend(per_set)

	valid_set = []
	for per in valid_per:
		per_set = glob.glob(os.path.join(per, '*.npy'))
		valid_set.extend(per_set)


	return training_set, valid_set

def load_data(paths, batch_size=8, method = 'Random', augmentation = None, n_classes=5):

	# training_set, valid_set, test_set = load_paths(paths)
	if method == 'Random':
		training_set, valid_set = load_paths(paths)
		print('Load Random Train Validation Split!!!')
	else:
		training_set, valid_set = load_predefined_paths(paths, n_classes)
		print('Load Predefined Split!!!')
	print(f'{n_classes} Classes Data Loaded =====> Data Loaded: Training Set: {len(training_set)}; Validation Set: {len(valid_set)}')

	if augmentation is not None:
		transform = get_transforms()
	else: 
		transform = None

	train_data = DataGenerator(paths=training_set, transform=transform, standard_size = (25, 256, 256), training=True, n_classes=n_classes)
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

	val_data = DataGenerator(paths=valid_set, n_classes=n_classes)
	val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)

	# test_data = DataGenerator(paths=test_set)
	# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

	return train_loader, val_loader

def load_eval_data(paths, batch_size = 1, method = 'Random'):
	if method == 'Random':
		training_set, valid_set = load_paths(paths)
	else:
		training_set, valid_set = load_predefined_paths(paths)

	eval_data = DataGenerator(paths=training_set, transform=None, standard_size = (25, 256, 256), training=True)
	eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=True, num_workers=2)

	return eval_loader


