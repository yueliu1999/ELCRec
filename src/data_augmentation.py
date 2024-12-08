import copy
import random


class Random(object):
    def __init__(self, tao=0.2, gamma=0.7, beta=0.2):
        self.data_augmentation_methods = [Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=beta)]
        print("Total augmentation numbers: ", len(self.data_augmentation_methods))

    def __call__(self, sequence):
        augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
        augment_method = self.data_augmentation_methods[augment_method_idx]
        return augment_method(sequence)


class Crop(object):
    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence):
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.tao * len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        if sub_seq_length < 1:
            return [copied_sequence[start_index]]
        else:
            cropped_seq = copied_sequence[start_index : start_index + sub_seq_length]
            return cropped_seq


class Mask(object):
    def __init__(self, gamma=0.7):
        self.gamma = gamma

    def __call__(self, sequence):
        copied_sequence = copy.deepcopy(sequence)
        mask_nums = int(self.gamma * len(copied_sequence))
        mask = [0 for i in range(mask_nums)]
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence


class Reorder(object):
    def __init__(self, beta=0.2):
        self.beta = beta

    def __call__(self, sequence):
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.beta * len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        sub_seq = copied_sequence[start_index : start_index + sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + copied_sequence[start_index + sub_seq_length :]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq

