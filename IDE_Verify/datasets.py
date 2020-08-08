from torchvision import datasets
import os
import numpy as np
from torchvision.datasets.folder import default_loader


class SiameseDataset(datasets.ImageFolder):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, root, transform):
        super(SiameseDataset, self).__init__(root, transform)
        self.labels = np.array(self.imgs)[:, 1]
        self.data = np.array(self.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        class_name = []
        for s in self.samples:
            filename = os.path.basename(s[0])
            class_name.append(filename.split('_')[0])
        self.class_name = np.asarray(class_name)
        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)

    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1

    def __getitem__(self, index):
        siamese_target = np.random.randint(0, 2)
        # 1 means same ID, 0 means different IDs
        img1, label1 = self.data[index], self.labels[index].item()
        if siamese_target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            different_label = np.random.choice(list(self.labels_set - set([label1])))
            siamese_index = np.random.choice(self.label_to_indices[different_label])
        img2, label2 = self.data[siamese_index], self.labels[siamese_index].item()
        img1 = default_loader(img1)
        img2 = default_loader(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        # return (img1, img2), siamese_target
        return (img1, img2), siamese_target, (int(label1), int(label2))

    def __len__(self):
        return len(self.imgs)
