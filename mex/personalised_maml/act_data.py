import os.path
import numpy as np
import csv


class ACTData:

    def __init__(self, root, batchsz, n_way, k_shot, k_query, test_index):
        # if data.npy exists, just load it.
        self.samples_per_class = 10

        self.x = self.load(root)
        print(self.x.shape)

        # [5, 7, n, 80, 16, 1]
        # TODO: can not shuffle here, we must keep training and test set distinct!
        # last user is the test task
        self.x_train = []
        self.x_test = []
        for i in range(30):
            if i == test_index:
                self.x_test = self.x[i:i+1]
            else:
                self.x_train.extend(self.x[i:i+1])

        self.x_test = np.array(self.x_test)
        self.x_train = np.array(self.x_train)

        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = self.x.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <= self.samples_per_class

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"], "test")}

    def load_data_cache(self, data_pack, mode="train"):
        """
        Collects several batches data for N-shot learning
        :param mode:
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        # querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            min_qry = 10000
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_person = np.random.choice(data_pack.shape[0], 1, False)[0]
                # selected_cls = np.random.choice(data_pack[selected_person].keys(), self.n_way, False)

                for cur_class in range(self.n_way):

                    x_spt.extend(data_pack[selected_person][cur_class][:self.k_shot])
                    if mode == "train":
                        qry = data_pack[selected_person][cur_class][self.k_shot:self.k_shot+self.k_query]
                    else:
                        qry = data_pack[selected_person][cur_class][self.k_shot:self.k_shot+1]
                    x_qry.extend(qry)
                    y_spt.extend([cur_class for _ in range(self.k_shot)])
                    y_qry.extend([cur_class for _ in range(len(qry))])

                # shuffle inside a batch
                # perm = np.random.permutation(self.n_way * self.k_shot)
                # x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 5*180)[perm]
                # y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                # perm = np.random.permutation(len(x_qry))
                # x_qry = np.array(x_qry).reshape(len(perm), 5*180)[perm]
                # y_qry = np.array(y_qry).reshape(len(perm))[perm]
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 5*180)
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)
                x_qry = np.array(x_qry).reshape(len(x_qry), 5*180)
                y_qry = np.array(y_qry).reshape(len(x_qry))

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)
                if min_qry > len(x_qry):
                    min_qry = len(x_qry)

            x_qrys_ = []
            y_qrys_ = []
            for inq, item in enumerate(x_qrys):
                indices = np.random.choice(len(x_qrys[inq]), min_qry, False)
                x_qrys_.append([f for ind, f in enumerate(x_qrys[inq]) if ind in indices])
                y_qrys_.append([f for ind, f in enumerate(y_qrys[inq]) if ind in indices])

            # [b, setsz, 1, 80, 16]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 5*180)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 80, 16]
            x_qrys_ = np.array(x_qrys_).astype(np.float32).reshape(self.batchsz, min_qry, 5*180)
            y_qrys_ = np.array(y_qrys_).astype(np.int).reshape(self.batchsz, min_qry)

            data_cache.append([x_spts, y_spts, x_qrys_, y_qrys_])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode], mode)

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch

    def load(self, path):
        alldata = []
        subjects = [f for f in os.listdir(path) if f != '.DS_Store']
        for subject in subjects:
            activity_items = []
            subject_path = os.path.join(path, subject)
            activities = [d for d in os.listdir(subject_path) if d != '.DS_Store']
            for activity in activities:
                items = []
                activity_path = os.path.join(subject_path, activity)
                samples = os.listdir(activity_path)
                for item in samples:
                    _data = csv.reader(open(os.path.join(activity_path, item), "r"), delimiter=",")
                    for row in _data:
                        item = [float(f) for f in row]
                        item = np.array(item)
                        item = np.reshape(item, (5*180))
                        items.append(item)
                activity_items.append(items)
            alldata.append(activity_items)
        return np.array(alldata)
