import logging
import numpy as np

logger = logging.getLogger()


def assign_examples_to_clients(y_train, n_parties, partition):
    """

    :param y_train: labels in the training set (distributed among clients)
    :param n_parties: number of clients
    :param partition: type of partition
    :return: dict with client indexes as keys and lists of indexes of assigned examples as values
    """
    # Homogeneous (although not class-balanced) distribution by default -- copies and pasted from utils,partition_data
    logger.info(f"Running custom partitioning: {partition}")
    n_train = y_train.shape[0]
    quantity_dist, label_dist = partition.split("-")
    min_size, min_require_size = 0, 10
    pos_idxs = np.argwhere(y_train == 1).squeeze()
    neg_idxs = np.argwhere(y_train == 0).squeeze()
    idxs = np.random.permutation(n_train)
    net_dataidx_map = {}
    if quantity_dist == "avg":
        while min_size < min_require_size:
            proportions = np.random.beta(100, 1, size=n_parties)
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * len(idxs))
    elif quantity_dist == "big":
        cnt = 0
        while min_size < 10 or cnt < n_parties // 2 + 2:
            proportions = np.random.beta(1, 4, size=n_parties)
            proportions = proportions / proportions.sum()
            cnt = (proportions * len(idxs) > 1.1 * n_train / n_parties).sum()
            # print(cnt)
            min_size = np.min(proportions * len(idxs))
    elif quantity_dist == "small":
        cnt = 0
        while min_size < 10 or cnt < n_parties // 2 + 2:
            proportions = np.random.beta(0.2, 0.9, size=n_parties)
            proportions = proportions / proportions.sum()
            cnt = (proportions * len(idxs) < 0.9 * n_train / n_parties).sum()
            # print(cnt)
            min_size = np.min(proportions * len(idxs))
    if label_dist == "equal":
        pos_proportions = (np.cumsum(proportions) * len(pos_idxs)).astype(int)[:-1]
        neg_proportions = (np.cumsum(proportions) * len(neg_idxs)).astype(int)[:-1]
        print(pos_proportions)
        print(neg_proportions)
        print(np.diff(pos_proportions))
        print(np.diff(neg_proportions))
        pos_batch_idxs = np.split(pos_idxs, pos_proportions)
        neg_batch_idxs = np.split(neg_idxs, neg_proportions)
        net_dataidx_map = {
            i: np.concatenate((pos_batch_idxs[i], neg_batch_idxs[i]))
            for i in range(n_parties)
        }
    elif label_dist == "random":
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        print(proportions)
        print(np.diff(proportions))
        print("random")
        np.random.shuffle(idxs)
        batch_idxs = np.split(idxs, proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    return net_dataidx_map
