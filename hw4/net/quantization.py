import numpy as np
import torch
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.cluster import KMeans


def apply_weight_sharing(model, bits=5):
    """
    Applies weight sharing to the given model
    """
    for name, module in model.named_children():
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        shape = weight.shape
        quan_range = 2**bits
        if len(shape) == 2:  # Fully connected layers
            print(
                f"{name:20} | {str(module.weight.size()):35} | => Quantize to {quan_range} indices"
            )
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)

            # Weight sharing by kmeans
            space = np.linspace(min(mat.data), max(mat.data), num=quan_range)
            kmeans = KMeans(
                n_clusters=len(space),
                init=space.reshape(-1, 1),
                n_init=1,
                algorithm="lloyd",
            )
            kmeans.fit(mat.data.reshape(-1, 1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            mat.data = new_weight

            # Insert to model
            module.weight.data = torch.from_numpy(mat.toarray()).to(dev)
        elif len(shape) == 4:  # Convolution layers
            #################################
            # TODO:
            #    Suppose the weights of a certain convolution layer are called "W"
            #       1. Get the unpruned (non-zero) weights, "non-zero-W",  from "W"
            #       2. Use KMeans algorithm to cluster "non-zero-W" to (2 ** bits) categories
            #       3. For weights belonging to a certain category, replace their weights with the centroid
            #          value of that category
            #       4. Save the replaced weights in "module.weight.data", and need to make sure their indices
            #          are consistent with the original
            #   Finally, the weights of a certain convolution layer will only be composed of (2 ** bits) float numbers
            #   and zero
            #   --------------------------------------------------------
            #   In addition, there is no need to return in this function ("model" can be considered as call by
            #   reference)
            #################################

            print(
                f"{name:20} | {str(module.weight.size()):35} | ** NEED TO BE IMPLEMENTED **"
            )
            pass
