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
                f"{name:20} | {str(module.weight.size()):35} | => Quantize to {quan_range} indices"
            )
            # Get non-zero weights
            weight_flat = weight.reshape(-1)
            non_zero_mask = weight_flat != 0
            non_zero_weights = weight_flat[non_zero_mask]

            if len(non_zero_weights) > 0:
                # Use KMeans to cluster non-zero weights
                space = np.linspace(
                    min(non_zero_weights), max(non_zero_weights), num=quan_range
                )
                kmeans = KMeans(
                    n_clusters=len(space),
                    init=space.reshape(-1, 1),
                    n_init=1,
                    algorithm="lloyd",
                )
                kmeans.fit(non_zero_weights.reshape(-1, 1))

                # Replace weights with cluster centroids
                new_weights = weight_flat.copy()
                new_weights[non_zero_mask] = kmeans.cluster_centers_[
                    kmeans.labels_
                ].reshape(-1)

                # Reshape back to original shape and update module weights
                module.weight.data = torch.from_numpy(new_weights.reshape(shape)).to(
                    dev
                )
            else:
                # If all weights are zero, just keep them as is
                module.weight.data = torch.from_numpy(weight).to(dev)
