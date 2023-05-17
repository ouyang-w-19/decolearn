import warnings
import pickle
import umap
import numpy as np
from bl.bl_base import BLBase
from bl.nn import NNFunc
from typing import List, Union


class UMAPBL(BLBase):
    def __init__(self, n_components: int, min_dist: np.float, local_connectivity: np.float,
                 metric: str, target_metric: str, n_neighbors: int, lr_umap: np.float, id: str,
                 classifier_config=None):
        """
        - Wrapper class for UMAP initialization, training, generating transformation map and generating
        - Can be given an classifier to test classification accuracy on UMAP transformation
          transformed pairs
        :param n_components: Dimensionality of latent space
        :param min_dist: Provides the minimum distance apart that points are allowed to be in the low
                         dimensional representation. Low values can be used if interested in clustering.
        :param local_connectivity:
        :param metric: Measurement of distance in feature space
        :param target_metric: Measurement of distance in latent space
        :param n_neighbors: Low values forces umap to focus on local structures, large values push UMAP to
                            look at large neighborhoods of each point when estimating the manifold
        :param lr_umap: Learning rate for embedding optimization
        :param id: Unique model identifier
        :param classifier_config: Optional classifier configuration given as tuple
        """
        self.n_components = n_components
        self.min_dist = min_dist
        self.local_connectivity = local_connectivity
        self.metric = metric
        self.target_metric = target_metric
        self.n_neighbors = n_neighbors
        self.extractor = None
        self.classifier_config = classifier_config
        self.classifier = None

        # Transformation is also stored in this class
        self.transformation = None
        super(UMAPBL, self).__init__(None, lr_umap, None, id)

    @staticmethod
    def flatten_features(x_train: np.ndarray) -> np.ndarray:
        """
        - Flattens the feature map for umap procession
        :param x_train:
        """
        x_train_shape = x_train.shape
        # Note that x_train can be the feature tensor of only ONE source, typical for T
        if len(x_train_shape) > 2:
            feature_dims = x_train_shape[1:]
            x_flat = np.reshape(x_train, (x_train_shape[0], np.prod(feature_dims)))
        else:
            x_flat = x_train

        return x_flat

    def build_extractor(self) -> umap.UMAP:
        """
        - Is called in self.build_model() to build extractor
        :return: Returns an unfitted UMAP transformer instance and assings self.model to it
        """
        extractor = umap.UMAP(n_components=self.n_components,
                              min_dist=self.min_dist,
                              local_connectivity=self.local_connectivity,
                              metric=self.metric,
                              target_metric=self.target_metric,
                              n_neighbors=self.n_neighbors,
                              learning_rate=self._lr
                              )
        return extractor

    def build_model(self):
        """
        - Assumption: Classifier is optional, must be given the correct input dimensions
        """
        self.extractor = self.build_extractor()
        if self.classifier_config:
            identifier = f'UMAP_Classifier_{self._id}'
            inp, layers, lr_mlp, loss, act, dropout_rates, metric_mlp, _, _ = self.classifier_config
            self.classifier = NNFunc(inp=inp, layers=layers, lr=lr_mlp, loss=loss, act=act,
                                     dropout_rates=dropout_rates, metric=metric_mlp, id=identifier)
            model = (self.extractor, self.classifier)
        else:
            model = self.extractor

        return model

    def train(self, x, y, batch_size=None, epochs=None, callbacks=None):
        """
        - Activates the UMAP supervised transformation learning, not the BL-classifier learning
        :param x: Feature (train) data
        :param y: True (train) labels
        :param batch_size: Only for interface standardization
        :param epochs: Only for interface standardization
        :param callbacks: Only for interface standardization
        """
        x_flatten = self.flatten_features(x)
        self.extractor.fit(x_flatten, y)

    def train_classifier(self, trans, y, batch_size=50, epochs=5, callbacks=None):
        """
        - Wrapper method to fit the classifier on umap transformation
        :param trans: UMAP transition
        :param y: True labels
        :param batch_size: Mini batch size
        :param epochs: NUmber of iterations in which models are fitted on entire dataset
        :param callbacks: PLaceholder for callback objects (for logging, lr-schedule...etc.)
        """
        if self.classifier:
            self.classifier.train(trans, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        else:
            warnings.warn('Can not train UMAP-classifier. UMAP-Classifier not configured.')
            return 1

    def get_prediction(self, x, batch_size=None, binary=None):
        """
        - Get the prediction of the associated classifier, i.e. link inference of both submodels
        - WARNING: An compatible input of the classifier must be adjusted prior to association with
          this class
        - If no classifier is associated with the instance, then a zero array will be returned
        :param x: Data to be inferred
        :param batch_size: Batch size for prediction
        :param binary: Deprecated
        :return: Prediction of the classifier based on umap transformation
        """
        if self.classifier:
            x_flat = self.flatten_features(x)
            trans = self.get_transformation(x_flat)
            pred = self.classifier.get_prediction(trans, batch_size=batch_size, binary=binary)
            return pred
        else:
            warnings.warn('No classifier is associated with the instance of UMAPBL. Abort operation!')
            return None

    def get_prediction_classifier(self, trans):
        if self.classifier:
            return self.classifier.get_prediction(trans)
        else:
            warnings.warn('No classifier is associated with the instance of UMAPBL. Abort operation')
            return None

    def get_transformation(self, x, remember=False, **kwargs):
        x_flat = self.flatten_features(x)
        transformation = self.extractor.transform(x_flat)
        if remember:
            self.transformation = transformation

        return transformation

    def save_model(self, path_file):
        obj_byte = pickle.dumps(self.model)
        obj_byte = np.asarray(obj_byte)
        np.save(path_file, obj_byte)

    def _retrieve_model_info(self):
        s = f'n_components:         {self.n_components}\n' \
            f'min_dist:             {self.min_dist}\n' \
            f'local_connectivity:   {self.local_connectivity}\n' \
            f'metric:               {self.metric}\n' \
            f'target_metric:        {self.target_metric}\n' \
            f'n_neighbors:          {self.n_neighbors}\n' \
            f'learning_rate:        {self._lr}\n' \

        return s

