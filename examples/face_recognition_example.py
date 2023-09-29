from typing import List
from autoannotator.types.faces import Face
from autoannotator.utils.image_reader import ImageReader
from autoannotator.detection.faces import SCRFD, YOLOv7, FaceDetEnsemble
from autoannotator.utils.image_alignment import ImageAlignmentSimilarityTransform
from autoannotator.feature_extraction.faces import FaceFeatureExtractorAdaface, FaceFeatureExtractorInsightface
from autoannotator.feature_extraction.faces import FaceFeatureExtractionEnsemle
from autoannotator.clustering import ClusteringDBSCAN


def auto_annotate_faces(img_files: List[str]) -> List[Face]:
    """
    Face recognition usage example

    Arguments:
        img_files: List of image paths to recognize

    Returns:
        (List[Face]) - List of recognized faces
    """
    # ini image file reader
    reader = ImageReader()

    # init face detector ensemble
    models = [SCRFD(), YOLOv7()]
    fd_ensemble = FaceDetEnsemble(models=models)

    # init alignment regressor
    regressor = ImageAlignmentSimilarityTransform()

    # init feature extractor ensemble
    models = [FaceFeatureExtractorAdaface(), FaceFeatureExtractorInsightface()]
    fr_ensemble = FaceFeatureExtractionEnsemle(models=models)

    # start processing images
    faces_arr = []
    descriptors = []
    for img_ind, img_file in enumerate(img_files):
        # read image
        img = reader(img_file)

        # detect faces with ensemble
        faces = fd_ensemble(img)

        # for each detected face run alignment and face descriptor extractor
        for face in faces:
            # get aligned image
            aligned_img = regressor(img, face.landmarks)

            # extract descriptor with the ensemble of descriptor extractors
            descriptor = fr_ensemble(aligned_img)

            # save face descriptors for further clusterization
            descriptors.append(descriptor)

            # save faces
            faces_arr.append(face)

    # init clusterization method
    dbscan = ClusteringDBSCAN(type="sklearn", eps=0.01, min_samples=2)

    # clusterize descriptors
    labels = dbscan(descriptors)

    # set cls_id for each face
    results = []
    for face, label in zip(faces_arr, labels):
        face.cls_id = label
        results.append(face)
    return results
