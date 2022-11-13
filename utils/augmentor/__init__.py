from .image_augmentor import FlowAugmentor as ImageFlowAugmentor
from .image_augmentor import SparseFlowAugmentor as ImageSparseFlowAugmentor
from .event_augmentor import EventFlowAugmentor as EventFlowAugmentor
from .event_augmentor import SparseEventFlowAugmentor as EventSparseFlowAugmentor


def fetch_augmentor(is_event=True, is_sparse=False, aug_params=None):
    if is_event:
        if is_sparse:
            return EventSparseFlowAugmentor(**aug_params)
        else:
            return EventFlowAugmentor(**aug_params)
    else:
        if is_sparse:
            return ImageSparseFlowAugmentor(**aug_params)
        else:
            return ImageFlowAugmentor(**aug_params)
