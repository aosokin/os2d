from PIL import Image
from torch import Tensor


class FeatureMapSize(object):
    """
    This class represents the spatial dimensions of feature maps and images.
    This class is used to avoid W,H vs H,W format confusion.
    This class is immutable.

    For PIL.Image.Image, FeatureMapSize is w, h
    For torch.tensor, FeatureMapSize is size(-1), size(-2)
    """
    w = None
    h = None
    def __init__(self, img=None, w=None, h=None):
        if w is not None and h is not None:
            pass
        elif isinstance(img, Image.Image):
            w, h = img.size
        elif isinstance(img, Tensor):
            w = img.size(-1)
            h = img.size(-2)
        else:
            raise RuntimeError("Cannot initialize FeatureMapSize")
        super(FeatureMapSize, self).__setattr__("w", w)
        super(FeatureMapSize, self).__setattr__("h", h)

    def __setattr__(self, *args):
        raise AttributeError("Attributes of FeatureMapSize cannot be changed")

    def __delattr__(self, *args):
        raise AttributeError("Attributes of FeatureMapSize cannot be deleted")

    def __repr__(self):
        return "{c}(w={w}, h={h})".format(c=FeatureMapSize.__name__,
                                          w=self.w, h=self.h)

    def __eq__(self, othr):
        return (isinstance(othr, type(self))
                and (self.w, self.h) == (othr.w, othr.h))

    def __hash__(self):
        return hash((self.w, self.h))
