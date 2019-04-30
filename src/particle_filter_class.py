# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:05:51 2019

@author: lee
"""

import cv2
import numpy as np
import copy

class Rect(object):
    """
    Class of Rectagle bounding box (lx, ly, w, h)
        lx: col index of the left-up conner
        ly: row index of the left-up conner
        w: width of rect (pixel)
        h: height of rect (pixel)
    """
    def __init__(self, lx=0, ly=0, w=0, h=0):
        self.lx = int(ly)
        self.ly = int(lx)
        self.w = int(w)
        self.h = int(h)
    
    def to_particle(self, ref_wh, sigmas=None):
        """
        Convert Rect to Particle
        :param ref_wh: reference size of particle
        :param sigmas: sigmas of (cx, cy, sx, sy) of particle
        :return: result particle
        """
        ptc = Particle(sigmas=sigmas)
        ptc.cx = self.lx + self.w / 2
        ptc.cy = self.ly + self.h / 2
        ptc.sx = self.w / ref_wh[0]
        ptc.sy = self.h / ref_wh[1]
        return ptc
    
    def clip_range(self, img_w, img_h):
        """
        Clip rect range into img size (Make sure the rect only containing pixels in image)
        :param img_w: width of image (pixel)
        :param img_h: height of image (pixel)
        :return:
        """
        self.lx = max(self.lx, 1)
        self.ly = max(self.ly, 1)
        
        self.w = min(img_w - self.lx - 1, self.w)
        self.h = min(img_h - self.ly - 1, self.h)
        return self


class Particle(object):
    """
    Class of particle (cx, cy, sx, sy), corresponding to a rectangle bounding box
        cx: col index of the center of rectangle (pixel)
        cy: row index of the center of rectangle (pixel)
        sx: width compared with a reference size
        xy: height compared with a reference size

        The following attrs are optional
        weight: weight of this particle
        sigmas: transition sigmas of this particle
    """
    def __init__(self, cx=0, cy=0, sx=0, sy=0, sigmas=None):
        self.cx = cx
        self.cy = cy
        self.sx = sx
        self.sy = sy
        self.weight = 0
        if sigmas is not None:
            self.sigmas = sigmas
        else:
            self.sigmas = [0, 0, 0, 0]
        
    def transition(self, dst_sigmas=None):
        """
        transition by Gauss Distribution with 'sigma'
	Fill it !!!!!!!!!!!!!!!
        """
        if dst_sigmas is None:
            sigmas = self.sigmas
        else:
            sigmas = dst_sigmas
        
        pass
        return self
    
    def update_weight(self, w):
        self.weight = w
    
    def to_rect(self, ref_wh):
        """
        Get the corresponding Rect of this particle
        :param ref_wh: reference size of particle
        :return: corresponding Rect
        """
        rect = Rect()
        rect.w = int(self.sx * ref_wh[0])
        rect.h = int(self.sy * ref_wh[1])
        rect.lx = int(self.cx - rect.w / 2)
        rect.ly = int(self.cy - rect.h / 2)
        return rect
    
    def clone(self):
        """
        Clone this particle
        :return: Deep copy of this particle
        """
        return copy.deepcopy(self)
    
    def display(self):
        print('cx: {}, cy: {}, sx: {}, sy: {}'.format(self.cx, self.cy, self.sx, self.sy))
        print('weight: {} sigmas:{}'.format(self.weight, self.sigmas))

    def __eq__(self, ptc):
        return self.weight == ptc.weight

    def __le__(self, ptc):
        return self.weight <= ptc.weight


def extract_feature(dst_img, rect, ref_wh, feature_type='intensity'):
    """
    Extract feature from the dst_img's ROI of rect.
    :param dst_img:
    :param rect: ROI range of dst_img
    :param ref_wh: reference size of particle
    :param feature_type:
    :return: A vector of features value
    """

    if feature_type == 'intensity':
        rect.clip_range(dst_img.shape[1], dst_img.shape[0])
        roi = dst_img[rect.lx:rect.lx + rect.w,
                      rect.ly:rect.ly + rect.h]
    
        scaled_roi = cv2.resize(roi, (ref_wh[0], ref_wh[1]))   # Fixed-size ROI
        return scaled_roi.astype(np.float).reshape(1, -1)/255.0
    else:
        print('Undefined feature type \'{}\' !!!!!!!!!!')
        return None


def transition_step(particles, sigmas):
    """
    Sample particles from Gaussian Distribution
    :param particles: Particle list
    :param sigmas: std of transition model
    :return: Transitioned particles
    """
    print('trans')
    for particle in particles:
        particle.transition(sigmas)
    return particles


def weighting_step(dst_img, particles, ref_wh, template, feature_type):
    """
    Compute each particle's weight by its similarity
    :param dst_img: Tracking image
    :param particles: Current particles
    :param ref_wh: reference size of particle
    :param template: template for matching
    :param feature_type:
    :return: weights of particles
    """
    pass


def resample_step(particles, weights, rsp_sigmas=None):
    """
    Resample particles according to their weights
    :param particles: Paricles needed resampling
    :param weights: Particles' weights
    :param rsp_sigmas: For transition of resampled particles
    """
    pass

    
def compute_similarities(features, template):
    """
    Compute similarities of a group of features with template
    :param features: features of particles
    :template: template for matching
    """
    pass
     
def compute_similarity(feature, template):
    """
    Compute similarity of a single feature with template
    :param feature: feature of a single particle
    :template: template for matching
    """
    pass

