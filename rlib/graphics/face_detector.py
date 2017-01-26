# import os
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
# import threading
# from .readers import read_image


FACE_FRONT_CV_CFG_PATH = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
FACE_PROFILE_CV_CFG_PATH = '/home/mephistopheies/models/opencv/lbpcascade_profileface.xml'


def _fd_profile_right_cv(img_rgb, fd_profile_left_cv,
                         face_scale_factor=1.1,
                         face_min_neighbors=10,
                         face_min_size=(40, 40)):
    fp_cv = fd_profile_left_cv.detectMultiScale(
        np.fliplr(img_rgb),
        scaleFactor=face_scale_factor,
        minNeighbors=face_min_neighbors,
        minSize=face_min_size)
    return [(img_rgb.shape[1] - x - w, y, w, h) for (x, y, w, h) in fp_cv]


def detect_single_face(img_rgb,
                       fd_front_dlib=None, fd_profile_left_cv=None, fd_front_cv=None,
                       face_scale_factor=1.1, face_min_neighbors=10, face_min_size=(40, 40),
                       rescale=None, detect_profile=False,
                       debug=False):

    if fd_front_dlib is None:
        fd_front_dlib = dlib.get_frontal_face_detector()
    if fd_profile_left_cv is None and detect_profile:
        fd_profile_left_cv = cv2.CascadeClassifier(FACE_PROFILE_CV_CFG_PATH)
    if fd_front_cv is None:
        fd_front_cv = cv2.CascadeClassifier(FACE_FRONT_CV_CFG_PATH)

    img_rgb_canvas = img_rgb.copy()
    face = None
    ff_dlib = fd_front_dlib(img_rgb, 1)
    if detect_profile:
        fp_cv = list(fd_profile_left_cv.detectMultiScale(
            img_rgb,
            scaleFactor=face_scale_factor,
            minNeighbors=face_min_neighbors,
            minSize=face_min_size)) + \
            _fd_profile_right_cv(
                img_rgb,
                fd_profile_left_cv,
                face_scale_factor=face_scale_factor,
                face_min_neighbors=face_min_neighbors,
                face_min_size=face_min_size)
    else:
        fp_cv = []

    ff_cv = fd_front_cv.detectMultiScale(
        img_rgb,
        scaleFactor=face_scale_factor,
        minNeighbors=face_min_neighbors,
        minSize=face_min_size)

    if len(ff_cv) > 0:
        ff_cv_max = sorted([(np.prod(t[2:]), t) for t in ff_cv],
                           key=lambda t: t[0], reverse=True)[0][1]
    if len(fp_cv) > 0:
        fp_cv_max = sorted([(np.prod(t[2:]), t) for t in fp_cv],
                           key=lambda t: t[0], reverse=True)[0][1]
    if len(ff_dlib) > 0:
        ff_dlib_max = sorted([(t.width()*t.height(), (t.left(), t.top(), t.width(), t.height()))
                              for t in ff_dlib],
                             key=lambda t: t[0], reverse=True)[0][1]

    if len(fp_cv) > 0 and len(ff_cv) == 0 and len(ff_dlib) == 0:
        # if there is only profile face
        if debug:
            print 'only profile face'
        face = sorted([(np.prod(t[2:]), t) for t in fp_cv], key=lambda t: t[0], reverse=True)[0][1]
    elif len(ff_cv) == 0 and len(ff_dlib) > 0:
        # if only dlib found front
        if debug:
            print 'only dlib found front'
        face = sorted([(t.width()*t.height(), (t.left(), t.top(), t.width(), t.height()))
                       for t in ff_dlib], key=lambda t: t[0], reverse=True)[0][1]
    elif len(ff_cv) > 0 and len(ff_dlib) > 0:
        # if cv and dlib both found front
        if debug:
            print 'cv and dlib both found front'
        face = zip((ff_cv_max[0], ff_cv_max[1], ff_cv_max[0] + ff_cv_max[2], ff_cv_max[1] + ff_cv_max[3]),
                   (ff_dlib_max[0], ff_dlib_max[1], ff_dlib_max[0] + ff_dlib_max[2], ff_dlib_max[1] + ff_dlib_max[3]))
        face = min(face[0]), min(face[1]), max(face[2]) - min(face[0]), max(face[3]) - min(face[1])
    elif len(ff_cv) > 0 and len(fp_cv) > 0 and len(ff_dlib) == 0 and \
            np.prod(ff_cv_max[2:])/float(np.prod(fp_cv_max[2:])) < 0.1:
        # if there is profile and artifact of open cv front
        face = fp_cv_max
    elif len(ff_cv) > 0 and len(ff_dlib) == 0:
        # if only cv found front
        if debug:
            print 'only dlib found front'
        face = ff_cv_max
    elif len(ff_cv) + len(ff_dlib) + len(fp_cv) == 0:
        if debug:
            print 'nothing'
        # plt.imshow(img_rgb_canvas)
        # plt.show()
    elif debug:
        print 'not processed'
        for (x, y, w, h) in fp_cv:
            cv2.rectangle(img_rgb_canvas, (x, y), (x + w, y + h),
                          (255, 0, 0), thickness=3, lineType=8, shift=0)
        print 'OpenCV profile (red): %i', len(fp_cv)
        for i, d in enumerate(ff_dlib):
            cv2.rectangle(img_rgb_canvas, (d.left(), d.top()), (d.right(), d.bottom()),
                          (0, 255, 0), thickness=3, lineType=8, shift=0)
        print 'dlib front (green): %i' % len(ff_dlib)
        for (x, y, w, h) in ff_cv:
            cv2.rectangle(img_rgb_canvas, (x, y), (x + w, y + h),
                          (0, 0, 255), thickness=3, lineType=8, shift=0)
        print 'OpenCV front (blue): %i', len(ff_cv)
        plt.imshow(img_rgb_canvas)
        plt.show()
    if rescale is not None and face is not None:
        (x, y, w, h) = face
        x = max(0, int(x + w/2 - w*rescale/2))
        y = max(0, int(y + h/2 - h*rescale/2))
        w = min(img_rgb_canvas.shape[1] - x, int(w*rescale))
        h = min(img_rgb_canvas.shape[0] - y, int(h*rescale))
        face = (x, y, w, h)
    if face is not None and debug:
        (x, y, w, h) = face
        cv2.rectangle(img_rgb_canvas, (x, y), (x + w, y + h),
                      (255, 255, 255), thickness=3, lineType=8, shift=0)
        plt.imshow(img_rgb_canvas)
        plt.show()
    return face
