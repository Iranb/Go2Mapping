"""Pose-guided spherical panorama registration and periodic seam blending."""
import math

import cv2
import numpy as np

from go2_photo_depth import CameraModel, rotation


OPTICAL = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def decode(photos):
    images = []
    for photo in photos:
        image = cv2.imdecode(np.frombuffer(photo['jpeg'], np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Could not decode a panorama photo')
        scale = min(1., 800/image.shape[1])
        images.append(cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image)
    return images


def orientations(photos, model, parameters):
    return [rotation(np.asarray(p.get('rpy', [0., 0., p['yaw']]))+parameters[3*i:3*i+3]) @
            rotation([0., math.radians(model.pitch_deg), math.radians(model.yaw_deg)]) @ OPTICAL
            for i, p in enumerate(photos)]


def rays(points, shape, orient, focal_scale=1., k1=0., hfov=100.):
    h, w = shape[:2]
    focal = w/(2*math.tan(math.radians(hfov)/2))*focal_scale
    distorted = (np.asarray(points)-[(w-1)/2, (h-1)/2])/focal
    undistorted = distorted.copy()
    for _ in range(8):
        undistorted = distorted/np.maximum(.3, 1+k1*np.sum(undistorted**2, axis=1))[:, None]
    optical = np.column_stack((undistorted, np.ones(len(points))))
    optical /= np.linalg.norm(optical, axis=1)[:, None]
    return optical @ orient.T


def feature_pairs(images, photos):
    sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=.035)
    features = [sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None) for image in images]
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = []
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            delta = math.atan2(math.sin(photos[i]['yaw']-photos[j]['yaw']), math.cos(photos[i]['yaw']-photos[j]['yaw']))
            if abs(delta) > math.radians(33):
                continue
            ka, da = features[i]
            kb, db = features[j]
            if da is None or db is None or min(len(da), len(db)) < 16:
                continue
            matches = [a for a, b in matcher.knnMatch(da, db, k=2) if a.distance < .72*b.distance]
            if len(matches) < 16:
                continue
            a = np.float64([ka[m.queryIdx].pt for m in matches])
            b = np.float64([kb[m.trainIdx].pt for m in matches])
            _, mask = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
            if mask is None or mask.sum() < 14 or mask.mean() < .35:
                continue
            a, b = a[mask.ravel() > 0], b[mask.ravel() > 0]
            if min(np.ptp(a[:, 0]), np.ptp(b[:, 0])) < images[i].shape[1]*.12:
                continue
            ids = np.linspace(0, len(a)-1, min(80, len(a))).astype(int)
            pairs.append((i, j, a[ids], b[ids]))
    return pairs


def refine(images, photos, model, pairs, width=3072):
    """Fit small pose corrections and shared focal/distortion with robust ray errors."""
    count = len(photos)
    parameters = np.zeros(count*3+2)
    if not pairs:
        return parameters, {'matched_pairs': 0, 'matched_features': 0, 'before_px': None, 'after_px': None}
    ids_a = np.concatenate([np.full(len(a), i) for i, j, a, b in pairs])
    ids_b = np.concatenate([np.full(len(a), j) for i, j, a, b in pairs])
    xy_a = np.concatenate([a for i, j, a, b in pairs])
    xy_b = np.concatenate([b for i, j, a, b in pairs])
    points = np.concatenate((xy_a, xy_b))
    ids = np.concatenate((ids_a, ids_b))
    n = len(xy_a)

    def residual(p):
        orient = orientations(photos, model, p)
        world = np.empty((len(points), 3))
        for i in range(count):
            selected = ids == i
            world[selected] = rays(points[selected], images[i].shape, orient[i], math.exp(p[-2]), p[-1], model.hfov_deg)
        return world[:n]-world[n:]

    initial = residual(parameters)
    limits = np.tile(np.radians([5., 8., 12.]), count).tolist()+[.25, .18]
    prior = np.tile(np.radians([2., 3., 4.]), count).tolist()+[.18, .12]
    regularizer = np.diag(1/np.asarray(prior)**2)*.015
    for _ in range(12):
        error = residual(parameters)
        lengths = np.linalg.norm(error, axis=1)
        weight = np.repeat(np.minimum(1., .008/np.maximum(lengths, 1e-8)), 3)
        jacobian = np.empty((n*3, len(parameters)))
        for k in range(len(parameters)):
            trial = parameters.copy()
            trial[k] += 1e-5
            jacobian[:, k] = ((residual(trial)-error)/1e-5).ravel()
        normal = jacobian.T@(weight[:, None]*jacobian)+regularizer+np.eye(len(parameters))*.001
        gradient = jacobian.T@(weight*error.ravel())+regularizer@parameters
        step = np.linalg.solve(normal, gradient)
        candidate = np.clip(parameters-step, -np.asarray(limits), limits)
        # Retain a step only when the robust data loss and pose prior improve.
        def loss(p):
            length = np.linalg.norm(residual(p), axis=1)
            return np.sum(np.where(length < .008, .5*length**2, .008*(length-.004)))+.5*p@regularizer@p
        old_loss = loss(parameters)
        for _ in range(5):
            if loss(candidate) <= old_loss:
                break
            candidate = (candidate+parameters)/2
        if loss(candidate) > old_loss:
            break
        change = np.max(np.abs(candidate-parameters))
        parameters = candidate
        if change < 1e-5:
            break
    final = residual(parameters)
    scale = width/(2*math.pi)
    stats = {'matched_pairs': len(pairs), 'matched_features': n,
             'before_px': float(np.median(np.linalg.norm(initial, axis=1))*scale),
             'after_px': float(np.median(np.linalg.norm(final, axis=1))*scale)}
    return parameters, stats


def warp_tiles(images, photos, model, parameters, width=3072, height=768):
    orient = orientations(photos, model, parameters)
    centers = np.asarray([math.atan2(r[1, 2], r[0, 2]) % (2*math.pi) for r in orient])
    order = np.argsort(centers)
    tiles = []
    latitude = (height/2-.5-np.arange(height))*2*math.pi/width
    for k, i in enumerate(order):
        center = centers[i]
        prev_gap = (center-centers[order[k-1]]) % (2*math.pi) if len(order) > 1 else 2*math.pi
        next_gap = (centers[order[(k+1) % len(order)]]-center) % (2*math.pi) if len(order) > 1 else 2*math.pi
        half_fov = math.radians(model.hfov_deg/2+8)
        left = center-min(half_fov, prev_gap/2+math.radians(8))
        right = center+min(half_fov, next_gap/2+math.radians(8))
        x0, x1 = math.floor(left*width/(2*math.pi)), math.ceil(right*width/(2*math.pi))
        longitude = (np.arange(x0, x1)+.5)*2*math.pi/width
        lon, lat = np.meshgrid(longitude, latitude)
        world = np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), axis=-1)
        optical = world@orient[i]
        normalized = optical[:, :, :2]/np.maximum(optical[:, :, 2:], 1e-6)
        distorted = normalized*(1+parameters[-1]*np.sum(normalized**2, axis=-1, keepdims=True))
        h, w = images[i].shape[:2]
        focal = w/(2*math.tan(math.radians(model.hfov_deg)/2))*math.exp(parameters[-2])
        uv = distorted*focal+[(w-1)/2, (h-1)/2]
        valid = ((optical[:, :, 2] > .1) & (uv[:, :, 0] >= 1) & (uv[:, :, 0] < w-2) &
                 (uv[:, :, 1] >= 1) & (uv[:, :, 1] < h-2) & (np.sum(normalized**2, axis=-1) < 5.))
        tile = cv2.remap(images[i], uv[:, :, 0].astype(np.float32), uv[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
        tiles.append((x0, tile, valid.astype(np.uint8)*255))
    return tiles


def blend_tiles(tiles, width, height):
    """Repeat tiles around a central revolution to blend across 0/360 degrees."""
    corners, images, masks = [], [], []
    padding = width//2
    for x, image, mask in tiles:
        for shift in (-width, 0, width, 2*width):
            start = x+shift+padding
            left, right = max(0, -start), min(image.shape[1], 2*width-start)
            if left < right:
                corners.append((start+left, 0))
                images.append(image[:, left:right].copy())
                masks.append(cv2.UMat(mask[:, left:right]))
    exposure = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN)
    exposure.feed(corners, images, masks)
    for i in range(len(images)):
        exposure.apply(i, corners[i], images[i], masks[i])
    seam = cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
    seam.find([im.astype(np.float32) for im in images], corners, masks)
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(4)
    blender.prepare((0, 0, 2*width, height))
    for image, mask, corner in zip(images, masks, corners):
        blender.feed(image.astype(np.int16), mask, corner)
    result, coverage = blender.blend(None, None)
    result = np.clip(result[:, padding:padding+width], 0, 255).astype(np.uint8)
    coverage = coverage[:, padding:padding+width]
    return np.dstack((result, coverage))


def build_panorama(photos, model=None, width=3072):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(7)
    model = model or CameraModel()
    images = decode(photos)
    if not images:
        raise ValueError('Capture panorama photos first')
    pairs = feature_pairs(images, photos)
    parameters, stats = refine(images, photos, model, pairs, width)
    height = width//4
    tiles = warp_tiles(images, photos, model, parameters, width, height)
    result = blend_tiles(tiles, width, height)
    ok, encoded = cv2.imencode('.png', result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError('Could not encode the panorama')
    return {'image': encoded.tobytes(), 'width': width, 'height': height,
            'latitude_min': -math.pi/4, 'latitude_max': math.pi/4,
            'frames': len(photos), 'focal_scale': math.exp(parameters[-2]), 'radial_k1': float(parameters[-1]),
            'corrections_deg': np.degrees(parameters[:-2].reshape(-1, 3)).tolist(), **stats}
