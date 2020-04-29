""" Software to create Precision-Recall-Gain curves.

Precision-Recall-Gain curves and how to cite this work is available at
http://www.cs.bris.ac.uk/~flach/PRGcurves/.
"""
import warnings
import numpy as np
from typing import Iterable
import matplotlib.pyplot as plt

epsilon = 1e-7


def precision(tp, fp):
    return tp / (tp + fp + epsilon)


def recall(tp, fn):
    return tp / (tp + fn + epsilon)


def precision_gain(tp, fn, fp, tn):
    """Calculates Precision Gain from the contingency table

    This function calculates Precision Gain from the entries of the contingency
    table: number of true positives (TP), false negatives (FN), false positives
    (FP), and true negatives (TN). More information on Precision-Recall-Gain
    curves and how to cite this work is available at
    http://www.cs.bris.ac.uk/~flach/PRGcurves/.
    """
    n_pos = tp + fn
    n_neg = fp + tn
    prec_gain = 1. - (n_pos / (n_neg + epsilon)) * (fp / (tp + epsilon))
    if isinstance(prec_gain, Iterable):
        prec_gain[tn + fn == 0] = 0
    elif tn + fn == 0:
        prec_gain = 0
    return prec_gain


def recall_gain(tp, fn, fp, tn):
    """Calculates Recall Gain from the contingency table

    This function calculates Recall Gain from the entries of the contingency
    table: number of true positives (TP), false negatives (FN), false positives
    (FP), and true negatives (TN). More information on Precision-Recall-Gain
    curves and how to cite this work is available at
    http://www.cs.bris.ac.uk/~flach/PRGcurves/.

    Args:
        tp (float) or ([float]): True Positives
        fn (float) or ([float]): False Negatives
        fp (float) or ([float]): False Positives
        tn (float) or ([float]): True Negatives
    Returns:
        (float) or ([float])
    """
    n_pos = tp + fn
    n_neg = fp + tn
    recall_gain = 1. - (n_pos / (n_neg + epsilon)) * (fn / (tp + epsilon))
    if isinstance(recall_gain, Iterable):
        recall_gain[tn + fn == 0] = 1
    elif tn + fn == 0:
        recall_gain = 1
    return recall_gain

def create_segments(y_true, y_pred):
    """
    for each class:
        sort descending confidence
        for each confidence level:
            n_positive, n_negative in gr truth
        , slice batch col
    sort descending confidence
    """
    n_samples, n_classes = np.shape(y_true)
    n_true_pos_per_class = y_true.sum(axis=0)
    n_true_neg_per_class = n_samples*np.ones(n_classes) - n_true_pos_per_class

    threshed_metrics = []
    for thresh in np.linspace(1,0,101):  # exactly 0.01 apart
        pred_threshed = (y_pred >= thresh)
        # num_correct = ((pred_threshed == y_true).sum(dim=0))

        # these are classwise
        tp = (y_true * y_pred).sum(dim=0)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)

        # these are class-wise inference
        precisions = precision(tp, fp)
        precision_gains = precision_gain(tp, fn, fp, tn)
        recalls = recall(tp, fn)
        recall_gains = recall_gain(tp, fn, fp, tn)

        f1s = 2* (precisions*recalls) / (precisions + recalls + epsilon)
        f1s = f1s.clamp(min=epsilon, max=1-epsilon)  # TODO should be unnecessary

        threshed_metrics.append(dict(thresh=thresh,
                                     tp=tp,
                                     tn=tn,
                                     fp=fp,
                                     fn=fn,
                                     precisions=precisions,
                                     precision_gains=precision_gains,
                                     recalls=recalls,
                                     recall_gains=recall_gains,
                                     f1s=f1s))

    return threshed_metrics

def get_point(curve, index):
    keys = curve.keys()
    point = np.zeros(len(keys))
    key_indices = dict()
    for i, key in enumerate(keys):
        point[i] = curve[key][index]
        key_indices[key] = i
    return [point, key_indices]


def insert_point(new_point, key_indices, curve, precision_gain=0,
                 recall_gain=0, is_crossing=0):
    for key in key_indices.keys():
        curve[key] = np.insert(curve[key], 0, new_point[key_indices[key]])
    curve['precision_gain'][0] = precision_gain
    curve['recall_gain'][0] = recall_gain
    curve['is_crossing'][0] = is_crossing
    new_order = np.lexsort((-curve['precision_gain'], curve['recall_gain']))
    for key in curve.keys():
        curve[key] = curve[key][new_order]
    return curve


def _create_crossing_points(curve, n_pos, n_neg, n_classes):
    n = n_pos + n_neg
    curve['is_crossing'] = np.zeros(n_classes)
    # introduce a crossing point at the crossing through the y-axis
    j = np.amin(np.where(curve['recall_gain'] >= 0)[0])
    if curve['recall_gain'][j] > 0:  # otherwise there is a point on the boundary and no need for a crossing point
        [point_1, key_indices_1] = get_point(curve, j)
        [point_2, key_indices_2] = get_point(curve, j - 1)
        delta = point_1 - point_2
        if delta[key_indices_1['TP']] > 0:
            alpha = (n_pos * n_pos / n - curve['TP'][j - 1]) / delta[key_indices_1['TP']]
        else:
            alpha = 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_point = point_2 + alpha * delta

        new_prec_gain = precision_gain(new_point[key_indices_1['TP']], new_point[key_indices_1['FN']],
                                       new_point[key_indices_1['FP']], new_point[key_indices_1['TN']])
        curve = insert_point(new_point, key_indices_1, curve,
                             precision_gain=new_prec_gain, is_crossing=1)

    # now introduce crossing points at the crossings through the non-negative part of the x-axis
    x = curve['recall_gain']
    y = curve['precision_gain']
    temp_y_0 = np.append(y, 0)
    temp_0_y = np.append(0, y)
    temp_1_x = np.append(1, x)
    indices = np.where(np.logical_and((temp_y_0 * temp_0_y < 0), (temp_1_x >= 0)))[0]
    for i in indices:
        cross_x = x[i - 1] + (-y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])
        [point_1, key_indices_1] = get_point(curve, i)
        [point_2, key_indices_2] = get_point(curve, i - 1)
        delta = point_1 - point_2
        if delta[key_indices_1['TP']] > 0:
            alpha = (n_pos * n_pos / (n - n_neg * cross_x) - curve['TP'][i - 1]) / delta[key_indices_1['TP']]
        else:
            alpha = (n_neg / n_pos * curve['TP'][i - 1] - curve['FP'][i - 1]) / delta[key_indices_1['FP']]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_point = point_2 + alpha * delta

        new_rec_gain = recall_gain(new_point[key_indices_1['TP']], new_point[key_indices_1['FN']],
                                   new_point[key_indices_1['FP']], new_point[key_indices_1['TN']])
        curve = insert_point(new_point, key_indices_1, curve,
                             recall_gain=new_rec_gain, is_crossing=1)
        i += 1
        indices += 1
        x = curve['recall_gain']
        y = curve['precision_gain']
        temp_y_0 = np.append(y, 0)
        temp_0_y = np.append(0, y)
        temp_1_x = np.append(1, x)
    return curve


def create_prg_curve(y_true, y_pred):
    """Precision-Recall-Gain curve

    This function creates the Precision-Recall-Gain curve from the vector of
    y_true and vector of scores where higher score indicates a higher
    probability to be positive. More information on Precision-Recall-Gain
    curves and how to cite this work is available at
    http://www.cs.bris.ac.uk/~flach/PRGcurves/.
    """
    threshed_metrics = create_segments(y_true, y_pred)
    # calculate recall gains and precision gains for all thresholds
    curve = dict()
    curve['pos_score'] = np.insert(segments['pos_score'], 0, np.inf) # start at inf
    curve['neg_score'] = np.insert(segments['neg_score'], 0, -np.inf) # start at -inf
    curve['TP'] = np.insert(np.cumsum(segments['pos_count']), 0, 0) # start at 0
    curve['FP'] = np.insert(np.cumsum(segments['neg_count']), 0, 0) # start at 0
    curve['FN'] = n_pos - curve['TP']
    curve['TN'] = n_neg - curve['FP']
    # curve['TP'] = (y_true * y_pred).sum(axis=0)  # .to(torch.float32)
    # curve['TN'] = ((1 - y_true) * (1 - y_pred)).sum(axis=0)  # .to(torch.float32)
    # curve['FP'] = ((1 - y_true) * y_pred).sum(axis=0)  # .to(torch.float32)
    # curve['FN'] = (y_true * (1 - y_pred)).sum(axis=0)  # .to(torch.float32)

    curve['precision'] = precision(curve['TP'], curve['FP'])
    curve['recall'] = recall(curve['TP'], curve['FN'])
    curve['precision_gain'] = precision_gain(curve['TP'], curve['FN'], curve['FP'], curve['TN'])
    curve['recall_gain'] = recall_gain(curve['TP'], curve['FN'], curve['FP'], curve['TN'])
    curve = _create_crossing_points(curve, n_pos, n_neg, n_classes)

    curve['in_unit_square'] = np.logical_and(curve['recall_gain'] >= 0,
                                              curve['precision_gain'] >= 0)
    return curve #tp, fp, tn, fn, precision_gain, recall_gain, n_thresh


def calc_auprg(prg_curve):
    """Calculate area under the Precision-Recall-Gain curve

    This function calculates the area under the Precision-Recall-Gain curve
    from the results of the function create_prg_curve. More information on
    Precision-Recall-Gain curves and how to cite this work is available at
    http://www.cs.bris.ac.uk/~flach/PRGcurves/.
    """
    area = 0
    recall_gain = prg_curve['recall_gain']
    precision_gain = prg_curve['precision_gain']
    for i in range(1, len(recall_gain)):
        if (not np.isnan(recall_gain[i - 1])) and (recall_gain[i - 1] >= 0):
            width = recall_gain[i] - recall_gain[i - 1]
            height = (precision_gain[i] + precision_gain[i - 1]) / 2
            area += width * height
    return (area)


def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    Source code from:
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return upper


def plot_prg(prg_curve, show_convex_hull=True, show_f_calibrated_scores=False):
    """Plot the Precision-Recall-Gain curve

    This function plots the Precision-Recall-Gain curve resulting from the
    function create_prg_curve using ggplot. More information on
    Precision-Recall-Gain curves and how to cite this work is available at
    http://www.cs.bris.ac.uk/~flach/PRGcurves/.

    @param prg_curve the data structure resulting from the function create_prg_curve
    @param show_convex_hull whether to show the convex hull (default: TRUE)
    @param show_f_calibrated_scores whether to show the F-calibrated scores (default:TRUE)
    @return the ggplot object which can be plotted using print()
    @details This function plots the Precision-Recall-Gain curve, indicating
        for each point whether it is a crossing-point or not (see help on
        create_prg_curve). By default, only the part of the curve
        within the unit square [0,1]x[0,1] is plotted.
    @examples
        labels = c(1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,1,0,0,0,1,0,1)
        scores = (25:1)/25
        plot_prg(create_prg_curve(labels,scores))
    """
    pg = prg_curve['precision_gain']
    rg = prg_curve['recall_gain']

    fig = plt.figure(figsize=(6, 5))
    plt.clf()
    plt.axes(frameon=False)
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1.25, 0.25))
    ax.set_yticks(np.arange(0, 1.25, 0.25))
    ax.grid(b=True)
    ax.set_xlim((-0.05, 1.02))
    ax.set_ylim((-0.05, 1.02))
    ax.set_aspect('equal')
    # Plot vertical and horizontal lines crossing the 0 axis
    plt.axvline(x=0, ymin=-0.05, ymax=1, color='k')
    plt.axhline(y=0, xmin=-0.05, xmax=1, color='k')
    plt.axvline(x=1, ymin=0, ymax=1, color='k')
    plt.axhline(y=1, xmin=0, xmax=1, color='k')
    # Plot cyan lines
    indices = np.arange(np.argmax(prg_curve['in_unit_square']) - 1,
                        len(prg_curve['in_unit_square']))
    plt.plot(rg[indices], pg[indices], 'c-', linewidth=2)
    # Plot blue lines
    indices = np.logical_or(prg_curve['is_crossing'],
                            prg_curve['in_unit_square'])
    plt.plot(rg[indices], pg[indices], 'b-', linewidth=2)
    # Plot blue dots
    indices = np.logical_and(prg_curve['in_unit_square'],
                             True - prg_curve['is_crossing'])
    plt.scatter(rg[indices], pg[indices], marker='o', color='b', s=40)
    # Plot lines out of the boundaries
    plt.xlabel('Recall Gain')
    plt.ylabel('Precision Gain')

    valid_points = np.logical_and(~ np.isnan(rg), ~ np.isnan(pg))
    upper_hull = convex_hull(zip(rg[valid_points], pg[valid_points]))
    rg_hull, pg_hull = zip(*upper_hull)
    if show_convex_hull:
        plt.plot(rg_hull, pg_hull, 'r--')
    if show_f_calibrated_scores:
        raise Exception("Show calibrated scores not implemented yet")
    plt.show()
    return fig


def plot_pr(prg_curve):
    p = prg_curve['precision']
    r = prg_curve['recall']

    fig = plt.figure(figsize=(6, 5))
    plt.clf()
    plt.axes(frameon=False)
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 1.25, 0.25))
    ax.set_yticks(np.arange(0, 1.25, 0.25))
    ax.grid(b=True)
    ax.set_xlim((-0.05, 1.02))
    ax.set_ylim((-0.05, 1.02))
    ax.set_aspect('equal')
    # Plot vertical and horizontal lines crossing the 0 axis
    plt.axvline(x=0, ymin=-0.05, ymax=1, color='k')
    plt.axhline(y=0, xmin=-0.05, xmax=1, color='k')
    plt.axvline(x=1, ymin=0, ymax=1, color='k')
    plt.axhline(y=1, xmin=0, xmax=1, color='k')
    # Plot blue lines
    plt.plot(r, p, 'ob-', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.show()
    return fig
