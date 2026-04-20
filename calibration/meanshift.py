import numpy

def circularNeighbors(img, x, y, radius):
    imh, imw = img.shape[:2]
    maxx = range(int(max(0,numpy.floor(x-radius) -1 )),int(min(imw, numpy.ceil(x + radius))))
    maxy = range(int(max(0,numpy.floor(y-radius) -1 )),int(min(imh, numpy.ceil(y + radius))))
    cx,cy = numpy.meshgrid(maxx,maxy)
    distance = ((cx-x) * (cx-x)) + ((cy-y) * (cy-y))
    mask = distance <= (radius * radius)
    inrangex = cx[mask]
    inrangey = cy[mask]
    R = img[inrangey-1,inrangex-1,0]
    G = img[inrangey-1,inrangex-1,1]
    B = img[inrangey-1,inrangex-1,2]
    X = numpy.column_stack((inrangex,inrangey,R,G,B)).astype(float)
    return X

def colorHistogram(cn, bins, x, y, h):
    hist = numpy.zeros((bins,bins,bins))
    bin_width = 256/bins
    for i in range(0,cn.shape[0]):
        rbin = int(min(numpy.floor(cn[i,2]/bin_width), bins -1))
        gbin = int(min(numpy.floor(cn[i,3]/bin_width), bins - 1))
        bbin = int(min(numpy.floor(cn[i,4]/bin_width), bins - 1))
        distance = (((cn[i,0] - x)/h)**2) + (((cn[i,1] - y)/h)**2)
        weight = 0
        if distance <= 1:
            weight = (1 - distance)
        hist[rbin,gbin,bbin] += weight
    weights = numpy.sum(hist) + 0.001
    hist = hist/weights
    return hist

def meanShiftWeights(X, q_model, p_test, bins):
    num_points = X.shape[0]
    w = numpy.zeros((num_points, 1))
    bin_range = 256 / bins
    for i in range(num_points):
        r_bin = int(min(numpy.floor(X[i,2]/bin_range),bins -1))
        g_bin = int(min(numpy.floor(X[i,3]/bin_range),bins -1))
        b_bin = int(min(numpy.floor(X[i,4]/bin_range),bins -1))
        target = q_model[r_bin, g_bin, b_bin]
        test = p_test[r_bin,g_bin,b_bin]
        if test > 0:
            w[i] = numpy.sqrt(target/test)
        else:
            w[i] = 0
    

    return w
