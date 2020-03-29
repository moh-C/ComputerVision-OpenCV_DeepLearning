import numpy as np
import cv2

def load_image(i):
    string = './MasterOpenCV/feature/' + str(i) + '.jpg'
    image = cv2.imread(string)
    return image
    
def gray(image):
    if image.shape[2] == 3:
        return cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    return image

def canny(image, min_Value=30, max_Value=250):
    gry = gray(cv2.GaussianBlur(image, (3,3), 1))
    return cv2.Canny(gry, min_Value, max_Value)

def otsu_(image):
    img = gray(image.copy())
    blur = cv2.GaussianBlur(img, (5,5), 2)
    _, th = cv2.threshold(blur, 0, 250, cv2.THRESH_OTSU)
    return th

def laplacian(image):
    return cv2.Laplacian(gray(image), cv2.CV_64F)
    
def sobel(image):
    return cv2.Sobel(gray(image), cv2.CV_64F, 1, 1, ksize=3)
    
def contours(image, method=1):
    edge = None
    if method==1:
        edge = canny(image)
    elif method==2:
        edge = laplacian(image)
    elif method==3:
        edge = sobel(image)
    else:
        edge = image
    _, contours, _ = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sorted_areas = sorted(contours, key=cv2.contourArea ,reverse=True)
    return sorted_areas

def drawContours(image, contour):
    copy = image.copy()
    return cv2.drawContours(copy, contour, -1, (0,255,0), 3)

def apply_kernel(image):
    kernel = np.float32([[0,-1,0],
                         [-1,4,-1],
                         [0,-1,0]], dtype=np.uint8)
    gry = gray(cv2.GaussianBlur(image, (7,7), 2))
    return cv2.filter2D(gry, -1, kernel)

def lines(image, minVal=5, maxVal=40, thresh=140):
    image_ = image.copy()
    edge = canny(image_, minVal, maxVal)

    lines = cv2.HoughLines(edge, 1, np.pi/180, thresh)
    for line_ in lines:
        rho = line_[0][0]
        theta = line_[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image_

def dilate(image, ksize=3, iter=1):
    kernel = np.ones((ksize,ksize))
    return cv2.dilate(image, kernel, iterations=iter)

def erode(image, ksize=3, iter=1):
    kernel = np.ones((ksize,ksize))
    return cv2.erode(image, kernel, iterations=iter)

def boundingRect(image, cnt):
    acc = 0.03 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, acc, True)
    return drawContours(image, [approx]), approx

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def warping(image, h):
    img = image.copy()
    img = resize(img, height = h)
    ratio = img.shape[0] / float(h)
    edge = canny(img, 5, 70)
    dilated = dilate(edge, 7, 9)
    eroded = erode(dilated, 7, 9)
    cnt = contours(eroded, method=5)[0]
    _, approx = boundingRect(img, cnt)
    return four_point_transform(img, approx.reshape(4, 2) * ratio)

def show(image):
    cv2.imshow('Image', image)
    cv2.waitKey()
    cv2.destroyAllWindows()