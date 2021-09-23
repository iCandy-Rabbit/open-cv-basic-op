
# 对比度/亮度
# 饱和度
# 色调
# 重新着色

import cv2 as cv
import numpy as np

# 柔化  使用3*3的模板和5*5的模板  使图片更加平滑,模板越大，平滑效果越明显
def smooth_blur(img_path):
    img = cv.imread(img_path)
    imgResize = cv.resize(img,(512,512))
    new_img_3_3 = cv.blur(imgResize,(3,3))
    new_img_5_5 = cv.blur(imgResize,(5,5))
    cv.imshow('pro_img',imgResize)
    cv.imshow('soften_3*3',new_img_3_3)
    cv.imshow('soften_5*5',new_img_5_5)
    cv.waitKey(0)
    cv.destroyAllWindows()

# 锐化  使用的使四邻域的拉普拉斯
def sharpen(img_path):
    img = cv.imread(img_path)
    imgResize = cv.resize(img, (512, 512))
    # 四邻域拉普拉斯
    kernel = np.array([[0,-1,0],
                      [-1,5,-1],
                      [0,-1,0]])
    new_img = cv.filter2D(imgResize,-1,kernel)
    lpls = cv.convertScaleAbs(new_img)#  # 转回uint8 否则将无法显示图像，而只是一副灰色的窗口
    cv.imshow('original img',imgResize)

    cv.imshow('sharpen img Laplacian',lpls) # 四邻域拉普拉斯
    cv.waitKey(0)
    cv.destroyAllWindows()
    pass
# 锐化 使用的sobel算子
def sharpen_sobel(img_path):
    original_img = cv.imread(img_path)
    original_imgResize = cv.resize(original_img, (512, 512))

    sobel_1 = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])
    sobel_2 = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])
    sobel_img = cv.filter2D(original_imgResize, -1, sobel_1) # -1表示与原图像深度相同  水平方向

    sobel_img = cv.filter2D(sobel_img,-1,sobel_2)# 垂直方向

    sobel_img = cv.convertScaleAbs(sobel_img)

    cv.imshow('original img', original_imgResize)
    cv.imshow('new img  sobel', sobel_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
# 亮度/对比度
# 原始形式  比较慢
# 参数a为 对比度  参数b为亮度
def brightness_ontrast_ratio_1(img_path,a,b):
    original_img = cv.imread(img_path)
    # 提升亮度和对比度
    """
    g(x,y) = a*f(x,y)+b
     f(x,y)代表源图像x行，y列的像素点的c通道的数值 g(x,y)代表目
     标图像x行，y列的像素点的c通道的数值
           a参数（a>0）表示放大的倍数（一般在0.0~3.0之间）
           b参数一般称为偏置，用来调节亮度
    """
    original_imgResize = cv.resize(original_img, (512, 512))
    new_img = original_imgResize.copy()
    rows, cols, channel = new_img.shape
    for i in range(rows):
        for j in range(cols):
            for c in range(3):
                color = new_img[i, j][c] * a + b
                if color > 255:
                    new_img[i, j][c] = 255
                elif color < 0:
                    new_img[i, j][c] = 0
                else:
                    new_img[i, j][c] = color
    cv.imshow('original img', original_imgResize)
    cv.imshow('new img original_method', new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
# 使用cv   比较快
# a是对比度的值，b是亮度的值
def brightness_ontrast_ratio_2(img_path, a, b):
    original_img = cv.imread(img_path)
    original_imgResize = cv.resize(original_img, (512, 512))
    # 提升亮度和对比度
    # 我们把图像img2定义为与图像img同样规格的全黑图片（像素全为0）。这样我们改变α \alphaα的值就相当于调整图像img的对比度和亮度了。
    # 使用图片相加
    # 获取shape的值，height：高度  width: 宽度  passageway: 通道
    height,width,passageway = original_imgResize.shape
    # 设置和原图片一样大小的纯黑图片
    img2 = np.zeros([height,width,passageway],original_imgResize.dtype)
    new_img = cv.addWeighted(original_imgResize,a,img2,1-a,b)
    cv.imshow('original img', original_imgResize)
    cv.imshow('new img  cv_method', new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    pass
# 色彩饱和度转HSI  提高S
def img_HSI_S(img_path):
    original_img = cv.imread(img_path)
    original_imgResize = cv.resize(original_img, (512, 512))
    hsv = cv.cvtColor(original_imgResize,cv.COLOR_RGB2HSV)
    H,S,V = cv.split(hsv)

    # 明度
    v = V.ravel()[np.flatnonzero(V)] #亮度非零的值
    average_v = sum(v)/len(v)
    print(average_v)

    # 饱和度
    s = S.ravel()[np.flatnonzero(S)]
    average_s = sum(s)/len(s)
    print(average_s)

    # 色调H
    h = H.ravel()[np.flatnonzero(H)]
    average_h = sum(h)/len(h)
    print(average_h)

    pass
# 色调 转HSI  提高H 再转会RGB

# 重新着色 包含 二值图(调整阈值)，
def resetColor(img_path):
    original_img = cv.imread(img_path)
    original_imgResize = cv.resize(original_img, (512, 512))

    bin_img_threshold1 = original_imgResize.copy()
    # 1.二值图
    GrayImg = cv.cvtColor(bin_img_threshold1,cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(GrayImg,127,255,cv.THRESH_BINARY)
    # 2.其他颜色图  灰度图 和其他图片 进行图片加法运算  和蓝色图片运算
    img_rest = np.ones((512, 512), dtype=np.uint8)
    bgr_img_rest = cv.cvtColor(img_rest, cv.COLOR_GRAY2BGR)
    # BGR  0-2
    bgr_img_rest[:, :, 0] = 255 #纯蓝色
    bgr_img_rest[:, :, 1] = 0
    bgr_img_rest[:, :, 2] = 0
    # blue_img = np.array(,GrayImg.dtype)
    a = 0.7
    b= 0
    print(GrayImg.shape)
    print(original_imgResize.shape)
    # new_img = cv.addWeighted(GrayImg, a, bgr_img_rest, 1 - a, b)
    # 灰度图转3通道
    out = cv.cvtColor(GrayImg,cv.COLOR_GRAY2RGB)
    new_img = cv.addWeighted(out, a, bgr_img_rest, 1 - a, b)
    cv.imshow('original img', original_imgResize)
    cv.imshow('bin_img_t_127', thresh)
    cv.imshow('bin_img_b', bgr_img_rest)
    cv.imshow('out', out)
    cv.imshow('new_img', new_img)
    # cv.imshow('new_img', new_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    pass
img_path = 'MyPic.jpg'
# smooth_blur(img_path) # 柔化 使用3*3的模板和5*5的模板
# sharpen(img_path)  # 使用拉普拉斯算子
# sharpen_sobel(img_path) # 使用sobel算子进行水平方向和垂直方向
# brightness_ontrast_ratio_1(img_path,1.2,100) # 原始方式
# brightness_ontrast_ratio_2(img_path,1.2,10)# cv 方式
# img_HSI_S(img_path)
resetColor(img_path)