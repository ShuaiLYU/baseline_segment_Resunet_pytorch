# import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


def plt_show_img(image,title=None):
    assert image is not None
    plt.figure()
    if title is not None:
        plt.title(title)
    plt.imshow(image,cmap="gray")
    plt.show()

def plt_show_imgs(imgs,title=None,):
    assert isinstance(imgs,(list,tuple))
    plt.figure()
    length=len(imgs)
    for i in range(length):
        plt.subplot(1,length,i+1)
        plt.imshow(imgs[i], cmap="gray")
       # plt.imshow(imgs[i],cmap="gray")
    plt.show()

def show_rects_on_img(img,rects,tittle=""):
    """
    :param img_path:
    :param rects:  (x,y,w,h）
    :param tittle:
    :return:
    """

    plt.figure(figsize=img.shape[:2])
    plt.imshow(img)
    plt.title(tittle)
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    for rect in rects:
        plt.gca().add_patch(plt.Rectangle(xy=(rect[0],rect[1]),width=rect[2],height=rect[3],))
    #                                       fill=False, edgecolor="red",linewidth=1))
    # for bbox,category_id,category in zip(bboxs,category_ids,categorys):
    #     """
    #     当前的图表和子图可以使用plt.gcf()和plt.gca()获得，分别表示GetCurrentFigure和GetCurrentAxes。
    #     在pyplot模块中，许多函数都是对当前的Figure或Axes对象进行处理，比如说：plt.plot()实际上会通过plt.gca()
    #     获得当前的Axes对象ax，然后再调用ax.plot()方法实现真正的绘图。
    #     """
    #
    #     plt.gca().add_patch(plt.Rectangle(xy=(bbox[0],bbox[1]),width=bbox[2],height=bbox[3],
    #                                       fill=False, edgecolor="red",linewidth=1))
    #     plt.text(x=bbox[0],y=bbox[1],s=category_id,ha='center',va='bottom',fontsize=10,color='red')
    plt.show()