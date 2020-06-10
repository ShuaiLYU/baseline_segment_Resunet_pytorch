#-*-coding:utf-8-*-
import os
import cv2
import numpy as np

def sub_bp_MOG2(img):
	"""
	:param img:   输入图像
	:return:返回
	"""
	mog = cv2.createBackgroundSubtractorMOG2()
	return mog.apply(img,None,0.01)

def cv_equalizeHist(img):
	"""
	:param img: 输入图像
	:return:  直方图均衡后的图像
	"""
	return cv2.equalizeHist(img)


def high_pass_fft(img,filter_size=None,power_thred=None):
    assert filter_size!=None or power_thred!=None
    if(filter_size !=None and power_thred !=None):
        raise Exception("filter_size and power_thred are incompatible!")
    img_float32 = np.float32(img)
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    # 将低频信息转换至图像中心
    dft_shift = np.fft.fftshift(dft)
    if power_thred !=None:
        # # 获取图像尺寸 与 中心坐标
        features = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])/np.sqrt(img.shape[0]*img.shape[1])
        mask = np.where(features > power_thred, 1, 0)[:, :, np.newaxis]
    if filter_size!=None:
        crow, ccol = int(img.shape[0] / 2), int(img.shape[1] / 2) # 求得图像的中心点位置
        mask = np.zeros((img.shape[0], img.shape[1], 2), np.uint8)
        mask[crow-filter_size:crow+filter_size, ccol-filter_size:ccol+filter_size] = 1
    # 掩码与傅里叶图像按位相乘  去除低频区域
    fshift = dft_shift * mask#
    # 之前把低频转换到了图像中间，现在需要重新转换回去
    f_ishift = np.fft.ifftshift(fshift)
    # 傅里叶逆变换
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back=(img_back-np.min(img_back))/(np.max(img_back)-np.min(img_back))*255
    return  mask[:, :, 0],img_back
def split(image,num,axis=1,offset=0):
    assert  axis==0 or axis==1
    h, w = image.shape
    if axis==0:
        h=(h+offset)//num-offset
        return  [image[i*(h+offset):i*(h+offset)+h,:]  for  i  in  range(num)]
    if axis==1:
        w =(w+offset)//num-offset

        return [image[:,i * (w+offset):i * (w+offset)+w] for i in range(num)]

def mask2rect(mask):
	cols,rows=np.where(mask>0)
	col1=np.amin(cols)
	col2 = np.amax(cols)
	row1=np.amin(rows)
	row2 = np.amax(rows)
	return (row1,col1,row2-row1,col2-col1)



def cv_dilate(mask, ksize=5, struct="ellipse"):
	assert struct in ["rect", "ellipse"]
	if struct == "rect": struct = cv2.MORPH_RECT
	if struct == "ellipse": struct = cv2.MORPH_ELLIPSE
	elment = cv2.getStructuringElement(struct, (ksize, ksize))
	mask_dilate = cv2.morphologyEx(mask, cv2.MORPH_DILATE, elment)
	return mask_dilate

def show_cams_on_images(img_batch, mask_batch,filenames,save_dirs):
	if img_batch.ndim!=4:img_batch=img_batch.unsqueeze(0)
	if img_batch.shape[-1] != 3: raise Exception("image[{}] must be RGB!".format(img_batch.shape))
	mask_batch=mask_batch.squeeze(1)
	batch=  len(img_batch) if isinstance(img_batch, list) else img_batch.shape[0]
	img_height,img_width=img_batch.shape[1:3]
	save_dirs=[save_dirs]*batch if not isinstance(save_dirs, list) else save_dirs
	for i, filename in enumerate(filenames):
		save_dir=save_dirs[i]
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		#filename = str(filename).split("'")[-2].replace("/","_")
		filename=filename.decode("utf-8")  if not isinstance(filename, str) else filename
		heatmap = cv2.applyColorMap(np.uint8(255 * mask_batch[i]), cv2.COLORMAP_JET)
		heatmap=cv2.resize(heatmap,(img_width,img_height))
		heatmap = np.float32(heatmap) / 255
		#img_show=cv2.cvtColor( np.uint8(255 * img_batch[i]), cv2.COLOR_GRAY2BGR)
		img_show =np.uint8(255 * img_batch[i])
		cam = heatmap + np.float32(img_show)/255
		#cam=np.float32(img_show) / 255
		cam = cam / np.max(cam)
		cam=np.uint8(255 * cam)
		visualization_path = os.path.join(save_dir,filename)
		print("write to {}".format(visualization_path))
		cv2.imwrite(visualization_path, cam)

def grub_cut_on_mask(img,mask,thred=0.2,n_iter=1):
	mask=np.where(mask>thred,3,0).astype(np.uint8)
	# if mask.sum()==0:
	# 	return mask
	rect=(0,0,0,0)
	# rect=mask2rect(mask)
	#assert  mask.sum()>0
	mode = cv2.GC_INIT_WITH_MASK
	# mode=cv2.GC_INIT_WITH_RECT
	bgdModel = np.zeros((1, 65), np.float64)
	fgdModel = np.zeros((1, 65), np.float64)
	try:
		cv2.grabCut(img, mask, rect, bgdModel, fgdModel, n_iter, mode=mode)
	except Exception as error:
		pass
			#print("错误:{}".format(error))
	mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
	return mask


def cv_resize(img,size=None,fxy=None):
	"""
	:param img: 输入图像
	:param size: 目标尺寸 (x,y)
	:param fxy:  放缩比例 (fx,fy) 和 size参数互斥
	:return:  resize 之后的图片
	"""
	if size==None and fxy!=None:
		assert isinstance(fxy, tuple) and len(fxy) == 2
		return cv2.resize(img, (0, 0), fx=fxy[0], fy=fxy[1])
	elif fxy==None and size!=None:
		assert isinstance(size, tuple) and len(fxy) == 2
		return cv2.resize(img, size)
	else:
		raise Exception("custom error！")

def cv_imread(file_path, flag=-1):
	"""
	解决cv包含中文路径的问题
	:param file_path:  路径
	:param flag:
	:return:
	"""
	cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)
	return cv_img

def show_cams_on_images(img_batch, mask_batch,filenames,save_dirs):
	if img_batch.ndim!=4:img_batch=img_batch.unsqueeze(0)
	if img_batch.shape[-1] != 3: raise Exception("image[{}] must be RGB!".format(img_batch.shape))
	mask_batch=mask_batch.squeeze(1)
	batch=  len(img_batch) if isinstance(img_batch, list) else img_batch.shape[0]
	img_height,img_width=img_batch.shape[1:3]
	save_dirs=[save_dirs]*batch if not isinstance(save_dirs, list) else save_dirs
	for i, filename in enumerate(filenames):
		save_dir=save_dirs[i]
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		#filename = str(filename).split("'")[-2].replace("/","_")
		filename=filename.decode("utf-8")  if not isinstance(filename, str) else filename
		heatmap = cv2.applyColorMap(np.uint8(255 * mask_batch[i]), cv2.COLORMAP_JET)
		heatmap=cv2.resize(heatmap,(img_width,img_height))
		heatmap = np.float32(heatmap) / 255
		#img_show=cv2.cvtColor( np.uint8(255 * img_batch[i]), cv2.COLOR_GRAY2BGR)
		img_show =np.uint8(255 * img_batch[i])
		cam = heatmap + np.float32(img_show)/255
		#cam=np.float32(img_show) / 255
		cam = cam / np.max(cam)
		cam=np.uint8(255 * cam)
		visualization_path = os.path.join(save_dir,filename)
		print("write to {}".format(visualization_path))
		cv2.imwrite(visualization_path, cam)


def Ostu(array):
	array = np.array(array * 255, dtype=np.uint8)
	best_threshold, binary_output = cv2.threshold(array, 100, 1, cv2.THRESH_BINARY)  # cv2.THRESH_OTSU
	area = np.sum(np.array(binary_output))
	predict =(area > 1)
	return predict
def cv_open(mask,ksize=5,struct="ellipse"):
	assert struct in ["rect","ellipse"]
	if struct=="rect": struct=cv2.MORPH_RECT
	if struct=="ellipse": struct=cv2.MORPH_ELLIPSE
	elment=cv2.getStructuringElement(struct, (ksize, ksize))
	mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, elment)
	return mask_open

def cv_close(mask,ksize=5,struct="ellipse"):
	assert struct in ["rect","ellipse"]
	if struct=="rect": struct=cv2.MORPH_RECT
	if struct=="ellipse": struct=cv2.MORPH_ELLIPSE
	elment=cv2.getStructuringElement(struct, (ksize, ksize))
	mask_open = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, elment)
	return mask_open
def cv_dyn_threshold(img,thred,ksize=15):
	img_blur=cv2.blur(img,ksize=(ksize,ksize))
	arr_blur=np.array(img_blur,dtype=np.float)
	arr=np.array(img,dtype=np.float)
	mask=np.where(np.abs(arr-arr_blur)>thred,1,0)
	return mask.astype(np.uint8)


def origin_LBP(img):
	dst = np.zeros(img.shape, dtype=img.dtype)
	h, w = img.shape
	for i in range(1, h - 1):
		for j in range(1, w - 1):
			center = img[i][j]
			code = 0
			code |= (img[i - 1][j - 1] >= center) << (np.uint8)(7)
			code |= (img[i - 1][j] >= center) << (np.uint8)(6)
			code |= (img[i - 1][j + 1] >= center) << (np.uint8)(5)
			code |= (img[i][j + 1] >= center) << (np.uint8)(4)
			code |= (img[i + 1][j + 1] >= center) << (np.uint8)(3)
			code |= (img[i + 1][j] >= center) << (np.uint8)(2)
			code |= (img[i + 1][j - 1] >= center) << (np.uint8)(1)
			code |= (img[i][j - 1] >= center) << (np.uint8)(0)

			dst[i - 1][j - 1] = code
	return dst


def match(img1,img2,filter_sz=3):
	dst = np.zeros(img1.shape, dtype=img1.dtype)
	h, w = img1.shape
	for i in range(0, h ):
		for j in range(0, w ):
			val=255
			for k in range(filter_sz):
				y=i+k-(filter_sz//2)
				if y<0 or y>h-1:
					continue
				for l in range(filter_sz):
					x = j + l - (filter_sz // 2)
					if x < 0 or x > w - 1:
						continue
					val=min(val,abs(img1[i,j]-img2[y,x]))
			dst[i,j]=val
	dst=dst.astype(np.float)
	dst=(dst-np.min(dst))/(np.max(dst)-np.min(dst))
	return (dst*255).astype(np.uint8)

if __name__=="__main__":
	dir=r"C:\Datasets\KolektorSDD"
