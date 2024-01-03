# Importamos librerías
import cv2
import os
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt

  # Función para el redimensionamiento y la obtención de valores de imágenes 
def fruit_data(path,h,w):
	data_set = []  
	label = ["dp","dn"]  
	  # Ciclo para la extracción de información de cada carpeta
	for form in label:
		c_num = label.index(form) 
		d_path = os.path.join(path,form)  
		  # Ciclo para cada una de las imágenes dentro de la carpeta
		for file in os.listdir(d_path):
			img = cv2.imread(os.path.join(d_path,file), cv2.IMREAD_GRAYSCALE) 
			img = cv2.resize(img,(h,w))  
			data_set.append([img,c_num])
	  # Declaración de nuevas matrices
	X=[] 
	Y=[]
	  # Ciclo para guardar los valores obtenidos anteriormente
	for info, label in data_set:
		X.append(info)
		Y.append(label)
	X = np.array(X).reshape(-1,h,w)
	Y = np.array(Y)
	return X,Y

  # To training
A,B = fruit_data("training2",50,50)

  # Primera capa de convolución 
class CNN:
	
	def __init__(self, num_filters,filter_size):
		self.num_filters = num_filters
		self.filter_size = filter_size
		self.conv_filter=np.random.randn(num_filters,filter_size,filter_size)/(filter_size*filter_size)

	def image_region(self, image):
		h = image.shape[0]
		w = image.shape[1]

		self.image = image
		for i in range(h-self.filter_size + 1):
			for j in range(w - self.filter_size + 1):
				image_patch = image[i : (i + self.filter_size), j : (j + self.filter_size)]
				yield image_patch, i, j 
	def forward_prop(self, image):
		h,w = image.shape
		a = h - self.filter_size + 1
		b = w - self.filter_size + 1
		conv_out = np.zeros((a,b,self.num_filters))
		for image_patch,i,j in self.image_region(image):
			conv_out[i,j]=np.sum(image_patch*self.conv_filter,axis= (1,2))
		return conv_out 

	def back_prop(self, dL_dout, learning_rate):
		dL_dF_params = np.zeros(self.conv_filter.shape)
		for image_patch, i, j in self.image_region(self.image):
			for k in range(self.num_filters):
				dL_dF_params[k] += image_patch*dL_dout[i,j,k]

		self.conv_filter -= learning_rate*dL_dF_params
		return dL_dF_params

  # Función para la agrupación máxima con un tammaño definido
class Max_Pool:

	def __init__(self, filter_size):
		self.filter_size = filter_size 
	
	def image_region(self, image):
		new_h = image.shape[0]//self.filter_size
		new_w = image.shape[1]//self.filter_size
		self.image = image
		for i in range(new_h):
			for j in range(new_w):
				a = i*self.filter_size
				b = i*self.filter_size + self.filter_size
				c = j*self.filter_size
				d = j*self.filter_size + self.filter_size
				image_patch = image[a:b,c:d]
				yield image_patch, i, j
	
	def forward_prop(self, image):
		height, widht, num_filters = image.shape
		output = np.zeros((height//self.filter_size, widht//self.filter_size, num_filters))
		
		for image_patch, i, j in self.image_region(image):
			output[i,j] = np.amax(image_patch, axis = (0,1))

		return output 

	def back_prop(self,dL_dout):
		dL_dmax_pool = np.zeros(self.image.shape)
		for image_patch, i, j in self.image_region(self.image):
			h,w,num_filters = image_patch.shape
			maximun_val = np.amax(image_patch, axis = (0,1))

			for x in range(h):
				for y in range(w):
					for z in range(num_filters):
						if image_patch[x,y,z] == maximun_val[z]:
							dL_dmax_pool[i*self.filter_size + x, j*self.filter_size +y,z]=dL_dout[i,j,z]
			return dL_dmax_pool

# Capa para la activación de softmax
class Softmax:
	def __init__(self, input_node, sofmax_node):
		 # Se reudce la viarianza de los valores iniciales
		self.weight = np.random.randn(input_node,sofmax_node)/input_node
		self.bias = np.zeros(sofmax_node)
    
	def forward_prop(self, image):
		self.orig_im_shape = image.shape
		image_modified = image.flatten()
		self.modified_input = image_modified
		output_val = np.dot(image_modified, self.weight) + self.bias
		self.out = output_val
		exp_out = np.exp(output_val)
		val = exp_out/np.sum(exp_out, axis=0)
		return val
	def back_prop(self, dL_dout, learning_rate):
		for i, grad in enumerate(dL_dout):
			if grad == 0:
				continue

			transformation_eq = np.exp(self.out)
			S_total = np.sum(transformation_eq)

			dy_dz = -transformation_eq[i]*transformation_eq/(S_total**2)
			dy_dz[i] = transformation_eq[i]*(S_total - transformation_eq[i])/(S_total**2)

			dz_dw = self.modified_input
			dz_db = 1
			dz_d_input = self.weight

			dL_dz = grad*dy_dz

			dL_dw = dz_dw[np.newaxis].T@dL_dz[np.newaxis]
			dL_db = dL_dz*dz_db
			dL_d_input = dz_d_input@dL_dz

			self.weight-=learning_rate*dL_dw
			self.bias -= learning_rate*dL_db

			return dL_d_input.reshape(self.orig_im_shape)

# Extracción de valores de imágenes obtenidos anteriormente
train_images = A[:50] 
train_labels = B[:50] 

conv = CNN(8,3)                  
pool = Max_Pool(2)              
softmax = Softmax(24*24*8, 10)   

def forward(image, label):
 
  out = conv.forward_prop((image / 255) - 0.5)
  out = pool.forward_prop(out)
  out = softmax.forward_prop(out)

  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

  # Función para el entrenamiento completo de la imagen y etiqueta dada
def train(im, label, lr=.005):
  
  out, loss, acc = forward(im, label)

  gradient = np.zeros(20)
  gradient[label] = -1 / out[label]

  gradient = softmax.back_prop(gradient, lr)
  gradient = pool.back_prop(gradient)
  gradient = conv.back_prop(gradient, lr)

  return loss, acc

print('Red Neuronal Convolucional Iniciada')

# Entrenamiento con una época
for epoca in range(1):
  print('Epoca %d ' % (epoca + 1))

  # Se permuta los datos de entrenamiento
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  loss = 0
  num_correct = 0
  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i % 100 == 99:
      loss = 0
      num_correct = 0

    l, acc = train(im, label)
    loss += l
    num_correct += acc
    
# Test the CNN
print('\nInicio de prueba')
loss = 0
correct = 0
n=0

cap = cv2.VideoCapture(5,cv2.CAP_DSHOW)
data_frame = []

while True:
  ret,frame= cap.read()
  img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
  cv2.imshow("Detect",frame)
  
  img2=cv2.resize(img,(50,50))
  X = (img2)
  X=np.array(X).reshape(-1,50,50)
  Y = np.array([0])
  test_images = X
  test_labels = Y
  test_images = X[:1]
  test_labels = Y[:1]

  for im, label in zip(test_images,test_labels):
  	_, l, acc = forward(im, label)
  	loss += l
  	correct += acc

  	if acc == 1:
  		print("Caja detectada")
  	else:
  		print("Sin datos")
  k=cv2.waitKey(5)
  if k==27:
  	break

cv2.destroyAllWindows()
