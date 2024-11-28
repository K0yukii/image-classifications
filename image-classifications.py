import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout, concatenate
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Add, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_uniform, constant
from tensorflow.keras.activations import selu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.preprocessing import MultiLabelBinarizer

"""## **Загрузка и настройка набора данных**
Используется набор данных, в котором приблизительно 50000 фотографий растений. Набор данных содержит 3 каталога.
Мы используем утилиту `!pip install -U kaggle`, которая позволяет нам эффективно загружать набор данных с сайта kaggle. [1]
"""

excluded = ['Blueberry___healthy','Peach___healthy','Peach___Bacterial_spot','Raspberry___healthy',
            'Soybean___healthy','Squash___Powdery_mildew']

!pip install -U kaggle

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/

!chmod 600 /content/kaggle.json

!kaggle datasets download -d asheniranga/augmented-leaf-dataset

!mkdir ~/asheniranga/augmented-leaf-dataset

!kaggle datasets download -d asheniranga/augmented-leaf-dataset

!kaggle datasets download -d nizorogbezuode/rice-leaf-images
!kaggle datasets download -d vipoooool/new-plant-diseases-dataset

"""## **Настройка набора данных**
Выбираем каталоги датасета для обучения

"""

'/content/rice-leaf-images.zip'  '/content/new-plant-diseases-dataset.zip'  '/content/augmented-leaf-dataset.zip'

import zipfile
zipFile = '/content/augmented-leaf-dataset.zip'
directory_to_extract_to = '/content/augmented-leaf-dataset/'
with zipfile.ZipFile(zipFile, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

import zipfile
zipFile = '/content/new-plant-diseases-dataset.zip'
directory_to_extract_to = '/content/new-plant-diseases-dataset/'
with zipfile.ZipFile(zipFile, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

import zipfile
zipFile = '/content/rice-leaf-images.zip'
directory_to_extract_to = '/content/rice-leaf-images/'
with zipfile.ZipFile(zipFile, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

"""## **Импорт TensorFlow и других библиотек**

Импорт дополнительных библиотек
"""

import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout, concatenate
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Add, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_uniform, constant
from tensorflow.keras.activations import selu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.preprocessing import MultiLabelBinarizer

"""## **Настройка набора данных под модель**
Используем предварительную выборку с буферизацией, чтобы была возможность получать данные с kaggle, не блокируя ввод-вывод. И подсчет колличества фотографий. [2]
"""

excluded = ['Blueberry___healthy','Peach___healthy','Peach___Bacterial_spot','Raspberry___healthy',
            'Soybean___healthy','Squash___Powdery_mildew']

train_dir = '/content/augmented-leaf-dataset/augmented'

if not os.path.exists('train'):
    os.mkdir('train')

for path in os.listdir(train_dir):
    src_class = os.path.join(train_dir, path)
    dst_class = path.lower().strip()

    if not os.path.exists(os.path.join('train',dst_class)):
        os.mkdir(os.path.join('train',dst_class))

    for file in os.listdir(src_class):
        shutil.copyfile(src=os.path.join(src_class, file),
                        dst=os.path.join('train',dst_class,file))

train_dir = '/content/new-plant-diseases-dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'

if not os.path.exists('train'):
    os.mkdir('train')

for path in os.listdir(train_dir):
    if path not in excluded:
        src_class = os.path.join(train_dir, path)
        dst_class = path.replace('___',' ').replace('_', ' ').lower().strip()

        if not os.path.exists(os.path.join('train',dst_class)):
            os.mkdir(os.path.join('train',dst_class))

        for file in os.listdir(src_class):
            shutil.copyfile(src=os.path.join(src_class, file),
                            dst=os.path.join('train',dst_class,file))

train_dir = '/content/rice-leaf-images/rice_images'

if not os.path.exists('train'):
    os.mkdir('train')

for path in os.listdir(train_dir):
    src_class = os.path.join(train_dir, path)
    dst_class = 'rice '+path.replace('_', '').lower()

    if not os.path.exists(os.path.join('train',dst_class)):
        os.mkdir(os.path.join('train',dst_class))

    files = os.listdir(src_class)

    for i in range(int(np.ceil(len(files)*0.9))):
        shutil.copyfile(src=os.path.join(src_class, files[i]),
                        dst=os.path.join('train',dst_class,files[i]))

images_path = os.listdir('train/')

for path in images_path:
    print(f"{path}: {len(os.listdir(os.path.join('train', path)))}")

"""## **Визуализация и подготовка изображений к обучению**
Первые наборы изображений из обучающего набора данных:
"""

images_path = os.listdir('train/')
figure, axes = plt.subplots(nrows=42, ncols=3, figsize=[6,72], dpi=100)
axes = axes.ravel()
files = []

for path in images_path:
    files.append([os.path.join('train',path,file) for file in os.listdir(os.path.join('train', path))[:3]])

files = np.asarray(files).ravel()

for i in range(len(files)):
    axes[i].imshow(load_img(files[i]))

"""Определим параметры для загружаемых изображений [3]:"""

generator = ImageDataGenerator(rescale=1. / 255,
                               rotation_range=45,
                               horizontal_flip=True,
                               vertical_flip=True,
                               validation_split=0.2
                              )

"""Разделим наш набор данных на 80% изображений для обучения, и 20% для тестирования."""

check_pointer = ModelCheckpoint(filepath='wb_1_best_checkpoint_only.hdf5',
                                save_best_only=True)

early_stop = EarlyStopping(monitor='val_loss',
                           patience=20,
                           min_delta=0,
                           restore_best_weights=True)

reduce_lr = [ReduceLROnPlateau(monitor='val_loss')]

"""Уменьшающий слой,
слой извлечения,
проекционный слой,
вывод
"""

def inception_module(x, filter_1x1, filter_3x3_reduce, filter_3x3, filter_5x5_reduce, filter_5x5, filters_pool_proj,
                     name=None):
    # reduction layer
    conv_3x3_reducer = Conv2D(filters=filter_3x3_reduce, kernel_size=(1, 1), activation=selu, padding='same')(x)
    conv_5x5_reducer = Conv2D(filters=filter_5x5_reduce, kernel_size=(1, 1), activation=selu, padding='same')(x)
    pool_3x3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)

    # extraction layer
    conv_3x3 = Conv2D(filters=filter_3x3, kernel_size=(3, 3), activation=selu, padding='same')(conv_3x3_reducer)
    conv_5x5 = Conv2D(filters=filter_5x5, kernel_size=(5, 5), activation=selu, padding='same')(conv_5x5_reducer)
    conv_1x1_proj = Conv2D(filters=filters_pool_proj, kernel_size=(1, 1), activation=selu, padding='same')(pool_3x3)

    # projection layer
    proj = Conv2D(filters=filter_1x1, kernel_size=(1, 1), activation=selu, padding='same')(x)

    # output
    x = concatenate([proj, conv_1x1_proj, conv_3x3, conv_5x5], axis=3, name=name)

    return x

def bottleneck_residual_block(x, kernel_size, filters, reduce=False, s=1):
    f1, f2, f3 = filters
    x_shortcut = x

    if reduce:
        x_shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=(s, s), padding='same', activation=relu)(x_shortcut)
        x_shortcut = BatchNormalization()(x_shortcut)

        x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(s, s), padding='same')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation(relu)(x)
    else:
        # first component of the main path
        x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(s, s), padding='valid')(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation(relu)(x)

    # second component of the main path
    x = Conv2D(filters=f2, kernel_size=kernel_size, strides=(s, s), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(relu)(x)

    # third component of the main path
    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(s, s), padding='same')(x)
    x = BatchNormalization(axis=3)(x)

    x = Add()([x, x_shortcut])
    x = Activation(relu)(x)

    return x

"""Разделение изображений на блоки"""

def wide_block(x, filters, name):
    f1, f2, f3 = filters
    prev_layer = x

    # wide shortcut
    conv_1x1_shortcut = Conv2D(filters=f1, kernel_size=(1, 1), activation=selu, padding='same')(prev_layer)
    conv_1x1_shortcut = BatchNormalization(axis=3)(conv_1x1_shortcut)

    # sub-deep path
    conv_5x5 = Conv2D(filters=f2, kernel_size=(5, 5), strides=(1, 1), activation=selu, padding='same')(prev_layer)
    conv_5x5 = BatchNormalization(axis=3)(conv_5x5)
    conv_3x3 = Conv2D(filters=f3, kernel_size=(3, 3), strides=(1, 1), activation=selu, padding='same')(conv_5x5)
    conv_3x3 = BatchNormalization(axis=3)(conv_3x3)

    # concatenation block 1
    sub_deep_block = concatenate([prev_layer, conv_3x3], axis=3)
    sub_deep_block = BatchNormalization(axis=3)(sub_deep_block)

    # concatenation block 2
    final_concat = concatenate([conv_1x1_shortcut, sub_deep_block], axis=3, name=name)

    return final_concat

"""Классификация изображений датасета по размерам"""

input_layer = Input(shape=(72,72,3))

x = Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), activation=selu)(input_layer)
x = BatchNormalization()(x)

# conv_3x3
x = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), activation=selu)(x)
x = BatchNormalization()(x)

x = wide_block(x,[128,16,32],'dw_1a')
x = wide_block(x,[128,16,32],'dw_1b')
x = wide_block(x,[128,16,32],'dw_1c')

x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
x = BatchNormalization()(x)

x = wide_block(x,[256,32,64],'dw_2a')
x = wide_block(x,[256,32,64],'dw_2b')
x = wide_block(x,[256,32,64],'dw_2c')

x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
x = BatchNormalization()(x)

x = wide_block(x,[512,64,128],'dw_3a')
x = wide_block(x,[512,64,128],'dw_3b')
x = wide_block(x,[512,64,128],'dw_3c')
#
x = MaxPooling2D(pool_size=(7,7), strides=(2,2))(x)
x = BatchNormalization()(x)

# classification
x = Flatten()(x)
x = Dense(units=512, activation=selu)(x)
x = Dropout(0.5)(x)
exp_output = Dense(units=42, activation=softmax)(x)

exp_conv = Model(input_layer,exp_output)
exp_conv.summary()

"""## **Построение модели нейроной сети**
[3]

"""

plot_model(model=exp_conv,
           to_file='Inception.png',
           show_shapes=True,
           show_dtype=True,
           show_layer_names=True)

exp_conv.compile(optimizer=Adam(),
                 loss=categorical_crossentropy,
                 metrics=['accuracy'])

train_gen = generator.flow_from_directory('train',
                                          target_size=(72,72),
                                          batch_size=348,
                                          subset='training')
validation_gen = generator.flow_from_directory('train',
                                               target_size=(72,72),
                                               batch_size=348,
                                               subset='validation')

"""## **Подключение к гугл диску**

Загрузка весов модели из диска если не требуется обучение модели
"""

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
from PIL import Image
import io
from google.colab import drive
drive.mount('/content/drive')

drive_path = '/content/drive/MyDrive/model'

"""## **Обучение модели**

"""

callbacks = [check_pointer,early_stop,reduce_lr]
exp_conv_history = exp_conv.fit(train_gen,
                                epochs=22,
                                callbacks=callbacks,
                                validation_data=validation_gen)



"""После применения увеличения данных и `tf.keras.layers.Dropout` переобучение меньше, чем раньше, а точность обучения и проверки более согласована:"""

figure, axes = plt.subplots(nrows=1, ncols=2, figsize=[18, 6], dpi=300)
axes = axes.ravel()
epochs = list(range(len(exp_conv_history.history['loss'])))

sns.lineplot(x=epochs, y=exp_conv_history.history['loss'], ax=axes[0], label='loss')
sns.lineplot(x=epochs, y=exp_conv_history.history['val_loss'], ax=axes[0], label='val loss')
sns.lineplot(x=epochs, y=exp_conv_history.history['accuracy'], ax=axes[1], label='accuracy')
sns.lineplot(x=epochs, y=exp_conv_history.history['val_accuracy'], ax=axes[1], label='val accuracy')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('loss')
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('accuracy')
plt.savefig('Inception_train_history.png')
plt.show()

"""
скорость обучения"""

figure, axes = plt.subplots(nrows=1, ncols=2, figsize=[12, 6], dpi=300)
axes = axes.ravel()

sns.lineplot(x=epochs, y=exp_conv_history.history['lr'], ax=axes[0], label='learning rate')
sns.lineplot(x=exp_conv_history.history['lr'], y=exp_conv_history.history['val_accuracy'], ax=axes[1], label='accuracy & lr')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('learning rate')
axes[1].set_xlabel('learning rate')
axes[1].set_ylabel('accuracy')

plt.savefig('Inception_lr_history.png')
plt.show()

"""## **Сохранение модели на гугл диск**

"""

exp_conv.save_weights(r'/content/drive/MyDrive/model_V2')
exp_conv.save(r'/content/drive/MyDrive/model_V2')

"""Чтобы в дальнейшем не тратить время на повторный запуск обучения, рекомендуется подключить свой Гугл Диск и сохранить свою модель: [5]"""

test = []
label = []

for file in os.listdir('/content/new-plant-diseases-dataset/test/test'):
    image = load_img(os.path.join('/content/new-plant-diseases-dataset/test/test',file), target_size=(72,72))
    test.append(img_to_array(image))
    label.append(file.split('.')[0])

"""## **Загрузка модели с гугл диска**
Добавим возможность добавления своих изображений и используем нашу модель для

классификации изображений, которых не было в наборах для обучения и тестирования.

"""

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
from PIL import Image
import io
from google.colab import drive
drive.mount('/content/drive')

drive_path = '/content/drive/MyDrive/model'

from PIL import Image
import io
import itertools
import os
import pathlib
import IPython

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import itertools
from google.colab import files

from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout, concatenate
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Add, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_uniform, constant
from tensorflow.keras.activations import selu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
images_path = ['apple apple scab',
 'apple black rot',
 'apple cedar apple rust',
 'apple healthy',
 'banana healthy',
 'banana segatoka',
 'banana xamthomonas',
 'cherry (including sour) healthy',
 'cherry (including sour) powdery mildew',
 'corn (maize) cercospora leaf spot gray leaf spot',
 'corn (maize) common rust',
 'corn (maize) healthy',
 'corn (maize) northern leaf blight',
 'grape black rot',
 'grape esca (black measles)',
 'grape healthy',
 'grape leaf blight (isariopsis leaf spot)',
 'orange haunglongbing (citrus greening)',
 'pepper, bell bacterial spot',
 'pepper, bell healthy',
 'potato early blight',
 'potato healthy',
 'potato late blight',
 'rice brownspot',
 'rice healthy',
 'rice hispa',
 'rice leafblast',
 'strawberry healthy',
 'strawberry leaf scorch',
 'tea leaf blight',
 'tea red leaf spot',
 'tea red scab',
 'tomato bacterial spot',
 'tomato early blight',
 'tomato healthy',
 'tomato late blight',
 'tomato leaf mold',
 'tomato septoria leaf spot',
 'tomato spider mites two-spotted spider mite',
 'tomato target spot',
 'tomato tomato mosaic virus',
 'tomato tomato yellow leaf curl virus']

drive_path = '/content/drive/MyDrive/model_V2'



model = tf.keras.models.load_model(drive_path)

"""## **Загрузка своих изображений и распознавание модели**

* Выбираем любой способ загрузки изображений из 2 ниже указанных.
* Загрузка изображений с помощью ссылки
"""

print('Вставьте ссылку на изображение')
url = str(input())
user_image = tf.keras.utils.get_file(origin=url,fname='/content/user_image', untar=True)
user_image
!mv /content/user_image.tar.gz /content/user_image
img_height, img_width = [72,72]
img = tf.keras.utils.load_img(
    user_image, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
normalized_image_array = (img_array.astype(np.float32) / 255.)

predictions = model.predict(normalized_image_array)
score = tf.nn.softmax(predictions[0])

print(
    "На этой фотографии {} с точностью {:.2f} процентов.\n"
    .format(images_path[np.argmax(score)], 100 * np.max(score))
)

IPython.display.Image(url, width = 250)

"""* Загрузка изображений файлом с названием "mod.jpg"
"""

from google.colab import files
!rm '/content/mod.jpg'
uploaded = files.upload()
for file in os.listdir('/content/'):
    image = load_img(os.path.join('mod.jpg'), target_size=(72,72))

img_bytes = list(uploaded.values())[0]
img = Image.open(io.BytesIO(img_bytes))
img = img.convert('RGB')
img = img.resize((72,72), Image.NEAREST)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
normalized_image_array = (img_array.astype(np.float32) / 255.)

predictions = model.predict(normalized_image_array)
score = tf.nn.softmax(predictions[0])
print(
    "На этой фотографии {} \n"
    .format(images_path[np.argmax(score)])
)
images_path = os.listdir('/content/')
plt.figure(figsize=(10, 10))
ax = plt.subplot(3, 3,1)
plt.imshow(img)
plt.axis("off")

"""## **Результат тестового набора**
Полученные данные из тестогого набора которым оценивается в сообществе точность обучения модели
"""

train_gen.class_indices

test = np.asarray(test)/255.

pred = exp_conv.predict(test)

pred_labels = [list(train_gen.class_indices.keys())[list(train_gen.class_indices.values()).index(pred_)] for pred_ in np.argmax(pred,axis=1)]

"""Результат модели"""

pd.options.display.max_rows = 76
compare = pd.DataFrame(np.asarray(label).reshape(-1,1), columns=['truth'])
compare['prediction'] = np.asarray(pred_labels)
compare
