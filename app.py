import numpy as np, streamlit as st, joblib, tensorflow as tf
from PIL import Image
from skimage.feature import hog
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as rn_preprocess

st.set_page_config(page_title='Banana Ripeness Classifier', page_icon='🍌', layout='wide')
CLASSES = ['overripe', 'ripe', 'rotten', 'unripe']  # ลำดับ label ตอนเทรน (0-3)
TH = {'overripe': 'สุกงอม', 'ripe': 'สุก', 'rotten': 'เน่า', 'unripe': 'ดิบ'}
IMG, FS = (224, 224), 64  # ขนาดภาพของ CNN / ของ SVM-KNN
MODELS = [('CNN M4 (ออกแบบเอง)', 'test 98.2%'), ('ResNet50 (transfer learning)', 'test 98.9%'), ('SVM (สี HSV + HOG)', 'test 96.6%'), ('KNN K=3 (สี HSV + HOG)', 'test 88.6%')]

@st.cache_resource  # โหลดโมเดลครั้งเดียว ไม่โหลดใหม่ทุกครั้งที่อัปโหลดรูป
def load_models():
    return (keras.models.load_model('model4.keras'), keras.models.load_model('resnet50.keras'), joblib.load('svm.joblib'), joblib.load('knn.joblib'))

def feat(img):  # สี HSV histogram 96 ค่า + HOG 1,764 ค่า เหมือนตอนเทรน SVM/KNN
    small = img.resize((FS, FS))
    hsv = np.array(small.convert('HSV'))
    color = np.concatenate([np.histogram(hsv[..., k], bins=32, range=(0, 256))[0] / (FS * FS) for k in range(3)])
    shape = hog(np.array(small.convert('L')), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return np.concatenate([color, shape]).astype(np.float32)[None]

def predict(img):  # M4: หาร 255 | ResNet50: preprocess_input | SVM/KNN: feat()
    m4, rn, svm, knn = load_models()
    img = img.convert('RGB')
    x = tf.image.resize(np.array(img, dtype=np.float32), IMG).numpy()[None]
    f = feat(img)
    return [m4.predict(x / 255.0, verbose=0)[0], rn.predict(rn_preprocess(x.copy()), verbose=0)[0], svm.predict_proba(f)[0], knn.predict_proba(f)[0]]

def show(box, name, score, p):
    best = int(np.argmax(p))
    box.markdown(f'**{name}** · {score}')
    box.markdown(f'### {CLASSES[best]} ({TH[CLASSES[best]]})')
    for i in np.argsort(p)[::-1]:
        box.progress(float(p[i]), text=f'{CLASSES[i]} ({TH[CLASSES[i]]}) {p[i]:.0%}')

st.title('ทายความสุกของกล้วยด้วย 4 โมเดล')
st.caption('อัปโหลดรูปใหม่ได้เรื่อยๆ รูปเก่าจะถูกแทนที่และทายผลทันที · โมเดลรู้จักแค่ 4 คลาส (ดิบ / สุก / สุกงอม / เน่า) ถ้าใส่รูปที่ไม่ใช่กล้วยก็จะยังตอบ 1 ใน 4 คลาส · Term project 517433 Deep Learning for Computer Vision')
left, right = st.columns([2, 3], gap='large')
up = left.file_uploader('+ อัปโหลดรูปใหม่', type=['jpg', 'jpeg', 'png', 'webp'])
ex = left.radio('หรือเลือกรูปตัวอย่างจาก test set', ['ไม่ใช้'] + [f'{c} ({TH[c]})' for c in CLASSES], horizontal=True)
img = Image.open(up) if up else (Image.open('examples/' + ex.split()[0] + '.jpg') if ex != 'ไม่ใช้' else None)
if img is None:
    right.info('อัปโหลดรูปกล้วย หรือเลือกรูปตัวอย่างทางซ้าย แล้วผลของทั้ง 4 โมเดลจะขึ้นตรงนี้')
else:
    left.image(img, caption='รูปที่กำลังทาย')
    with st.spinner('กำลังทาย...'):
        probs = predict(img)
    boxes = right.columns(2) + right.columns(2)
    for box, (name, score), p in zip(boxes, MODELS, probs):
        show(box.container(border=True), name, score, p)
