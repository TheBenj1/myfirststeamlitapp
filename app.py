import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# תמיכה ב-HEIC (תמונות מאייפון)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # אם pillow-heif לא מותקן, פשוט נדלג

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage import filters

# ====================== Streamlit Config ======================
st.set_page_config(
    page_title="Image Segmentation Playground",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="expanded"
)

# כותרת יפה
st.title("🧫 Image Segmentation Playground")
st.markdown("###try different segmantation Watershed • K-Means • Otsu")

# ====================== Sidebar ======================
with st.sidebar:
    st.header("📤 העלאת תמונה")
    uploaded_file = st.file_uploader(
        "העלה תמונה", 
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp", "heic", "heif", "gif"]
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Pre-filter (תצוגה מקדימה)")
    prefilter_mode = st.selectbox(
        "סוג פילטר",
        ["None", "Gaussian", "Mean", "Canny"],
        index=1
    )

    sigma = st.slider("Sigma (ל-Gaussian/Mean)", 0.1, 10.0, 2.0, 0.1, key="sigma_prefilter")

    if prefilter_mode == "Canny":
        col1, col2 = st.columns(2)
        canny_low = col1.slider("Canny Low", 0, 255, 50, 5)
        canny_high = col2.slider("Canny High", 0, 255, 150, 5)
    else:
        canny_low, canny_high = None, None

    st.markdown("---")
    st.caption("הפילטר כאן משפיע רק על התצוגה המקדימה (לא על הסגמנטציה הסופית)")

# ====================== Session State ======================
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "filtered_image" not in st.session_state:
    st.session_state.filtered_image = None
if "final_result" not in st.session_state:
    st.session_state.final_result = None

# ====================== Helper Functions ======================
def apply_pre_filter(img, mode, sigma):
    if mode == "None":
        return img.copy()
    if mode == "Gaussian":
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if mode == "Mean":
        k = int(max(3, 2 * round(sigma) + 1)) | 1  # חייב להיות אי זוגי
        return cv2.blur(img, (k, k))
    return img.copy()

def preprocess_for_display(img_color, mode, sigma, low=None, high=None):
    if mode == "Canny":
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        edges = cv2.Canny(blurred, low or 50, high or 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    else:
        return apply_pre_filter(img_color, mode, sigma)

def run_watershed_classic(img_color, min_dist, shift_sp, shift_sr, inverted=False):
    shifted = cv2.pyrMeanShiftFiltering(img_color, shift_sp, shift_sr)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if inverted:
        thresh = 255 - thresh

    mask = thresh > 0
    if mask.sum() == 0:
        st.warning("מסכת Otsu ריקה לחלוטין")
        return np.zeros_like(img_color)

    D = ndi.distance_transform_edt(mask)
    local_max = peak_local_max(D, min_distance=min_dist, labels=mask)
    if len(local_max) == 0:
        st.warning(f"לא נמצאו פסגות מקומיות עם min_distance={min_dist}")
        return np.zeros_like(img_color)

    markers = ndi.label(peak_local_max(D, min_distance=min_dist, labels=mask))[0]
    labels = watershed(-D, markers, mask=mask)

    result = img_color.copy()
    rng = np.random.default_rng(42)
    for label in np.unique(labels):
        if label == 0:
            continue
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        result[labels == label] = color
    return result

def run_kmeans(img_color, k, sigma):
    filtered = apply_pre_filter(img_color, "Gaussian", sigma)
    data = filtered.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    return segmented.reshape(img_color.shape)

def run_otsu(img_color, inverted=False, prefilter_mode="Gaussian", sigma=2.0):
    filtered = apply_pre_filter(img_color, prefilter_mode, sigma)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    t = filters.threshold_otsu(gray)
    mask = (gray < t) if not inverted else (gray > t)
    mask_3c = np.stack([mask*255]*3, axis=-1).astype(np.uint8)
    return cv2.bitwise_and(img_color, mask_3c)

# ====================== Main App ======================
if uploaded_file is not None:
    # קריאת התמונה
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.session_state.uploaded_image = img_cv

    # הצגת התמונה המקורית
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="תמונה מקורית", use_column_width=True)
    with col2:
        if st.session_state.uploaded_image is not None:
            filtered = preprocess_for_display(
                st.session_state.uploaded_image,
                prefilter_mode,
                sigma,
                canny_low if prefilter_mode == "Canny" else None,
                canny_high if prefilter_mode == "Canny" else None
            )
            filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
            st.image(filtered_rgb, caption=f"תצוגה מקדימה: {prefilter_mode}", use_container_width=True)
            st.session_state.filtered_image = filtered

    st.markdown("---")
    st.header("🔬 בחירת שיטת סגמנטציה")

    method = st.selectbox(
        "בחר שיטה",
        ["Watershed Classic", "-Watershed Classic (הפוך)", "K-Means", "Otsu Thresholding", "-Otsu Thresholding (הפוך)"],
        index=0
    )

    col1, col2 = st.columns(2)

    with col1:
        if "Watershed" in method:
            st.subheader("Watershed Parameters")
            min_dist = st.slider("Min Distance", 5, 50, 10)
            sp = st.slider("MeanShift Spatial Window (sp)", 10, 100, 21)
            sr = st.slider("MeanShift Color Window (sr)", 10, 100, 51)
        elif method == "K-Means":
            st.subheader("K-Means Parameters")
            k = st.slider("מספר אשכולות (K)", 2, 20, 5)
            k_sigma = st.slider("Sigma (Gaussian prefilter)", 0.0, 10.0, 2.0, 0.1)
        elif "Otsu" in method:
            st.subheader("Otsu Parameters")
            otsu_sigma = st.slider("Sigma", 0.0, 10.0, 2.0, 0.1)
            otsu_filter = st.selectbox("סוג פילטר מקדים", ["Gaussian", "Mean"], index=0)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # רווח
        run_btn = st.button("🚀 הרץ סגמנטציה", type="primary", use_container_width=True)

    if run_btn or True:  # תמיד נריץ אחרי בחירת פרמטרים (אפשר גם להסיר את ה-"or True")
        with st.spinner("מעבד..."):
            img = st.session_state.uploaded_image

            if method == "Watershed Classic":
                result = run_watershed_classic(img, min_dist, sp, sr, inverted=False)
            elif method == "-Watershed Classic (הפוך)":
                result = run_watershed_classic(img, min_dist, sp, sr, inverted=True)
            elif method == "K-Means":
                result = run_kmeans(img, k, k_sigma)
            elif method == "Otsu Thresholding":
                result = run_otsu(img, inverted=False, prefilter_mode=otsu_filter, sigma=otsu_sigma)
            elif method == "-Otsu Thresholding (הפוך)":
                result = run_otsu(img, inverted=True, prefilter_mode=otsu_filter, sigma=otsu_sigma)

            st.session_state.final_result = result
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption=f"תוצאה: {method}", use_column_width=True)

            # כפתור הורדה
            _, buf = cv2.imencode(".png", result)
            st.download_button(
                label="💾 הורד תוצאה",
                data=buf.tobytes(),
                file_name=f"segmentation_{method.replace(' ', '_')}.png",
                mime="image/png"
            )

else:
    st.info("👈 אנא העלה תמונה מהסרגל הצידי כדי להתחיל")

# פוטר
st.markdown("---")
st.caption("that was fun/ i can wire here whatever i want")