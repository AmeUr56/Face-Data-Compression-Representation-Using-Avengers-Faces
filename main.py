import streamlit as st
from PIL import Image
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
from io import BytesIO
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection

# Loading Estimators
estimators = {
    "PCA": [
        joblib.load("estimators/pca_2d.pkl"),
        joblib.load("estimators/pca_3d.pkl")
    ],
    "Kernel PCA": [
        joblib.load("estimators/kpca_2d.pkl"),
        joblib.load("estimators/kpca_3d.pkl")
    ],
    "Random Projection": [
        joblib.load("estimators/grp_2d.pkl"),
        joblib.load("estimators/grp_3d.pkl")
    ],
    "LLE": [
        joblib.load("estimators/lle_2d.pkl"),
        joblib.load("estimators/lle_3d.pkl")
    ],
    "MDS": [
        joblib.load("estimators/mds_2d.pkl"),
        joblib.load("estimators/mds_3d.pkl")
    ],
    "Isomap": [
        joblib.load("estimators/isomap_2d.pkl"),
        joblib.load("estimators/isomap_3d.pkl")
    ]
}
# Loading Images and their labels
images = np.load("artifacts/images.npy")
labels = np.load("artifacts/labels.npy")

# Visualization Methods
CLASSES = {0:"chris_evans",1:"chris_hemsworth",2:"mark_ruffalo",3:"robert_downey_jr",4:"scarlett_johansson"}

def visualization_2d(estimator_name, reduced_images, new_point):
    plt.figure(figsize=(8, 6))
    
    colors = sns.color_palette(n_colors=len(CLASSES))
    for i in np.unique(labels):
        indices = np.where(np.array(labels) == i)[0]
        sns.scatterplot(
            x=reduced_images[indices, 0],
            y=reduced_images[indices, 1],
            color=[colors[i]],
            label=CLASSES[i],
            s=30
        )

    sns.scatterplot(
        x=[new_point[0, 0]],
        y=[new_point[0, 1]],
        color='black',
        label='New Image',
        s=100,
        marker='X'
    )

    plt.title(f"2D Visualization with {estimator_name}")
    plt.legend(title="Class")

    plot = BytesIO()
    plt.savefig(plot, format="png", dpi=300, bbox_inches='tight')
    plt.close()

    return plot

def visualization_3d(estimator_name, reduced_images, new_point):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = sns.color_palette(n_colors=len(CLASSES))
    for i in np.unique(labels):
        indices = np.where(np.array(labels) == i)[0]
        ax.scatter(
            reduced_images[indices, 0],
            reduced_images[indices, 1],
            reduced_images[indices, 2],
            c=[colors[i]],
            label=CLASSES[i],
            s=30
        )


    ax.scatter(
        [new_point[0,0]], [new_point[0,1]], [new_point[0,2]],
        c='black',
        label='New Image',
        s=100,
        marker='X'
    )

    
    ax.set_title(f"3D Visualization with {estimator_name}")
    ax.legend(title="Class")
    plot = BytesIO()
    plt.savefig(plot, format="png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot 

linkedin_profile_badge = """
<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
<div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="ameur-b-25a155247" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://dz.linkedin.com/in/ameur-b-25a155247?trk=profile-badge">Ameur B.</a></div>
"""

# Page Configurations
st.set_page_config(
    page_title="Face Data Compression Representation Using Avengers Faces",
    page_icon="🦸‍♂️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS to hide hamburger menu and footer
hide_streamlit_style = """
    
"""

# Inject custom CSS
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🦸‍♂️ Face Data Compression Representation Using Avengers Faces")
st.sidebar.markdown("<a href='https://www.linkedin.com/in/ameur-b-25a155247/' target='_blank'><h1>Linkedin</h1></a>",unsafe_allow_html=True)
st.sidebar.markdown("<a href='https://x.com/Ame44i' target='_blank'><h1>X (Twitter)</h1></a>",unsafe_allow_html=True)

# Main
st.title("🦸‍♂️ Face Data Compression Representation Using Avengers Faces")
st.subheader("Face compression and reconstruction and also face 2d and 3d Visualization")

with st.form("form"):
    image = st.file_uploader("Face Image",type=['png','jpg','jpeg'])
    
    compression_method = st.selectbox("Compression Method(Machine Learning Algorithm)",options=['PCA','Kernel PCA','Random Projection','LLE','MDS','Isomap','TSNE'])
    compression_level = st.number_input("Compression Level(Desired Number of Dimention)",min_value=1,step=1)
     
    if st.form_submit_button("Submit"):
        image = Image.open(image)
        # Processing the image
        processed_image = image.convert("RGB") # Converting to RGB
        processed_image = processed_image.resize((90,90)) # Resizing
        processed_image = np.array(processed_image).reshape(1,-1) / 255.0 # Normalizing 

        if compression_level > 24300:
            st.error(f"Compression Level must be lower: '{24300}'")
        else:
            st.subheader("Face Compression & Reconstruction")
            if compression_method not in ["PCA","Kernel PCA","Random Projection"]:
                st.error(f"The compression method '{compression_method}' is not valid for reconstruction, try PCA, Kernel PCA, Random Projection.")
            else:
                # Reconstruction and Presentation of the Original and Reconstructed Image                
                match compression_method:
                    case "PCA":
                        estimator = PCA(n_components=compression_level,svd_solver='auto')
                    case "Kernel PCA":
                        estimator = KernelPCA(n_components=compression_level, kernel='rbf', gamma=0.1,fit_inverse_transform=True)
                    case "Random Projection":
                        estimator = GaussianRandomProjection(n_components=compression_level)
                try:
                    estimator.fit(images)
                except:
                    estimator = TruncatedSVD(n_components=compression_level)
                    estimator.fit(images)
                    
                reduced_image = estimator.transform(processed_image)
                recovered_image = estimator.inverse_transform(reduced_image).reshape(90,90,3)
                
                if recovered_image.max() <= 1.0:
                    recovered_image = (recovered_image * 255).astype(np.uint8)
                else:
                    recovered_image = recovered_image.astype(np.uint8)
                
                col_1,col_2 = st.columns([15]*2)
                with col_1:
                    st.image(image, caption="Original Image", use_container_width =True)
                with col_2:
                    st.image(Image.fromarray(recovered_image), caption="Reconstructed Image", use_container_width =True)

        # Face 2D and 3D Visualization
        st.subheader("Face 2D and 3D Visualization")
        if compression_method != "TSNE":
            estimator_2d,estimator_3d = estimators[compression_method]
            
            # 2D Process
            images_2d = estimator_2d.transform(images)
            image_2d = estimator_2d.transform(processed_image)
            plot_2d = visualization_2d(compression_method,images_2d,image_2d)
                        
            # 3D Process
            images_3d = estimator_3d.transform(images)
            image_3d = estimator_3d.transform(processed_image)
            plot_3d = visualization_3d(compression_method,images_3d,image_3d)            
        else:
            combined_images = np.concatenate([images, processed_image], axis=0)
            
            # 2D Process
            combined_2d_images = TSNE(n_components=2, method='exact', random_state=42).fit_transform(combined_images)
            images_2d,image_2d = combined_2d_images[:-1], combined_2d_images[-1].reshape(1,-1)
            plot_2d = visualization_2d(compression_method,images_2d,image_2d)
            
            # 3D Process
            combined_3d_images = TSNE(n_components=3, method='exact', random_state=42).fit_transform(combined_images)
            images_3d,image_3d = combined_3d_images[:-1], combined_3d_images[-1].reshape(1,-1)
            plot_3d = visualization_3d(compression_method,images_3d,image_3d)    
    
        col_1,col_2 = st.columns([15]*2)
        with col_1:
            st.image(plot_2d, caption="2D Plot", use_container_width =True)
        with col_2:
            st.image(plot_3d, caption="3D Plot", use_container_width =True)
        
        