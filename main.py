import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow import keras
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

@st.cache_data
def read_data():
    all_vecs = np.load("all_vecs.npy")
    all_names = np.load("all_names.npy")
    return all_vecs , all_names

vecs , names = read_data()


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.markdown('<h1 style="color: blue;">VisioVerse Artisan Hub: Bridging gap Between Creators and Customers</h1>', unsafe_allow_html=True)
st.markdown('<style>body { background-color: red; }</style>', unsafe_allow_html=True)
st.image(["All_pics/Terracotta.jpg", "All_pics/Handicrafts.jpg"], caption=['Handicraft Image 1', 'Handicraft Image 2'], width=300)

def main():
    # Navigation
    page = st.sidebar.radio("Navigation", ["Home","About","Register/Login","Products", "Pay"])

    if page == "Home":
        show_homepage()
    elif page == "Products":
        show_products()
    elif page == "About":
        show_about()
    elif page =="Register/Login":
        show_login()
    elif page =='Pay':
        show_payment()

def show_homepage():
    st.header("Welcome to our Handicraft Website!")
    st.write("Discover unique handmade items!")

def show_products():
    st.header("Products")

    products = [
        {
            "name": "Handmade Jewelry",
            "description": "Beautiful handmade necklaces, bracelets, and earrings.",
            "image": "All_pics\earrings (2).jpg"  # Replace with actual image URL
        },
        {
            "name": "Handwoven Baskets",
            "description": "Sturdy and stylish baskets made from natural materials.",
            "image": "All_pics\Bamboo_Basket.jpg"  # Replace with actual image URL
        },
        {
            "name": "Hand-painted Pottery",
            "description": "Decorative pottery items painted by skilled artisans.",
            "image": "All_pics\pot.jpg"  # Replace with actual image URL
        },
        # Add more products as needed
    ]

    for product in products:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(product["image"], use_column_width=True)
        with col2:
            st.subheader(product["name"])
            st.write(product["description"])
        st.write("---")

def show_about():
    st.header("About Us")

    st.write("Handicraft Website is dedicated to showcasing the finest handmade products from artisans around the world.")
    st.write("Our mission is to promote traditional craftsmanship and support local artisans.")
    st.write("Thank you for visiting!")

def show_login():


    # Sample user database (in-memory)
    user_database = {
        'Pragati': '111',
        'Navya': '222',
        'Mehul': '333',
        'Nimisha':'444'
    }

    def register(username, password):
        if username in user_database:
            return False  # Username already exists
        else:
            user_database[username] = password
            return True

    def login(username, password):
        if username in user_database and user_database[username] == password:
            return True
        else:
            return False

    # Streamlit layout
    st.title("Login and Registration Page")

    # Create a sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Login", "Register"])

    if page == "Login":
        st.header("Login")
        
        # Create the login form
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        # Login button
        if st.button("Login"):
            if login(username, password):
                st.success("Login Successful")
            else:
                st.error("Invalid Username or Password")

    elif page == "Register":
        st.header("Register")
        
        # Create the registration form
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")

        # Register button
        if st.button("Register"):
            if register(new_username, new_password):
                st.success("Registration Successful")
            else:
                st.error("Username already exists. Please choose a different username.")

def show_payment():

    def process_payment(card_number, expiry_date, cvv, amount):
        # Simulate payment processing
        if len(card_number) == 16 and len(expiry_date) == 5 and len(cvv) == 3 and amount > 0:
            return True
        else:
            return False

    # Streamlit layout
    st.title("Payment Page")

    # Payment form
    st.header("Enter Payment Details")

    card_number = st.text_input("Card Number", max_chars=16)
    expiry_date = st.text_input("Expiry Date (MM/YY)", max_chars=5)
    cvv = st.text_input("CVV", max_chars=3, type="password")
    amount = st.number_input("Amount", min_value=0.01, format="%.2f")

    # Process payment button
    if st.button("Pay Now"):
        if process_payment(card_number, expiry_date, cvv, amount):
            st.success("Payment Successful")
        else:
            st.error("Payment Failed. Please check your details and try again.")


if __name__ == "__main__":
    main()


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# steps
# file upload -> save
uploaded_file = st.file_uploader("Upload a file")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        st.info(uploaded_file.name)
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices = recommend(features,feature_list)
        # show
        c1 , c2 , c3 , c4 , c5, c6 = st.columns(6)
        c1.image(Image.open("./All_pics/" + names[indices[0]][0]))
        c1.info(names[indices[0][0]])
        c2.image(Image.open("./All_pics/" + names[indices[0][1]]))
        c2.info(names[indices[0][1]])
        c3.image(Image.open("./All_pics/" + names[indices[0][2]]))
        c3.info(names[indices[0][2]])
        c4.image(Image.open("./All_pics/" + names[indices[0][3]]))
        c4.info(names[indices[0][3]])
        c5.image(Image.open("./All_pics/" + names[indices[0][4]]))
        c5.info(names[indices[0][4]])
        c6.image(Image.open("./All_pics/" + names[indices[0][5]]))
        c6.info(names[indices[0][5]])



        