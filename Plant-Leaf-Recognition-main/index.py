import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib

# Set matplotlib backend for compatibility
matplotlib.use('Agg')

# Load dataset
data = "Plant-Leaf-Recognition-main/static/Copy of dataset testing.xlsx"
df = pd.read_excel(data)

# Rename columns for clarity
df.columns = [
    "Name", "Scientific Name", "Type", "Height", "Lifespan",
    "Oxygen Level", "Description", "Google Link", "YouTube Link", "Location"
]

# Combine relevant text columns for vectorization
df['combined_features'] = (
    df['Type'].astype(str) + ' ' + df['Height'].astype(str) + ' ' +
    df['Lifespan'].astype(str) + ' ' + df['Oxygen Level'].astype(str) +
    ' ' + df['Description'].astype(str) + ' ' + df['Location'].astype(str)
)

# Helper function to sort ranges
def sort_ranges(values):
    sorted_values = sorted(values, key=lambda x: tuple(map(float, x.split("-"))))
    return ["All"] + sorted_values

# Preprocess and sort unique values
df["Height"] = df["Height"].astype(str) + " m"  
height_sorted = sort_ranges(df["Height"].str.replace(" m", "").unique().tolist())
height_sorted = [f"{val} m" if val != "All" else val for val in height_sorted]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(df['combined_features'])

# Cosine similarity
cosine_sim = cosine_similarity(features, features)

# Helper function: Get similar plants by name
def get_similar_plants(plant_name):
    if plant_name not in df['Name'].values:
        return pd.DataFrame()
    try:
        plant_idx = df[df['Name'] == plant_name].index[0]
        similar_indices = cosine_sim[plant_idx].argsort()[:-6:-1]
        similar_plants = df.iloc[similar_indices]
        return similar_plants
    except Exception as e:
        print(f"Error finding similar plants: {e}")
        return pd.DataFrame()

# Helper function: Filter plants by dropdowns
def filter_plants(plant_type, height, lifespan, oxygen_level):
    filtered_df = df.copy()
    if plant_type != "All":
        filtered_df = filtered_df[filtered_df["Type"] == plant_type]
    if height != "All":
        filtered_df = filtered_df[filtered_df["Height"] == height]
    if lifespan != "All":
        filtered_df = filtered_df[filtered_df["Lifespan"] == lifespan]
    if oxygen_level != "All":
        filtered_df = filtered_df[filtered_df["Oxygen Level"] == oxygen_level]
    return filtered_df

# Generate visualizations
def generate_visualizations(filtered_df):
    if filtered_df.empty:
        return None
    
    filtered_df["Height"] = pd.to_numeric(filtered_df["Height"].str.replace(" m", ""), errors="coerce")
    filtered_df.dropna(subset=["Height"], inplace=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 9))

    type_counts = filtered_df["Type"].value_counts()
    if not type_counts.empty:
        axs[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
        axs[0, 0].set_title("Type Distribution")
    else:
        axs[0, 0].axis('off')

    if not filtered_df.empty:
        filtered_df.plot.bar(x="Name", y="Height", ax=axs[0, 1], color='red', legend=False)
        axs[0, 1].set_title("Height Distribution")
        axs[0, 1].set_ylabel("Height (m)")
    else:
        axs[0, 1].axis('off')

    location_counts = filtered_df["Location"].value_counts()
    if not location_counts.empty:
        axs[1, 0].pie(location_counts.values, labels=location_counts.index, autopct='%1.1f%%', startangle=90)
        axs[1, 0].set_title("Location Distribution")
    else:
        axs[1, 0].axis('off')

    oxygen_counts = filtered_df["Oxygen Level"].value_counts()
    if not oxygen_counts.empty:
        axs[1, 1].bar(oxygen_counts.index, oxygen_counts.values, color='green')
        axs[1, 1].set_title("Oxygen Level Distribution")
        axs[1, 1].set_ylabel("Count")
    else:
        axs[1, 1].axis('off')

    img = BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

# Streamlit UI
st.set_page_config(page_title="Plant Recommendation", layout="wide")

st.markdown("<h1 style='text-align: center;'>🌱 Plant Recommendation & Filtering System</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image("Plant-Leaf-Recognition-main/logoheade.png", use_container_width=True)

st.subheader("🔍 Search for Similar Plants")
plant_name_options = ["Select a plant"] + sorted(df["Name"].dropna().unique().tolist())
plant_name = st.selectbox("Search for a plant by name:", plant_name_options)

if plant_name != "Select a plant":
    similar_plants = get_similar_plants(plant_name)

    if not similar_plants.empty:
        st.write("### 🌿 Similar Plants")

        col1, col2, col3 = st.columns(3)  # Three-column layout for horizontal display

        displayed_names = set()
        for index, row in similar_plants.iterrows():
            if row["Name"] not in displayed_names:
                displayed_names.add(row["Name"])

                with [col1, col2, col3][index % 3]:  # Distribute cards across columns
                    st.markdown(
                        f"""
                        <div style="border: 2px solid #ddd; border-radius: 10px; padding: 15px; 
                                    margin-bottom: 0px; background-color:hsl(0, 10.00%, 96.10%); width: 100%;">
                            <h4 style="text-align: center; color: darkgreen; margin-bottom: 0px;">
                                {row['Name']} ({row['Scientific Name']})
                            </h4>
                            <p><strong>Type:</strong> {row['Type']}</p>
                            <p><strong>Height:</strong> {row['Height']} meters</p>
                            <p><strong>Lifespan:</strong> {row['Lifespan']} years</p>
                            <p><strong>Oxygen Level:</strong> {row['Oxygen Level']}</p>
                            <p style="font-size: 14px;">{row['Description']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Buttons with smaller size
                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        st.markdown(
                            f'<a href="{row["Google Link"]}" target="_blank">'
                            '<button style="width: 100%; font-size: 12px; padding: 5px;">🌿 More Info</button>'
                            '</a>', unsafe_allow_html=True
                        )

                    with col_b:
                        st.markdown(
                            f'<a href="{row["YouTube Link"]}" target="_blank">'
                            '<button style="width: 100%; font-size: 12px; padding: 5px;">🎥 Watch Video</button>'
                            '</a>', unsafe_allow_html=True
                        )

                    with col_c:
                        st.markdown(
                            f'<a href="https://www.google.com/maps/search/{row["Location"]}" target="_blank">'
                            f'''
<div style="margin-bottom: 50px;">
    <button style="width: 100%; font-size: 12px; padding: 5px;">
        📍 Found in {row["Location"]}
    </button>
</div>
'''

                            '</a>', unsafe_allow_html=True
                        )


st.subheader("🎯 Filter Plants by Category")
plant_type = st.selectbox("Select Type", ["All"] + df["Type"].unique().tolist())
height = st.selectbox("Select Height", height_sorted)
lifespan = st.selectbox("Select Lifespan", ["All"] + df["Lifespan"].unique().tolist())
oxygen_level = st.selectbox("Select Oxygen Level", ["All"] + df["Oxygen Level"].unique().tolist())

# Custom Green Button using HTML
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        padding: 8px 20px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Filter Button
if st.button("Filter"):
    filtered_df = filter_plants(plant_type, height, lifespan, oxygen_level)
    if not filtered_df.empty:
        st.write("### 🌱 Filtered Plants")
        st.dataframe(filtered_df)
        vis_img = generate_visualizations(filtered_df)
        if vis_img:
            st.image(vis_img, caption="Plant Data Visualizations", use_container_width=True)
    else:
        st.write("🚫 No plants match the selected filters.")

