# Create a Streamlit app to analyze housing data

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk 
# import model from pickle file
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pickle
model = pickle.load(open('xg_model.pkl', 'rb'))


# # Load the data
def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv")
    return data

data = load_data()

# Replace the colum long with lon 
data.rename(columns={'long': 'lon'}, inplace=True)


def cat_grade(grade):
    short = [1,2,3]
    average = [4,5,6,7,8,9,10]
 
    if grade in short:
        return 0
    elif grade in average:
        return 1
    else:
        return 2

def encode_zip(zipcode):
    zips = data['zipcode'].unique()
    for i in range(1, len(zips)+1):
        if zipcode == zips[i-1]:
            return i
        




# Sidebar
st.sidebar.header("Fill in the following information to get a price estimate:")

sqft_living_room = st.sidebar.text_input("Enter the square footage of the living room: ")
st.sidebar.caption("Square footage of the apartment's interior living space")

# Create Dropdown of Zipcode using the data
unique_zipcode = data['zipcode'].unique()

min_latitude, max_latitude =  data['lat'].min(), data['lat'].max()



min_longitude, max_longitude =  data['lon'].min(), data['lon'].max()

selected_zipcode = st.sidebar.selectbox('Zipcode', unique_zipcode)

# Create two columns for the sidebar - both of which will hold our min_bedrooms and min_bathrooms widgets
col1, col2 = st.sidebar.columns(2)

# Create a slider widget for the user to enter the minimum number of bedrooms
with col1:
    min_bedrooms = st.slider("Minimum number of bedrooms", min_value=1, max_value=10, value=1, step=1)

# Create a slider widget for the user to enter the minimum number of bathrooms
with col2:
    min_bathrooms = st.slider("Minimum number of bathrooms", min_value=1, max_value=10, value=1, step=1)



# Create two columns in the sidebar - both of which will hold our Latitude and Longitude widgets
col3, col4 = st.sidebar.columns(2)

# Create a slider widget for the user to enter the Latitude of the location
with col3:
    lat = st.slider("Latitude", min_value=47.0, max_value=48.0, value=47.0, step=0.01)

# Create a slider widget for the user to enter the Longitude of the location
with col4:
    lon = st.slider("Longitude", min_value=-122.0, max_value=-121.0, value=-122.0, step=0.01)


col5, col6 = st.sidebar.columns(2)

with col5:
    min_floors = st.slider("Minimum number of floors", min_value=1, max_value=5, value=1, step=1)




min_grade = st.sidebar.slider('Minimum number of grade', 1, 13, 1)

st.sidebar.caption(" 1-3 Good, 7 Average, 11-13 Excellent. In terms of building and construction design.")


sqft_lot = st.sidebar.text_input("Enter the square footage of the lot: ")
st.sidebar.caption("Square footage of the land space")

sqft_above = st.sidebar.text_input("Enter the square footage of interior above ground level: ")
st.sidebar.caption("Square footage of interior housing space that is above ground level")

min_condition = st.sidebar.slider('Condition of the Apartment', 1, 5, 1)

sqft_lving15_neighborhood = st.sidebar.text_input("Enter the square footage of interior housing living space for the nearest 15 neighbors: ")


st.sidebar.caption("Fill in all the fields to get a price estimate.")

price_button = st.sidebar.button("Predict Price")
 




# Main Panel

# Display the header and the dataset
st.header("Housing Price Prediction ")

if selected_zipcode:
    # Draw map of selected zipcode
    st.subheader("Map of selected zipcode: " + str(selected_zipcode))
    chart_data = data[data['zipcode'] == selected_zipcode][['lat', 'lon']]
    st.pydeck_chart(pdk.Deck(map_style=None,
                             initial_view_state=pdk.ViewState(latitude=chart_data['lat'].mean(), longitude=chart_data['lon'].mean(), zoom=11, pitch=50),
                             layers=[
                                pdk.Layer(
                                    'HexagonLayer',
                                    data=chart_data,
                                    get_position='[lon, lat]',
                                    radius=200,
                                    elevation_scale=4,
                                    elevation_range=[0, 1000],
                                    pickable=True,
                                    extruded=True,
                                
                                ),
                                 pdk.Layer(
                                     'ScatterplotLayer',
                                     data=chart_data,
                                     get_position='[lon, lat]',
                                     get_color='[200, 30, 0, 160]',
                                     get_radius=200,
                                 ),
                             ],
                             tooltip={"text": "Zipcode: " + str(selected_zipcode)}))

    # st.map(data[data['zipcode'] == selected_zipcode][['lon', 'lat']])


# Display Selected Values so Far if the user has selected values
if sqft_living_room and selected_zipcode and min_bedrooms and min_bathrooms and min_floors and min_grade and sqft_lot and sqft_above and min_condition and sqft_lving15_neighborhood:
    st.sidebar.subheader("Selected Values so Far")
    st.sidebar.write("Square footage of the living room: ", sqft_living_room)
    st.sidebar.write("Selected Zipcode: ", selected_zipcode)
    st.sidebar.write("Minimum number of bedrooms: ", min_bedrooms)
    st.sidebar.write("Minimum number of bathrooms: ", min_bathrooms)
    st.sidebar.write("Minimum number of floors: ", min_floors)
    st.sidebar.write("Minimum number of grade: ", min_grade)
    st.sidebar.write("Square footage of the lot: ", sqft_lot)
    st.sidebar.write("Square footage of the interior above ground level: ", sqft_above)
    st.sidebar.write("Condition of the Apartment: ", min_condition)
    st.sidebar.write("Square footage of interior housing living space for the nearest 15 neighbors: ", sqft_lving15_neighborhood)

def check_fields():
    if sqft_living_room and selected_zipcode and min_bedrooms and min_bathrooms and min_floors and min_grade and sqft_lot and sqft_above and min_condition and sqft_lving15_neighborhood:
        return True 
    else:
        return False 

# Display the price prediction if the user has clicked the button
if price_button and check_fields():
    #     # Create Data Frame for the selected values
    # # Display the selected data
    # st.subheader("Selected Data")
    # st.write(selected_data)
    features_to_scale= ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
           'grade', 'sqft_above', 'zip_enc', 'sqft_living15',
           'lat', 'long']

    # Create a Pandas Data Frame
    selected_data = pd.DataFrame({'sqft_living': [sqft_living_room],
                                    'bedrooms': [min_bedrooms], 
                                    'bathrooms': [min_bathrooms],
                                    'floors': [min_floors],
                                    'sqft_lot': [sqft_lot],
                                    'sqft_above': [sqft_above],
                                    'sqft_living15': [sqft_lving15_neighborhood],
                                    'lat': [lat],
                                    'long': [lon],
                                    'grade': [min_grade],
                                    'zip_enc': [selected_zipcode],

                                    }, columns= features_to_scale)

    enc_grade = cat_grade(min_grade)
    enc_zip = encode_zip(selected_zipcode)

    selected_data['grade'] = enc_grade
    selected_data['zip_enc'] = enc_zip

    selected_data['bedrooms'] = selected_data['bedrooms'].astype('float64')
    selected_data['bathrooms'] = selected_data['bathrooms'].astype('float64')
    selected_data['sqft_living'] = selected_data['sqft_living'].astype('float64')
    selected_data['sqft_lot'] = selected_data['sqft_lot'].astype('float64')
    selected_data['floors'] = selected_data['floors'].astype('float64')
    selected_data['grade'] = selected_data['grade'].astype('int64')
    selected_data['sqft_above'] = selected_data['sqft_above'].astype('float64')
    selected_data['zip_enc'] = selected_data['zip_enc'].astype('int8')
    selected_data['sqft_living15'] = selected_data['sqft_living15'].astype('float64')
    selected_data['lat'] = selected_data['lat'].astype('float64')
    selected_data['long'] = selected_data['long'].astype('float64')


    # Pass the DataFrame to the Model 
    prediction = model.predict(selected_data)
    if prediction[0]:
        st.balloons()

    # Display the price prediction
    st.subheader("Price Prediction")
    st.write("The predicted price of the house is: ")

    # Format the price to be in USD
    st.subheader("${:,.2f}".format(prediction[0]))


if price_button and not check_fields():
    st.subheader("Please select all the values to get the prediction")


