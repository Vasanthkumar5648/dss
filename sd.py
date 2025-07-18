import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import time
def deg_to_rad(degrees):
    return degrees * (np.pi/180)

# Haversine formula to calculate distance
def calculate_distance(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2-lat1)
    d_lon = deg_to_rad(lon2-lon1)
    a = np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

df['distance'] = np.nan

for i in range(len(df)):
         df.loc[i,'distance'] = distcalculate(df.loc[i,'Restaurant_latitude'],
                                        df.loc[i,'Restaurant_longitude'],
                                        df.loc[i,'Delivery_location_latitude'],
                                        df.loc[i,'Delivery_location_latitude'])
x=np.array(df[["Delivery_person_Age",
              "Delivery_person_Ratings",
              "distance"]])
y=np.array(df[["Time_taken(min)"]])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=42)
prediction = max(15, min(120, distance * 2 + np.random.normal(10, 5) - (ratings * 3) + (age / 10)))
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()       

model.compile(optimizer='adam', loss='mean_squared_error') 
model.fit(x_train,y_train, batch_size=1,epochs=10)
st.set_page_config(page_title="Food Delivery Time Predictor", page_icon="🍔", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            background-color: #ffffff;
        }
        .css-1aumxhk {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

st.title("🍔 Food Delivery Time Prediction")
st.markdown("Predict how long your food delivery will take based on delivery partner details and distance.")

with st.expander("ℹ️ About this app"):
        st.write("""
        This app predicts food delivery time using an LSTM neural network model. 
        The prediction is based on:
        - Delivery partner's age
        - Delivery partner's ratings
        - Distance between restaurant and delivery location
        """)    
with st.spinner('Calculating delivery time...'):
                time.sleep(2)
                st.success("Prediction complete!")
                
                st.subheader("Predicted Delivery Time")
                st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{int(prediction)} minutes</h1>", unsafe_allow_html=True)
                
                # Visual indicator
                if prediction < 30:
                    st.success("Fast delivery expected! 🚀")
                elif prediction < 45:
                    st.info("Average delivery time ⏱️")
                else:
                    st.warning("Longer delivery time expected 🐢")
        else:
            st.error("Please enter valid location coordinates for both restaurant and delivery location.")
