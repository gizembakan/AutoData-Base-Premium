import streamlit as st
import pandas as pd
import altair as alt
import pickle
import numpy as np


st.set_page_config(page_title="Araba Sigorta Poliçesi", page_icon="🚗")

st.markdown("# 🚗 Araba Sigorta Poliçesi" )
st.sidebar.header("🚗 Araba Sigorta Poliçesi")

col1, col2= st.columns([7, 3] ,  gap = 'small')

with col2:
    st.write('\n')
    col2.image('https://www.linkpicture.com/q/TRAFİK-v2-1.jpg')

with col1:
    st.write('\n')
    st.write('\n')

    st.write(
    """
       Arabalarına sigorta poliçesi yaptıran müşterilerin kaza yapıp yapmama ihtimalleri üzerinden,
       sigorta şirketinin ödeyebileceği muhtemel baz primlerin tahminlenmesi amaçlanmıştır.
       \nFrekans-Severity methodu ile modelleme kurulmuştur.  
       """
    )
    
#Başlık    
subheader = '<p font-size: 24px;"><br/><b>👇Aşağıdaki değerleri değiştirerek muhtemel baz prim hesabı yaptırabilirsiniz.</b><br/></p>'
st.markdown(subheader, unsafe_allow_html=True)



df_dict = {}

#Kullanıcıdan Girdilerin Alınması
features = {
    'Select number of kids you have:': ('KIDSDRIV', int, (0, 10)),
    'Select your age:': ('AGE', int, (18, 100)),
    'Select number of kids you have in home:': ('HOMEKIDS', int, (0, 10)),
    'Select year of joining to company as customer:': ('YOJ', float, (0, 23)),
    'Select annual income you have:': ('INCOME', float, (0, 500000)),
    'Select if your parent is alive or not:': ('PARENT1', str, ['alive', 'deceased']),
    'Select value of your home: ': ('HOME_VAL', float, (0, 1000000)),
    'Select your marital status:': ('MSTATUS', str, ['married', 'single']),
    'Select your gender:': ('GENDER', str, ['male', 'female']),
    'Select your most recent education level:': ('EDUCATION', str, ['PhD', 'High School', 'Bachelors', 'Masters']),
    'Select your occupation:': ('OCCUPATION', str, ['Professional', 'Blue Collar', 'Manager', 'Clerical', 'Doctor', 'Lawyer', 'Home Maker', 'Student', 'nan']),
    'Select your traveling time (in hours/weeks):': ('TRAVTIME', int, (0, 150)),
    'Select your main purpose of using your car': ('CAR_USE', str, ['Private', 'Commercial']),
    'Select legal citation system:': ('BLUEBOOK', float, (0, 100000)),
    'Select TIF: ':('TIF', int, (0, 25)),
    'Select type of your car:': ('CAR_TYPE', str, ['Minivan', 'Van', 'z_SUV', 'Sports Car', 'Panel Truck', 'Pickup']),
    'Select colour of your car (if it is red)': ('RED_CAR', str, ['yes', 'no']),
    'Select old claim of your car:': ('OLDCLAIM', float, (0, 100000)),
    'Select number of times you had claims:': ('CLM_FREQ', int, (0, 10)),
    'Select if claim has revoked:': ('REVOKED', str, ['Yes', 'No']),
    'Select claim points:': ('MVR_PTS', int, (0, 15)), 
    'Select age of your car:': ('CAR_AGE', int, (0, 50))       
}


#specify_features = st.selectbox('Please enter the information required to calculate your base premium.', list(features.keys()))


for feature, (col_name, data_type, options) in features.items():
    if data_type == int or data_type == float:
        base, top = options
        value = st.slider(feature, base, top)
    else:
        value = st.radio(feature, options)
        
#Feature Engineering
#PARENT1 No -> deceased, Yes-> alive
    if value == 'alive':
        value = 'Yes'
    if value == 'deceased':
        value = 'No'
#MSTATUS z_No -> single, Yes -> married
    if value == 'single':
        value = 'z_No'
    if value == 'married':
        value = 'Yes'
#GENDER M -> male, z_F -> female
    if value == 'male':
        value = 'M'
    if value == 'female':
        value = 'z_F'
#EDUCATION 'PhD', 'z_High School'-> High School, 'Bachelors', '<High School', 'Masters'
    if value == 'High School':
        value = 'z_High School'
#OCCUPATION 'Professional', 'z_Blue Collar'-> 'Blue Collar', 'Manager', 'Clerical', 'Doctor',
#       'Lawyer', nan, 'Home Maker', 'Student'
    if value == 'Blue Collar':    
        value = 'z_Blue Collar'


    df_dict[col_name] = [value]
    df = pd.DataFrame(df_dict)
    
if st.button('Get my data'):
    st.write(df)



#Kullanıcı Frekans Tahmini
target_encoder = pickle.load(open('All_homeworks/models/Target_Encoder_Autodata.sav', 'rb'))
df = target_encoder.transform(df)

FRQ_Model = pickle.load(open('All_homeworks/models/FRQ_Model.sav', 'rb'))
pred_FRQ = FRQ_Model.predict(df)

#Kullanıcı Severity Tahmini
SEV_Model = pickle.load(open('All_homeworks/models/SEV_Model.sav', 'rb'))
pred_SEV = SEV_Model.predict(df)

#Kullanıcı Baz Prim Tahmini
base_premium = pred_FRQ[0]  * pred_SEV[0]

    
if st.button('Calculate my base premium'):
    st.write('Calculation results for your data: ', base_premium)

    with st.expander('See more: '):
        st.write('Frequency prediction result: ', pred_FRQ[0])
        st.write('Severity prediction result: ', pred_SEV[0])



st.sidebar.success(f"###  👉 Tahmini Baz Prim: ${base_premium}")  

