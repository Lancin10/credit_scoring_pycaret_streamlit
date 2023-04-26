# -*- coding: utf-8 -*-
"""
Created on 15 - 07 - 2021

@author: Lancine
"""

from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


def predict_client(model, df):
    
    predictions_data = predict_model(model, df)
    
    return predictions_data['Label'][0]
    
model = load_model('Final_Model_xgboost')


st.title(' App Web de crédit scoring')
st.write("Il s'agit d'une application Web pour classer les clients d’une banque en bon ou mauvais prêteur.")


Maturite_mois = st.sidebar.slider(label = 'Maturité mois', min_value = 16,
                          max_value = 24 ,
                          value = 23,
                          step = 1)

Montant_Pret_BAM = st.sidebar.slider(label = 'Montant Prêt BAM', min_value = 5000,
                          max_value = 15000 ,
                          value = 8200,
                          step = 25)

Sexe = st.sidebar.selectbox('Sexe',['Male','Female'])

Etat_Civil = st.sidebar.selectbox('Etat Civil',['Divorced','Married','Single','Widowed'])

Niveau_Formation = st.sidebar.selectbox(
    'Niveau Formation',
    ['Completed University','Some College Courses','Completed Vocational Training','High School Diploma','Secondary School to Grade 10'])

Age_ans = st.sidebar.slider(label = 'Age ans', min_value = 18,
                          max_value = 65 ,
                          value = 32,
                          step = 1)                          

Ans_a_ladresse = st.sidebar.slider(label = "Ans à l'adresse", min_value = 0.5,
                          max_value = 30.0 ,
                          value = 1.0,
                          step = 0.5)

Locataire_Proprietaire = st.sidebar.selectbox("Locataire / Propriétaire",['RENT','Own'])

Nbre_de_Dependants = st.sidebar.slider(label = 'Nbre de Dépendants', min_value = 0,
                          max_value = 5,
                          value = 2,
                          step = 1)

Ans_en_Activite = st.sidebar.slider(label = 'Ans en Activité ', min_value = 0.5,
                          max_value = 15.0 ,
                          value = 1.0,
                          step = 0.5)

Emplacement_du_business = st.sidebar.selectbox("Emplacement du business",['Region1','Region2','Region3','Region4','Region5'])

Credit_Bureau_negative = st.sidebar.slider(label = 'Crédit Bureau négatif=1', min_value = 0,
                          max_value = 1 ,
                          value = 0,
                          step = 1)

NbreEmployesFamille = st.sidebar.slider(label = 'NbreEmployésFamille', min_value = 0,
                          max_value = 7 ,
                          value = 1,
                          step = 1)

Type_dActivite = st.sidebar.selectbox(
    "Type d'Activité",
    ['Craftsperson','Personal Services','Car Repair','Child Care','Convenience Store','Small Grocers','General Contractor'])
                         
Ventes_Mensuelles_BAM = st.sidebar.slider(label = 'Ventes Mensuelles BAM', min_value = 400,
                          max_value = 20000,
                          value = 2550,
                          step = 25)

features = {'Maturité mois': Maturite_mois, 'Montant Prêt BAM': Montant_Pret_BAM,
            'Sexe': Sexe, 'Etat Civil': Etat_Civil,'Niveau Formation':Niveau_Formation,
            'Age ans':Age_ans, "Ans à l'adresse":Ans_a_ladresse, "Locataire / Propriétaire": Locataire_Proprietaire,
            'Nbre de Dépendants': Nbre_de_Dependants, 'Ans en Activité ': Ans_en_Activite,
            "Emplacement du business": Emplacement_du_business, 'Crédit Bureau négatif=1': Credit_Bureau_negative, 
            'NbreEmployésFamille': NbreEmployesFamille, "Type d'Activité":Type_dActivite,
            'Ventes Mensuelles BAM':Ventes_Mensuelles_BAM
            }
 

features_df  = pd.DataFrame([features])

map_dict = {0:'Bon client',
            1:'Mauvais client'}
st.table(features_df)  

if st.button('Predict'):
    
    prediction = predict_client(model, features_df)
    
    st.write("La prediction est : {}".format(map_dict [prediction]))
    
 
    
