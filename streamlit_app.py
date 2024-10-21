import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.title('🤖AppliML🤖')

st.info('This is app predicts the species of penguins')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df

  st.write('**X**')
  X_raw = df.drop('species', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.species
  y_raw

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

with st.expander('Visualization: Species by Island and Gender'):
    st.write('**Bar Chart**')

    # Get unique islands and species
    unique_islands = df['island'].unique()
    unique_species = df['species'].unique()

    # Define the layout for the subplots (1 row, as many columns as there are islands)
    fig, axes = plt.subplots(1, len(unique_islands), figsize=(15, 6), sharey=True)

    # Loop through each island and create a bar plot
    for i, island in enumerate(unique_islands):
        # Filter the data for the current island
        island_data = df[df['island'] == island]
        
        # Group the data by species and sex, count occurrences, and reindex to include all species
        grouped_data = island_data.groupby(['species', 'sex']).size().unstack(fill_value=0)
        grouped_data = grouped_data.reindex(unique_species, fill_value=0)  # Reindex to ensure all species are shown

        # Plot bars for each sex category
        species = grouped_data.index
        sex_categories = grouped_data.columns
        bar_width = 0.35
        x = np.arange(len(species))

        ax = axes[i]  # Select the current subplot
        for j, sex in enumerate(sex_categories):
            ax.bar(x + j * bar_width, grouped_data[sex], width=bar_width, label=sex)
        
        # Set titles and labels
        ax.set_title(f'{island} Island')
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(species)
        ax.legend(title='Sex')
        if i == 0:
            ax.set_ylabel('Count')
        ax.set_xlabel('Species')

    # Adjust layout and display the plot
    plt.tight_layout()
    st.pyplot(fig)

# Input features
with st.sidebar:
  st.header('Input features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))
  
  # Create a DataFrame for the input features
  data = {'island': island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'sex': gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
  st.write('**Input penguin**')
  input_df
  st.write('**Combined penguins data**')
  input_penguins


# Data preparation
# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (input penguin)**')
  input_row
  st.write('**Encoded y**')
  y


# Model training and inference
## Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie',
                                 1: 'Chinstrap',
                                 2: 'Gentoo'})

# Display predicted species
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba,
             column_config={
               'Adelie': st.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Chinstrap': st.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Gentoo': st.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)


penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction][0]))
