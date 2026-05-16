
##STEP ONE - IMPORT DATA##

import pandas as pd
import numpy as np
file_path = '/Users/ezrabayewitz/Downloads/portfolio/astronomy_data.csv'          
df = pd.read_csv(file_path)                                             #pandas reads over file


##STEP TWO - UNDERSTAND DATA##

print(df.head())                                                        #read over first few rows of dataframe

print(df.info())                                                        #get some important info on the data                                                                       #Non-Null Count tells us how many entries are left blank

print(df.describe())                                                    #get statistical data on columns

 
###STEP THREE - GET RID OF DUPLICATE ROWS###

df.drop_duplicates(inplace=True)                               
print(df.info())



###STEP FOUR - STANDARDIZE COLUMN NAMES###

df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]



###STEP FIVE - FIX DATA TYPES###

print(df.dtypes)

        #We see that the data type for 'distance_light_years' should be an integer, not an object

df['distance_light_years'] = pd.to_numeric(df['distance_light_years'], errors='coerce')         
print(df.dtypes)                                            



###STEP SIX - HANDLE OUTLIERS###


        #check how many rows have outliers in 'apparent_size_arcminutes'

outliers_count = df[df['apparent_size_arcminutes'] > 15].shape[0]              
print(f"Rows where apparent_size_arcminutes > 15: {outliers_count}")

        #get rid of outliers in 'apparent_size_arcminutes' column

df = df[df['apparent_size_arcminutes'] < 15]                    
print(df['apparent_size_arcminutes'].describe())

        #get rid of outlier in 'distance_light_years' column

df=df[df['distance_light_years'] < 20000]

###STEP SEVEN - CHECK & FILL MISSING VALUES###

print(df['magnitude'].isna().sum())                         #97 missijng values
print(df['distance_light_years'].isna().sum())              #47 missing values
print(df['spectral_type'].isna().sum())
print(df['discovery_year'].isna().sum())
print(df['observed_position_ra'].isna().sum())
print(df['observed_position_dec'].isna().sum())
print(df['apparent_size_arcminutes'].isna().sum())
print(df['planetary_system'].isna().sum())                  #307 missing values

        #fill magnitude with random value based on min and max magnitude of each spectral type 

spectral_type_ranges = df.groupby('spectral_type')['magnitude'].agg(['min', 'max'])

        #define function that fills missing value based on spectral type's magnitude range

def randomize_magnitude(row):
    if pd.isnull(row['magnitude']):
        #Get spectral type for each row
        spectral_type = row['spectral_type']

        #Find min and max magnitudes for spectral type
        min_mag = spectral_type_ranges.loc[spectral_type, 'min']
        max_mag = spectral_type_ranges.loc[spectral_type, 'max']

        #Randomly pick a value between the min and max for this spectral type
        return np.random.uniform(min_mag, max_mag)
    else:
        return row['magnitude']

        #apply function to fill missing magnitudes
    
df['magnitude'] = df.apply(randomize_magnitude, axis=1)
print(df['magnitude'].describe())
    
df.fillna({                                                 #missing values are filled with mean for light years                         
    'distance_light_years':df['distance_light_years'].mean(),
    }, inplace=True)

df['planetary_system'].fillna('unknown', inplace=True)      #blanks in object column are filled in with value

# Define the realistic range in arcseconds
min_realistic = 0.01
max_realistic = 1.0

# Get the minimum and maximum of the original 'apparent_size_arcminutes' column
min_original = df['apparent_size_arcminutes'].min()
max_original = df['apparent_size_arcminutes'].max()

# Scale the 'apparent_size_arcminutes' column to the realistic range (arcseconds)
df['scaled_apparent_size'] = ((df['apparent_size_arcminutes'] - min_original) /
                                     (max_original - min_original)) * (max_realistic - min_realistic) + min_realistic

###STEP EIGHT - REMOVE/CORRECT WEIRD CHARACTERS###    

        #check data, there are'nt any weird characters in this data so it's not necessary here
print(df[['apparent_size_arcminutes', 'scaled_apparent_size']])


###FINAL STEP - SAVE CLEANED DATA TO NEW CSV FILE###

df.to_csv('/Users/ezrabayewitz/Downloads/portfolio/cleaned_astronomy_data.csv', index=False)

print(df.loc[2, 'magnitude'])
