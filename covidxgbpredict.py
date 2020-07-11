import joblib
from datetime import datetime


def xgbregressor(country, date):
    '''
    Takes in country and date, return the predicted number of confirmed covid cases.
    
    Args:
        country (string): Name of a country.
        date (string): The date to be predicted in the form "YYYY-MM-DD".
        
    Returns:
        int: The predicted number of confirmed cases. -1 if country does not exist.
    '''
    country = rename_country(country)
    date = datetime.strptime(date, '%Y-%m-%d')
    date = date.timetuple().tm_yday - 21
    
    xgb = joblib.load('covidxgbmodel.pkl')
    countrydf = joblib.load('covidcountrydf.pkl')
    
    popmean = countrydf.Population.mean()
    countrydf = countrydf[countrydf.Country == country]
    if countrydf.empty:
        return -1
    
    pop = countrydf.Population.median()
    countrydf.DayNum = date
    countrydf = countrydf.drop(columns=['Date','Country','Confirmed','ConfirmedNormalized','Population'])
    
    pred = xgb.predict(countrydf)[0]
    pred = int(pred / popmean * pop)
    return pred
    

    
def rename_country(c):
    if c in ["Côte d'Ivoire"]: c = "Cote d'Ivoire"
    elif c in ["Bolivia (Plurinational State of)"]: c = "Bolivia"
    elif c in ["Brunei Darussalam"]: c = "Brunei"
    elif c in ["Czech Republic", "Czech Republic (Czechia)"]: c = "Czechia"
    elif c in ["Iran (Islamic Republic of)"]: c = "Iran"
    elif c in ["Lao People's Democratic Republic"]: c = "Laos"
    elif c in ["Republic of Moldova"]: c = "Moldova"
    elif c in ["Republic of North Macedonia"]: c = "North Macedonia"
    elif c in ["Russian Federation"]: c = "Russia"
    elif c in ["Saint Kitts & Nevis"]: c = "Saint Kitts and Nevis"
    elif c in ["Sao Tome & Principe"]: c = "Sao Tome and Principe"
    elif c in ["South Korea", "Republic of Korea", "Korea South"]: c = "Korea, South"
    elif c in ["St. Vincent & Grenadines"]: c = "Saint Vincent and the Grenadines"
    elif c in ["Syrian Arab Republic"]: c = "Syria"
    elif c in ["Taiwan Province of China", "Taiwan*"]: c = "Taiwan"
    elif c in ["United Republic of Tanzania"]: c = "Tanzania"
    elif c in ["United States", "United States of America"]: c = "US"
    elif c in ["United Kingdom of Great Britain and Northern Ireland"]: c = "United Kingdom"
    elif c in ["Venezuela (Bolivarian Republic of)", "Venezuela"]: c = "Venezuela"
    elif c in ["Viet Nam"]: c = "Vietnam"
    return c