import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import warnings
from dotenv import load_dotenv
import requests
import os
warnings.filterwarnings('ignore')
load_dotenv()
os.environ["API_KEY"] = "eca001c11a5444b781600138260401"


class CrowdPredictionInputProcessor:
    def __init__(self):
        """Initialize all the mappings and factors from your data generation script"""
        
        self.category_base = {
            "temple": 0.80,
            "memorial_park": 0.70,
            "lake_dam": 0.65,
            "waterfall": 0.60,
            "natural_scenic": 0.62,
        }
        
        self.places_data = {
            "Hanuman Vatika": {"category": "temple", "place_factor": 1.20},
            "Vedvyas Temple": {"category": "temple", "place_factor": 1.15},
            "Indira Gandhi Park": {"category": "memorial_park", "place_factor": 1.25},
            "Vaishno Devi Temple, Rourkela": {"category": "temple", "place_factor": 1.05},
            "Jagannath Temple, Sector 3": {"category": "temple", "place_factor": 0.98},
            "Mandira Dam": {"category": "lake_dam", "place_factor": 1.20},
            "Pitamahal Dam": {"category": "lake_dam", "place_factor": 1.05},
            "Koel Riverbank": {"category": "lake_dam", "place_factor": 0.95},
            "Rani Sati Mandir": {"category": "temple", "place_factor": 0.85},
            "Ghoghar Temple": {"category": "temple", "place_factor": 0.82},
            "Deodhar George": {"category": "natural_scenic", "place_factor": 1.00},
            "Darjeeng Picnic Spot": {"category": "lake_dam", "place_factor": 0.95},
            "Ranighatra Picnic Spot": {"category": "lake_dam", "place_factor": 0.90},
            "Sitakund Waterfall": {"category": "waterfall", "place_factor": 0.95},
            "Mirigikhoj Waterfall": {"category": "waterfall", "place_factor": 0.85},
        }
        
        self.weekly_factor = {0: 0.82, 1: 0.86, 2: 0.90, 3: 0.95, 4: 1.05, 5: 1.28, 6: 1.32}
        
        self.month_factor = {1:1.18,2:1.15,3:1.05,4:0.90,5:0.80,6:0.85,7:0.92,8:0.95,9:0.96,10:1.05,11:1.18,12:1.22}
        
        self.hour_curve = {
            "temple":        {7:0.80,8:0.95,9:0.75,10:0.60,11:0.55,12:0.60,13:0.65,14:0.70,15:0.75,16:0.85,17:0.95,18:1.00,19:0.90,20:0.70},
            "memorial_park": {7:0.30,8:0.55,9:0.75,10:0.85,11:0.90,12:0.80,13:0.70,14:0.70,15:0.75,16:0.90,17:1.00,18:0.90,19:0.65,20:0.45},
            "lake_dam":      {7:0.20,8:0.40,9:0.65,10:0.80,11:0.90,12:0.85,13:0.75,14:0.70,15:0.75,16:0.90,17:1.00,18:0.95,19:0.70,20:0.50},
            "waterfall":     {7:0.10,8:0.20,9:0.40,10:0.60,11:0.80,12:0.90,13:0.95,14:1.00,15:0.90,16:0.75,17:0.50,18:0.30,19:0.15,20:0.05},
            "natural_scenic":{7:0.35,8:0.60,9:0.80,10:0.90,11:0.95,12:0.85,13:0.75,14:0.75,15:0.85,16:1.00,17:0.95,18:0.85,19:0.60,20:0.40},
        }
        
        self.holiday_rules = {
            "2024-08-15": ("Independence Day", 1.15, 1.15),
            "2024-08-19": ("Raksha Bandhan", 1.05, 1.20),
            "2024-08-26": ("Janmashtami", 1.05, 1.25),
            "2024-09-14": ("Nuakhai", 1.20, 1.35),
            "2024-10-02": ("Gandhi Jayanti", 1.10, 1.10),
            "2024-10-31": ("Diwali", 1.08, 1.20),
            "2024-11-01": ("Diwali", 1.10, 1.22),
            "2024-11-02": ("Diwali", 1.08, 1.20),
            "2024-12-25": ("Christmas", 1.12, 1.10),
            "2025-01-26": ("Republic Day", 1.12, 1.12),
            "2025-07-07": ("Rath Yatra", 1.20, 1.40),
            "2025-08-09": ("Raksha Bandhan", 1.05, 1.20),
            "2025-08-15": ("Independence Day", 1.15, 1.15),
        }
        
        self.season_map = {
            1:  "winter2",       # January
            2:  "winter2",       # February
            3:  "summer0",       # March
            4:  "summer1",       # April
            5:  "summer1",       # May
            6:  "monsoon0",      # June
            7:  "monsoon1",      # July
            8:  "monsoon2",      # August
            9:  "post-monsoon", # September
            10: "post-monsoon", # October
            11: "winter0",       # November
            12: "winter1"        # December
        }
        
        self.le_place = LabelEncoder()
        self.le_category = LabelEncoder()
        self.le_weekday = LabelEncoder()
        self.le_holiday = LabelEncoder()
        self.le_season = LabelEncoder()
        
        self.outdoor_categories = {"lake_dam", "memorial_park", "waterfall", "natural_scenic"}
        
        self._fit_encoders()
    
    def _fit_encoders(self):
        place_names = list(self.places_data.keys())
        self.le_place.fit(place_names)
        
        categories = list(self.category_base.keys())
        self.le_category.fit(categories)
        
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.le_weekday.fit(weekdays)
        
        holiday_labels = ['Not a holiday', 'Weekend'] + [rule[0] for rule in self.holiday_rules.values()]
        self.le_holiday.fit(list(set(holiday_labels)))
        
        seasons = list(set(self.season_map.values()))
        self.le_season.fit(seasons)
    
    def get_current_weather(self, api_key=None):
        if api_key:
            try:
                url = "http://api.weatherapi.com/v1/current.json"
                params = {
                    'key': api_key,
                    'q': 'Rourkela,Odisha,India', 
                    'aqi': 'no'  
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                temperature = data['current']['temp_c']
                
                rain_flag = 0
                current_condition = data['current']['condition']['text'].lower()
                precipitation_mm = data['current']['precip_mm']
                
                rain_conditions = [
                    'rain', 'drizzle', 'shower', 'thunderstorm', 'storm',
                    'precipitation', 'wet', 'pour'
                ]
                
                if precipitation_mm > 0:
                    rain_flag = 1
                elif any(condition in current_condition for condition in rain_conditions):
                    rain_flag = 1
                
                print(f"Weather API: {temperature}°C, Condition: {data['current']['condition']['text']}, Rain: {'Yes' if rain_flag else 'No'}")
                return temperature, rain_flag
                
            except Exception as e:
                print(f"Weather API error: {e}")
                print("Falling back to estimated weather...")
                return self._get_fallback_weather()
        else:
            print("⚠️ No API key provided. Using fallback weather estimation...")
            return self._get_fallback_weather()
    
    def _get_fallback_weather(self):
        """
        Fallback weather estimation when API is unavailable
        Uses seasonal patterns for Rourkela
        """
        now = datetime.now()
        month = now.month
        hour = now.hour
        
        base_temp = {1:16,2:19,3:26,4:32,5:36,6:32,7:30,8:29,9:29,10:27,11:21,12:17}[month]
        
        variation = -5*np.cos((hour-14)/12*np.pi)
        
        temperature = max(5, min(45, base_temp + variation))
        
        rain_prob = {1:0.02,2:0.03,3:0.05,4:0.08,5:0.15,6:0.35,7:0.42,8:0.40,9:0.28,10:0.12,11:0.04,12:0.02}[month]
        
        if 14 <= hour <= 18 and month in [6, 7, 8, 9]:
            rain_prob *= 1.5
        
        rain_flag = 1 if np.random.random() < rain_prob else 0
        
        return temperature, rain_flag
    

    def get_holiday_info(self, date_str, weekday_name):
        if date_str in self.holiday_rules:
            return self.holiday_rules[date_str]
        elif weekday_name in ['Saturday', 'Sunday']:
            return ("Weekend", 1.0, 1.0)  # Weekend gets neutral multiplier
        else:
            return ("Not a holiday", 1.0, 1.0)
    

    def get_holiday_multiplier(self, date_str, category, weekday_name):
        if date_str in self.holiday_rules:
            _, non_religious_mult, religious_mult = self.holiday_rules[date_str]
            return religious_mult if category == "temple" else non_religious_mult
        elif weekday_name in ['Saturday', 'Sunday']:
            return 1.2 if category == "temple" else 1.1  
        else:
            return 1.0
    
    def get_weather_multiplier(self, temperature, rain_flag, category, hour):
        weather_mult = 1.0
        
        if temperature > 38 and 12 <= hour <= 16 and category in self.outdoor_categories:
            weather_mult *= 0.88
        
        if rain_flag == 1 and category in self.outdoor_categories:
            weather_mult *= 0.70
            
        return weather_mult
    

    def prepare_model_input(self, place_name, target_datetime=None, target_hour=None, weather_api_key=None):
 
        if place_name not in self.places_data:
            raise ValueError(f"Place '{place_name}' not found. Available places: {list(self.places_data.keys())}")
        
        if target_datetime is None:
            target_datetime = datetime.now()
        
        if target_hour is None:
            current_hour = target_datetime.hour
            target_hour = max(7, min(20, current_hour))
        else:
            target_hour = max(7, min(20, target_hour))
        
        date_str = target_datetime.date().isoformat()
        weekday = target_datetime.weekday()
        weekday_name = target_datetime.strftime("%A")
        month = target_datetime.month
        
        place_info = self.places_data[place_name]
        category = place_info["category"]
        place_factor = place_info["place_factor"]
        
        category_base_value = self.category_base[category]
        base_factor = category_base_value * place_factor
        
        weekday_factor = self.weekly_factor[weekday]
        month_factor_value = self.month_factor[month]
        hourly_multiplier = self.hour_curve[category][target_hour]
        
        holiday_label, _, _ = self.get_holiday_info(date_str, weekday_name)
        holiday_multiplier = self.get_holiday_multiplier(date_str, category, weekday_name)
        
        temperature_c, rain_flag = self.get_current_weather(weather_api_key)
        weather_multiplier = self.get_weather_multiplier(temperature_c, rain_flag, category, target_hour)
        
        day_of_year = target_datetime.timetuple().tm_yday
        long_term_trend = 1.0 + 0.02*np.sin(day_of_year/365 * 2*np.pi)
        
        season = self.season_map[month]
        
        place_encoded = self.le_place.transform([place_name])[0]
        category_encoded = self.le_category.transform([category])[0]
        weekday_encoded = self.le_weekday.transform([weekday_name])[0]
        holiday_encoded = self.le_holiday.transform([holiday_label])[0]
        season_encoded = self.le_season.transform([season])[0]
        

        
        feature_vector = [
            target_hour,                
            place_encoded,             
            category_encoded,           
            weekday_encoded,            
            month,                      
            holiday_encoded,            
            round(temperature_c, 1),    
            rain_flag,                 
            category_base_value,        
            place_factor,               
            round(base_factor, 3),     
            weekday_factor,             
            month_factor_value,        
            hourly_multiplier,          
            holiday_multiplier,         
            weather_multiplier,         
            round(long_term_trend, 3),  
            season_encoded              
        ]
        
        return feature_vector
    
    def get_prediction_context(self, place_name, target_datetime=None, target_hour=None, weather_api_key=None):

        if target_datetime is None:
            target_datetime = datetime.now()
        
        if target_hour is None:
            current_hour = target_datetime.hour
            target_hour = max(7, min(20, current_hour))
        
        date_str = target_datetime.date().isoformat()
        weekday_name = target_datetime.strftime("%A")
        place_info = self.places_data[place_name]
        holiday_label, _, _ = self.get_holiday_info(date_str, weekday_name)
        season = self.season_map[target_datetime.month]
        temperature, rain_flag = self.get_current_weather(weather_api_key)
        
        context = {
            "place": place_name,
            "category": place_info["category"],
            "date": date_str,
            "time": f"{target_hour:02d}:00",
            "weekday": weekday_name,
            "season": season,
            "temperature": f"{temperature:.1f}°C",
            "rain_expected": "Yes" if rain_flag else "No",
            "holiday": holiday_label if holiday_label else "None",
        }
        
        return context


def main():
    
    processor = CrowdPredictionInputProcessor()
    
    API_KEY = "0203b89a4bf94e8ab5f220531252308" 
    

    place = "Hanuman Vatika"
    feature_vector = processor.prepare_model_input(place, weather_api_key=API_KEY)
    context = processor.get_prediction_context(place, weather_api_key=API_KEY)
    
    print(f"Place: {place}")
    print(f"Context: {context}")
    print(f"Feature Vector: {feature_vector}")
    print(f"Feature Vector Length: {len(feature_vector)}")
    
    feature_vector_fallback = processor.prepare_model_input(place)
    context_fallback = processor.get_prediction_context(place)
    
    print(f"Context: {context_fallback}")
    
    target_date = datetime(2025, 1, 26, 18, 0)  
    feature_vector_specific = processor.prepare_model_input(place, target_date, 18, API_KEY)
    context_specific = processor.get_prediction_context(place, target_date, 18, API_KEY)
    
    print(f"Context: {context_specific}")
    print(f"Feature Vector: {feature_vector_specific}")


if __name__ == "__main__":
    main()