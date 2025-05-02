from flask import Flask, render_template, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
from datetime import datetime
import crops
import random
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

CORS(app)  # Basic usage (allows all origins)
# OR for specific options:
CORS(app, resources={
    r"/ticker/*": {
        "origins": "*",          # Allow all origins
        "methods": ["GET"],      # Allow only GET requests
        "allow_headers": ["*"]   # Allow all headers
    }
})


# Define paths and constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "static")

commodity_dict = {
    "arhar": os.path.join(DATA_DIR, "Arhar.csv"),
    "bajra": os.path.join(DATA_DIR, "Bajra.csv"),
    "barley": os.path.join(DATA_DIR, "Barley.csv"),
    "copra": os.path.join(DATA_DIR, "Copra.csv"),
    "cotton": os.path.join(DATA_DIR, "Cotton.csv"),
    "sesamum": os.path.join(DATA_DIR, "Sesamum.csv"),
    "gram": os.path.join(DATA_DIR, "Gram.csv"),
    "groundnut": os.path.join(DATA_DIR, "Groundnut.csv"),
    "jowar": os.path.join(DATA_DIR, "Jowar.csv"),
    "maize": os.path.join(DATA_DIR, "Maize.csv"),
    "masoor": os.path.join(DATA_DIR, "Masoor.csv"),
    "moong": os.path.join(DATA_DIR, "Moong.csv"),
    "niger": os.path.join(DATA_DIR, "Niger.csv"),
    "paddy": os.path.join(DATA_DIR, "Paddy.csv"),
    "ragi": os.path.join(DATA_DIR, "Ragi.csv"),
    "rape": os.path.join(DATA_DIR, "Rape.csv"),
    "jute": os.path.join(DATA_DIR, "Jute.csv"),
    "safflower": os.path.join(DATA_DIR, "Safflower.csv"),
    "soyabean": os.path.join(DATA_DIR, "Soyabean.csv"),
    "sugarcane": os.path.join(DATA_DIR, "Sugarcane.csv"),
    "sunflower": os.path.join(DATA_DIR, "Sunflower.csv"),
    "urad": os.path.join(DATA_DIR, "Urad.csv"),
    "wheat": os.path.join(DATA_DIR, "Wheat.csv")
}

annual_rainfall = [29, 21, 37.5, 30.7, 52.6, 150, 299, 251.7, 179.2, 70.5, 39.8, 10.9]
base = {
    "Paddy": 1975,
    "Arhar": 3200,
    "Bajra": 1175,
    "Barley": 1980,
    "Copra": 5500,
    "Cotton": 3600,
    "Sesamum": 4200,
    "Gram": 2800,
    "Groundnut": 4900,
    "Jowar": 2120,
    "Maize": 2675,
    "Masoor": 2800,
    "Moong": 3500,
    "Niger": 3500,
    "Ragi": 1500,
    "Rape": 4800,
    "Jute": 1675,
    "Safflower": 2500,
    "Soyabean": 3800,
    "Sugarcane": 2250,
    "Sunflower": 3700,
    "Urad": 4300,
    "Wheat": 2650
}

commodity_list = []


class Commodity:
    def __init__(self, csv_name: str):
        self.name = csv_name
        try:
            dataset = pd.read_csv(csv_name)
            self.X = dataset.iloc[:, :-1].values
            self.Y = dataset.iloc[:, 3].values

            from sklearn.tree import DecisionTreeRegressor
            depth = random.randrange(7, 18)
            self.regressor = DecisionTreeRegressor(max_depth=depth)
            self.regressor.fit(self.X, self.Y)
        except Exception as e:
            logging.error(f"Error loading commodity data from {csv_name}: {e}")
            raise

    def getPredictedValue(self, value: list) -> float:
        if value[1] >= 2019:
            fsa = np.array(value).reshape(1, 3)
            return self.regressor.predict(fsa)[0]
        else:
            c = self.X[:, 0:2]
            x = [i.tolist() for i in c]
            fsa = [value[0], value[1]]
            try:
                ind = x.index(fsa)
                return self.Y[ind]
            except ValueError:
                logging.warning(f"Value {fsa} not found in dataset for {self.name}")
                return 0.0

    def getCropName(self) -> str:
        return os.path.splitext(os.path.basename(self.name))[0]


def TopFiveWinners():
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    prev_month = current_month - 1
    prev_rainfall = annual_rainfall[prev_month - 1]
    current_month_prediction = []
    prev_month_prediction = []
    change = []

    for i in commodity_list:
        current_predict = i.getPredictedValue([float(current_month), current_year, current_rainfall])
        current_month_prediction.append(current_predict)
        prev_predict = i.getPredictedValue([float(prev_month), current_year, prev_rainfall])
        prev_month_prediction.append(prev_predict)
        change.append((((current_predict - prev_predict) * 100 / prev_predict), commodity_list.index(i)))
    
    sorted_change = change
    sorted_change.sort(reverse=True)
    to_send = []
    for j in range(0, 5):
        perc, i = sorted_change[j]
        # Use the full crop name without splitting
        name = commodity_list[i].getCropName()
        to_send.append([name, round((current_month_prediction[i] * base[name]) / 100, 2), round(perc, 2)])
    
    return to_send


def TopFiveLosers():
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    prev_month = current_month - 1
    prev_rainfall = annual_rainfall[prev_month - 1]
    current_month_prediction = []
    prev_month_prediction = []
    change = []

    for i in commodity_list:
        current_predict = i.getPredictedValue([float(current_month), current_year, current_rainfall])
        current_month_prediction.append(current_predict)
        prev_predict = i.getPredictedValue([float(prev_month), current_year, prev_rainfall])
        prev_month_prediction.append(prev_predict)
        change.append((((current_predict - prev_predict) * 100 / prev_predict), commodity_list.index(i)))
    
    sorted_change = change
    sorted_change.sort()
    to_send = []
    for j in range(0, 5):
        perc, i = sorted_change[j]
        # Use the full crop name without splitting
        name = commodity_list[i].getCropName()
        to_send.append([name, round((current_month_prediction[i] * base[name]) / 100, 2), round(perc, 2)])
    
    return to_send


def SixMonthsForecast():
    month1 = []
    month2 = []
    month3 = []
    month4 = []
    month5 = []
    month6 = []
    for i in commodity_list:
        crop = SixMonthsForecastHelper(i.getCropName())
        k = 0
        for j in crop:
            time = j[0]
            price = j[1]
            change = j[2]
            if k == 0:
                month1.append((price, change, i.getCropName(), time))
            elif k == 1:
                month2.append((price, change, i.getCropName(), time))
            elif k == 2:
                month3.append((price, change, i.getCropName(), time))
            elif k == 3:
                month4.append((price, change, i.getCropName(), time))
            elif k == 4:
                month5.append((price, change, i.getCropName(), time))
            elif k == 5:
                month6.append((price, change, i.getCropName(), time))
            k += 1
    month1.sort()
    month2.sort()
    month3.sort()
    month4.sort()
    month5.sort()
    month6.sort()
    crop_month_wise = []
    crop_month_wise.append([month1[0][3], month1[len(month1) - 1][2], month1[len(month1) - 1][0], month1[len(month1) - 1][1], month1[0][2], month1[0][0], month1[0][1]])
    crop_month_wise.append([month2[0][3], month2[len(month2) - 1][2], month2[len(month2) - 1][0], month2[len(month2) - 1][1], month2[0][2], month2[0][0], month2[0][1]])
    crop_month_wise.append([month3[0][3], month3[len(month3) - 1][2], month3[len(month3) - 1][0], month3[len(month3) - 1][1], month3[0][2], month3[0][0], month3[0][1]])
    crop_month_wise.append([month4[0][3], month4[len(month4) - 1][2], month4[len(month4) - 1][0], month4[len(month4) - 1][1], month4[0][2], month4[0][0], month4[0][1]])
    crop_month_wise.append([month5[0][3], month5[len(month5) - 1][2], month5[len(month5) - 1][0], month5[len(month5) - 1][1], month5[0][2], month5[0][0], month5[0][1]])
    crop_month_wise.append([month6[0][3], month6[len(month6) - 1][2], month6[len(month6) - 1][0], month6[len(month6) - 1][1], month6[0][2], month6[0][0], month6[0][1]])
    return crop_month_wise

# def SixMonthsForecastHelper(name):
#     current_month = datetime.now().month
#     current_year = datetime.now().year
#     name = name.lower()
#     commodity = None

#     for i in commodity_list:
#         if name == i.getCropName().lower():
#             commodity = i
#             break

#     if commodity is None:
#         logging.warning(f"Commodity '{name}' not found for forecast helper.")
#         return []

#     result = []

#     for offset in range(1, 7):  # Next 6 months
#         month = current_month + offset
#         year = current_year
#         if month > 12:
#             month -= 12
#             year += 1

#         rainfall = annual_rainfall[month - 1]
#         predicted_price = commodity.getPredictedValue([float(month), year, rainfall])

#         prev_month = month - 1
#         prev_year = year
#         if prev_month == 0:
#             prev_month = 12
#             prev_year -= 1
#         prev_rainfall = annual_rainfall[prev_month - 1]
#         previous_price = commodity.getPredictedValue([float(prev_month), prev_year, prev_rainfall])

#         if previous_price != 0:
#             change = ((predicted_price - previous_price) * 100) / previous_price
#         else:
#             change = 0

#         result.append([f"{month:02d}-{year}", round((predicted_price * base[commodity.getCropName()]) / 100, 2), round(change, 2)])

#     return result

def TwelveMonthsForecast(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    name = name.lower()
    commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 13):
        if current_month + i <= 12:
            month_with_year.append((current_month + i, current_year, annual_rainfall[current_month + i - 1]))
        else:
            month_with_year.append((current_month + i - 12, current_year + 1, annual_rainfall[current_month + i - 13]))
    max_index = 0
    min_index = 0
    max_value = 0
    min_value = 9999
    wpis = []
    current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
    change = []

    for m, y, r in month_with_year:
        current_predict = commodity.getPredictedValue([float(m), y, r])
        if current_predict > max_value:
            max_value = current_predict
            max_index = month_with_year.index((m, y, r))
        if current_predict < min_value:
            min_value = current_predict
            min_index = month_with_year.index((m, y, r))
        wpis.append(current_predict)
        change.append(((current_predict - current_wpi) * 100) / current_wpi)

    max_month, max_year, r1 = month_with_year[max_index]
    min_month, min_year, r2 = month_with_year[min_index]
    min_value = min_value * base[name.capitalize()] / 100
    max_value = max_value * base[name.capitalize()] / 100
    crop_price = []
    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y, m, 1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i] * base[name.capitalize()]) / 100, 2), round(change[i], 2)])
    x = datetime(max_year, max_month, 1)
    x = x.strftime("%b %y")
    max_crop = [x, round(max_value, 2)]
    x = datetime(min_year, min_month, 1)
    x = x.strftime("%b %y")
    min_crop = [x, round(min_value, 2)]

    return max_crop, min_crop, crop_price

def TwelveMonthPrevious(name):
    name = name.lower()
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    commodity = commodity_list[0]
    wpis = []
    crop_price = []
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    month_with_year = []
    for i in range(1, 13):
        if current_month - i >= 1:
            month_with_year.append((current_month - i, current_year, annual_rainfall[current_month - i - 1]))
        else:
            month_with_year.append((current_month - i + 12, current_year - 1, annual_rainfall[current_month - i + 11]))

    for m, y, r in month_with_year:
        current_predict = commodity.getPredictedValue([float(m), 2013, r])
        wpis.append(current_predict)

    for i in range(0, len(wpis)):
        m, y, r = month_with_year[i]
        x = datetime(y, m, 1)
        x = x.strftime("%b %y")
        crop_price.append([x, round((wpis[i] * base[name.capitalize()]) / 100, 2)])
    new_crop_price = []
    for i in range(len(crop_price) - 1, -1, -1):
        new_crop_price.append(crop_price[i])
    return new_crop_price

def CurrentMonth(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    name = name.lower()
    commodity = commodity_list[0]
    for i in commodity_list:
        if name == str(i):
            commodity = i
            break
    current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
    current_price = (base[name.capitalize()] * current_wpi) / 100
    return current_price

@app.route('/')
def index():
    context = {
        "top5": TopFiveWinners(),
        "bottom5": TopFiveLosers(),
        "sixmonths": SixMonthsForecast()
    }
    return render_template('index.html', context=context)


@app.route("/explore")
def explore():
    return render_template("explore.html")

@app.route('/ticker/<item>/<number>')
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def ticker(item, number):
    try:
        n = int(number)
        i = int(item)
        data = SixMonthsForecast()
        if n < len(data) and i < len(data[n]):
            context = str(data[n][i])
            if i == 2 or i == 5:
                context = '₹' + context
            elif i == 3 or i == 6:
                context = context + '%'
            return context
        else:
            return "Invalid index", 404
    except Exception as e:
        logging.error(f"Error in ticker route: {e}")
        return "Internal Server Error", 500


@app.route('/commodity/<name>')
def crop_profile(name):
    try:
        max_crop, min_crop, forecast_crop_values = TwelveMonthsForecast(name)
        prev_crop_values = TwelveMonthPrevious(name)
        forecast_x = [i[0] for i in forecast_crop_values]
        forecast_y = [i[1] for i in forecast_crop_values]
        previous_x = [i[0] for i in prev_crop_values]
        previous_y = [i[1] for i in prev_crop_values]
        current_price = CurrentMonth(name)


        crop_data = crops.crop(name)
        context = {
            "name": name,
            "max_crop": max_crop,
            "min_crop": min_crop,
            "forecast_values": forecast_crop_values,
            "forecast_x": str(forecast_x),
            "forecast_y": forecast_y,
            "previous_values": prev_crop_values,
            "previous_x": previous_x,
            "previous_y": previous_y,
            "current_price": current_price,
            "image_url": crop_data[0],
            "prime_loc": crop_data[1],
            "type_c": crop_data[2],
            "export": crop_data[3]
        }
        return render_template('commodity.html', context=context)
    except Exception as e:
        logging.error(f"Error in crop_profile for {name}: {e}")
        return render_template('error.html', error=str(e)), 500



if __name__ == "__main__":
    # Load commodities dynamically
    for key, path in commodity_dict.items():
        try:
            commodity = Commodity(path)
            commodity_list.append(commodity)
        except Exception as e:
            logging.error(f"Failed to load commodity {key} from {path}: {e}")

    app.run(debug=True)
